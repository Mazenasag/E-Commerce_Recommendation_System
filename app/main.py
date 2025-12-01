"""
FastAPI Application for E-Commerce Recommendation System
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import os
import logging

from app.services.recommender_service import RecommenderService
from app.middleware import PerformanceMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Recommendation API",
    description="API for product-to-user recommendations using collaborative filtering and content-based methods",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance middleware
app.add_middleware(PerformanceMiddleware)

# Initialize recommender service (loads artifacts on startup)
recommender_service = RecommenderService()

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    product_id: int = Field(..., description="Product ID to get recommendations for")
    n: int = Field(10, ge=1, le=100, description="Number of users to recommend")
    method: str = Field("hybrid", description="Recommendation method: 'als', 'content', 'hybrid', or 'popularity'")

class RecommendationResponse(BaseModel):
    product_id: int
    method: str
    recommendations: List[dict] = Field(..., description="List of recommended users with scores")
    total_recommendations: int

class SimilarProductRequest(BaseModel):
    product_id: int = Field(..., description="Product ID to find similar products for")
    n: int = Field(10, ge=1, le=50, description="Number of similar products to return")

class SimilarProductResponse(BaseModel):
    product_id: int
    similar_products: List[dict] = Field(..., description="List of similar products with similarity scores")
    total_similar: int

class HealthResponse(BaseModel):
    status: str
    message: str
    stats: dict

# Global cached stats for health check (computed once at startup)
_health_stats = None

async def cache_health_stats():
    """Cache health stats at startup"""
    global _health_stats
    try:
        _health_stats = recommender_service.get_stats()
        print("✅ Health stats cached")
    except:
        pass

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load all artifacts on startup"""
    try:
        recommender_service.load_artifacts()
        print("✅ All artifacts loaded successfully!")
        # Cache health stats after loading
        await cache_health_stats()
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise

# Health check endpoint (ultra-fast - returns cached stats)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - ultra-fast response with cached stats"""
    global _health_stats
    try:
        # Use cached stats (computed at startup)
        if _health_stats is None:
            _health_stats = recommender_service.get_stats()
        
        return HealthResponse(
            status="healthy",
            message="Recommendation service is running",
            stats=_health_stats
        )
    except Exception as e:
        # Fallback to basic response if stats fail
        return HealthResponse(
            status="healthy",
            message="Recommendation service is running",
            stats={"loaded": recommender_service.loaded}
        )

# Get recommendations for a product
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get top-N user recommendations for a given product.
    
    Methods:
    - 'als': Collaborative filtering (warm users only)
    - 'content': Content-based recommendations
    - 'hybrid': Combined ALS + content-based
    - 'popularity': Popularity-based baseline
    """
    try:
        # Fast validation before processing (O(1) lookup)
        if request.product_id not in recommender_service._valid_product_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Product ID {request.product_id} not found in the system."
            )
        
        # Get recommendations (with internal error handling)
        recommendations = recommender_service.get_recommendations(
            product_id=request.product_id,
            n=request.n,
            method=request.method
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for product_id={request.product_id}. Product exists but has insufficient interaction data."
            )
        
        return RecommendationResponse(
            product_id=request.product_id,
            method=request.method,
            recommendations=recommendations,
            total_recommendations=len(recommendations)
        )
    except HTTPException:
        raise
    except Exception as e:
        # Log error but don't expose internal details
        logger.error(f"Error in get_recommendations for product {request.product_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing recommendations. Please try again."
        )

# ## Get similar products - COMMENTED OUT FOR PERFORMANCE
# @app.post("/similar-products", response_model=SimilarProductResponse)
# async def get_similar_products(request: SimilarProductRequest):
#     """
#     Get top-N similar products based on product name embeddings.
#     """
#     try:
#         # Fast validation
#         if request.product_id not in recommender_service._valid_product_ids:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Product ID {request.product_id} not found in the system."
#             )
#         
#         similar_products = recommender_service.get_similar_products(
#             product_id=request.product_id,
#             n=request.n
#         )
#         
#         if not similar_products:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No similar products found for product_id={request.product_id}. Product exists but has no embeddings."
#             )
#         
#         return SimilarProductResponse(
#             product_id=request.product_id,
#             similar_products=similar_products,
#             total_similar=len(similar_products)
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         print(f"Error in get_similar_products: {e}")
#         print(traceback.format_exc())
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error while finding similar products. Please try again."
#         )

# Get recommendations (GET endpoint for convenience)
@app.get("/recommendations/{product_id}")
async def get_recommendations_get(
    product_id: int,
    n: int = Query(10, ge=1, le=100),
    method: str = Query("hybrid", regex="^(als|content|hybrid|popularity)$")
):
    """Get recommendations via GET request"""
    request = RecommendationRequest(product_id=product_id, n=n, method=method)
    return await get_recommendations(request)

# ## Get similar products (GET endpoint) - COMMENTED OUT FOR PERFORMANCE
# @app.get("/similar-products/{product_id}")
# async def get_similar_products_get(
#     product_id: int,
#     n: int = Query(10, ge=1, le=50)
# ):
#     """Get similar products via GET request"""
#     request = SimilarProductRequest(product_id=product_id, n=n)
#     return await get_similar_products(request)

# Get valid IDs endpoint (for testing)
@app.get("/valid-ids")
async def get_valid_ids():
    """Get sample valid product and user IDs for testing"""
    try:
        # Get sample valid IDs (cached)
        if not hasattr(get_valid_ids, '_cached_ids'):
            sample_products = sorted(list(recommender_service._valid_product_ids))[:100]
            sample_users = sorted(list(recommender_service._valid_user_ids))[:100]
            get_valid_ids._cached_ids = {
                "total_products": len(recommender_service._valid_product_ids),
                "total_users": len(recommender_service._valid_user_ids),
                "sample_product_ids": sample_products,
                "sample_user_ids": sample_users,
                "product_range": (min(sample_products), max(sample_products)) if sample_products else None,
                "user_range": (min(sample_users), max(sample_users)) if sample_users else None
            }
        return get_valid_ids._cached_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting valid IDs: {str(e)}")

# ## User History endpoint - COMMENTED OUT FOR PERFORMANCE
# @app.get("/user-history/{user_id}")
# async def get_user_history(user_id: int):
#     """
#     Get interaction history for a specific user.
#     Returns all products the user has interacted with.
#     """
#     try:
#         # Fast validation
#         if user_id not in recommender_service._valid_user_ids:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"User ID {user_id} not found in the system."
#             )
#         
#         history = recommender_service.get_user_history(user_id)
#         
#         if not history:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"User ID {user_id} exists but has no interaction history."
#             )
#         
#         return history
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         print(f"Error in get_user_history: {e}")
#         print(traceback.format_exc())
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error while retrieving user history. Please try again."
#         )

# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Root endpoint - serve the HTML page
@app.get("/")
async def root():
    """Root endpoint - serves the HTML interface"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {
            "message": "E-Commerce Recommendation API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "recommendations": "/recommendations/{product_id}",
                "similar_products": "/similar-products/{product_id}",
                "docs": "/docs"
            },
            "note": "Static files not found. Place HTML/CSS/JS files in the 'static' directory."
        }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

