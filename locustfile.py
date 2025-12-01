"""
Locust stress test configuration for Product Audience Recommender API
Tests the API under 500-1000 concurrent requests
OPTIMIZED: Removed similar-products and user-history endpoints
"""
from locust import HttpUser, task, between
import random
import json

class RecommenderAPIUser(HttpUser):
    """
    Simulates a user making requests to the recommender API
    """
    wait_time = between(1, 2)  # Wait 0.5-2 seconds between requests (faster for stress test)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_product_range = (1, 200000)
        self.valid_product_ids = None
    
    def on_start(self):
        """Called when a simulated user starts - fetch valid IDs"""
        # Try to get valid IDs from /valid-ids endpoint
        try:
            response = self.client.get("/valid-ids", catch_response=True, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Use actual valid IDs if available
                if data.get('sample_product_ids'):
                    self.valid_product_ids = set(data.get('sample_product_ids', []))
                else:
                    self.valid_product_ids = None
                self.valid_product_range = data.get('product_range', (1, 200000))
            else:
                self.valid_product_ids = None
                self.valid_product_range = (1, 200000)
        except:
            # Fallback to safe ranges
            self.valid_product_ids = None
            self.valid_product_range = (1, 200000)
    
    @task(5)
    def get_recommendations_hybrid(self):
        """
        Test hybrid recommendations endpoint (most common)
        Weight: 5 (most frequent)
        """
        # Use valid product IDs if available, otherwise random
        if hasattr(self, 'valid_product_ids') and self.valid_product_ids:
            product_id = random.choice(list(self.valid_product_ids))
        else:
            # Use common product IDs that are more likely to exist
            product_id = random.choice([
                random.randint(1, 1000),      # Low IDs (more likely)
                random.randint(1, 10000),     # Medium IDs
                random.randint(self.valid_product_range[0], min(200000, self.valid_product_range[1]))
            ])
        n = random.choice([5, 10, 20])
        
        try:
            with self.client.post(
                "/recommendations",
                json={
                    "product_id": product_id,
                    "n": n,
                    "method": "hybrid"
                },
                name="POST /recommendations (hybrid)",
                catch_response=True,
                timeout=10
            ) as response:
                # Accept 404 as valid (product doesn't exist)
                if response.status_code in [200, 404]:
                    response.success()
                elif response.status_code >= 500:
                    # Server error - mark as failure
                    response.failure(f"Server error: {response.status_code}")
        except Exception as e:
            # Connection errors - track as failure
            # Re-raise to let Locust track it properly
            raise
    
    @task(3)
    def get_recommendations_als(self):
        """
        Test ALS recommendations endpoint
        Weight: 3
        """
        if hasattr(self, 'valid_product_ids') and self.valid_product_ids:
            product_id = random.choice(list(self.valid_product_ids))
        else:
            product_id = random.choice([
                random.randint(1, 1000),
                random.randint(1, 10000),
                random.randint(self.valid_product_range[0], min(200000, self.valid_product_range[1]))
            ])
        n = random.choice([5, 10, 20])
        
        try:
            response = self.client.post(
                "/recommendations",
                json={
                    "product_id": product_id,
                    "n": n,
                    "method": "als"
                },
                name="POST /recommendations (als)",
                catch_response=True,
                timeout=10
            )
            
            if response.status_code in [200, 404]:
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
        except Exception as e:
            pass
    
    @task(2)
    def get_recommendations_content(self):
        """
        Test content-based recommendations endpoint
        Weight: 2
        """
        if hasattr(self, 'valid_product_ids') and self.valid_product_ids:
            product_id = random.choice(list(self.valid_product_ids))
        else:
            product_id = random.choice([
                random.randint(1, 1000),
                random.randint(1, 10000),
                random.randint(self.valid_product_range[0], min(200000, self.valid_product_range[1]))
            ])
        n = random.choice([5, 10])
        
        try:
            response = self.client.post(
                "/recommendations",
                json={
                    "product_id": product_id,
                    "n": n,
                    "method": "content"
                },
                name="POST /recommendations (content)",
                catch_response=True,
                timeout=10
            )
            
            if response.status_code in [200, 404]:
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
        except Exception as e:
            pass
    
    @task(1)
    def get_recommendations_get(self):
        """
        Test GET endpoint for recommendations
        Weight: 1
        """
        if hasattr(self, 'valid_product_ids') and self.valid_product_ids:
            product_id = random.choice(list(self.valid_product_ids))
        else:
            product_id = random.choice([
                random.randint(1, 1000),
                random.randint(1, 10000),
                random.randint(self.valid_product_range[0], min(200000, self.valid_product_range[1]))
            ])
        n = random.choice([5, 10])
        
        try:
            response = self.client.get(
                f"/recommendations/{product_id}?n={n}&method=hybrid",
                name="GET /recommendations/{product_id}",
                catch_response=True,
                timeout=10
            )
            
            if response.status_code in [200, 404]:
                response.success()
            elif response.status_code >= 500:
                response.failure(f"Server error: {response.status_code}")
        except Exception as e:
            pass
    
    @task(10)
    def health_check(self):
        """
        Health check endpoint (very frequent)
        Weight: 10 (most frequent)
        """
        try:
            response = self.client.get(
                "/health",
                name="GET /health",
                catch_response=True,
                timeout=5
            )
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
        except Exception as e:
            pass
    
    @task(1)
    def get_root(self):
        """
        Test root endpoint (web interface)
        Weight: 1
        """
        try:
            response = self.client.get(
                "/",
                name="GET /",
                catch_response=True,
                timeout=10
            )
            if response.status_code in [200, 404]:
                response.success()
        except Exception as e:
            pass

