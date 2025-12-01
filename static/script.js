// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const productIdInput = document.getElementById('productId');
const searchBtn = document.getElementById('searchBtn');
const numResultsSelect = document.getElementById('numResults');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultProductId = document.getElementById('resultProductId');
const totalRecommendations = document.getElementById('totalRecommendations');

// Model result containers
const hybridResults = document.getElementById('hybridResults');
const alsResults = document.getElementById('alsResults');
const contentResults = document.getElementById('contentResults');
const popularityResults = document.getElementById('popularityResults');
// const similarProducts = document.getElementById('similarProducts');  // COMMENTED OUT

// Allow Enter key to trigger search
productIdInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        searchRecommendations();
    }
});

/**
 * Main function to search for recommendations
 * Loads hybrid first for best performance, then other methods if enabled
 */
async function searchRecommendations() {
    const productId = parseInt(productIdInput.value);
    const numResults = parseInt(numResultsSelect.value);

    // Validation
    if (!productId || productId < 1) {
        showError('Please enter a valid product ID (must be a positive number)');
        return;
    }

    // Hide previous results and errors
    hideError();
    hideResults();
    showLoading();

    try {
        // Step 1: Load Hybrid first (main recommendation) - Similar Products commented out for performance
        const hybridData = await fetchRecommendations(productId, numResults, 'hybrid');
        // const similarData = await fetchSimilarProducts(productId, 10);  // COMMENTED OUT

        // Display hybrid results immediately
        displayHybridResults(productId, hybridData, null);  // Pass null for similarData
        showResults();

        // Step 2: Load other methods if toggle is enabled (in background)
        const showOther = document.getElementById('showOtherMethods')?.checked;
        if (showOther) {
            loadOtherMethods(productId, numResults);
        }

    } catch (error) {
        console.error('Error fetching recommendations:', error);
        showError(`Failed to fetch recommendations: ${error.message}`);
        hideLoading();
    } finally {
        hideLoading();
    }
}

/**
 * Load other methods in background (non-blocking)
 */
async function loadOtherMethods(productId, numResults) {
    try {
        const [alsData, contentData, popularityData] = await Promise.all([
            fetchRecommendations(productId, numResults, 'als'),
            fetchRecommendations(productId, numResults, 'content'),
            fetchRecommendations(productId, numResults, 'popularity')
        ]);

        // Display other methods
        displayModelResults('als', alsData, alsResults);
        displayModelResults('content', contentData, contentResults);
        displayModelResults('popularity', popularityData, popularityResults);
    } catch (error) {
        console.error('Error loading other methods:', error);
    }
}

/**
 * Display hybrid results immediately
 */
function displayHybridResults(productId, hybridData, similarData) {
    resultProductId.textContent = productId;
    
    // Display hybrid results
    displayModelResults('hybrid', hybridData, hybridResults);
    
    // ## Display similar products - COMMENTED OUT FOR PERFORMANCE
    // displaySimilarProducts(similarData);
    
    // Update stats
    const total = hybridData?.total_recommendations || 0;
    totalRecommendations.textContent = `${total} recommendations`;
}

/**
 * Fetch recommendations from API
 */
async function fetchRecommendations(productId, n, method) {
    try {
        const response = await fetch(`${API_BASE_URL}/recommendations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                product_id: productId,
                n: n,
                method: method
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${method} recommendations:`, error);
        return null;
    }
}

// ## Fetch similar products from API - COMMENTED OUT FOR PERFORMANCE
// async function fetchSimilarProducts(productId, n) {
//     try {
//         const response = await fetch(`${API_BASE_URL}/similar-products`, {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({
//                 product_id: productId,
//                 n: n
//             })
//         });
//
//         if (!response.ok) {
//             const errorData = await response.json();
//             throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
//         }
//
//         return await response.json();
//     } catch (error) {
//         console.error('Error fetching similar products:', error);
//         return null;
//     }
// }

/**
 * Toggle other methods visibility
 */
function toggleOtherMethods() {
    const checkbox = document.getElementById('showOtherMethods');
    const otherSection = document.getElementById('otherMethodsSection');
    
    if (checkbox.checked) {
        otherSection.style.display = 'block';
        // Load other methods if not already loaded
        const productId = parseInt(productIdInput.value);
        const numResults = parseInt(numResultsSelect.value);
        if (productId && productId > 0) {
            loadOtherMethods(productId, numResults);
        }
    } else {
        otherSection.style.display = 'none';
    }
}

/**
 * Display recommendations for a specific model
 */
function displayModelResults(modelName, data, container) {
    if (!data || !data.recommendations || data.recommendations.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p style="font-weight: 600; margin-bottom: 0.5rem;">No recommendations available</p>
                <p style="font-size: 0.875rem; opacity: 0.7;">
                    ${data ? 'This product may not have enough interaction data' : 'Failed to fetch recommendations'}
                </p>
            </div>
        `;
        return;
    }

        const recommendations = data.recommendations;
    container.innerHTML = recommendations.map((rec, index) => {
        // Remove comma formatting - show user ID as plain number
        const userId = rec.user_id.toString();
        // ## User history link commented out - just show user ID
        return `
        <div class="recommendation-item" data-index="${index}">
            <span class="rank">#${index + 1}</span>
            <div class="user-info">
                <span class="user-id-link" style="cursor: default; text-decoration: none;">
                    User ID: ${userId}
                </span>
                <!-- <a href="/static/user-history.html?user_id=${userId}" class="user-id-link" onclick="handleUserClick(event, ${userId})">
                    User ID: ${userId}
                </a> -->
            </div>
            <div class="score">${rec.score.toFixed(4)}</div>
        </div>
    `;
    }).join('');
    
    // Add animation to items
    animateRecommendations(container);
}

// ## Display similar products - COMMENTED OUT FOR PERFORMANCE
// function displaySimilarProducts(data) {
//     if (!data || !data.similar_products || data.similar_products.length === 0) {
//         similarProducts.innerHTML = `
//             <div class="empty-state">
//                 <div class="empty-state-icon">🔍</div>
//                 <p>No similar products found</p>
//             </div>
//         `;
//         return;
//     }
//
//     const products = data.similar_products;
//     
//     // Fix similarity score calculation
//     // The API returns distance, we need to convert it properly
//     // Lower distance = higher similarity
//     // Normalize to 0-1 range, then convert to percentage
//     const maxDistance = Math.max(...products.map(p => p.distance));
//     const minDistance = Math.min(...products.map(p => p.distance));
//     const range = maxDistance - minDistance || 1; // Avoid division by zero
//     
//     similarProducts.innerHTML = products.map((prod, index) => {
//         // Calculate normalized similarity (invert distance, normalize to 0-1)
//         // If distance is 0, similarity is 100%, otherwise use inverse
//         let similarityPercent;
//         if (prod.distance === 0) {
//             similarityPercent = 100;
//         } else {
//             // Normalize: (max - current) / range, then scale to percentage
//             const normalized = (maxDistance - prod.distance) / range;
//             // Scale to 0-100%, with minimum of 0%
//             similarityPercent = Math.max(0, Math.min(100, normalized * 100));
//         }
//         
//         return `
//         <div class="similar-product-item" data-index="${index}">
//             <div class="product-id">${prod.product_id}</div>
//             <div class="similarity">Similarity</div>
//             <div class="similarity-score">${similarityPercent.toFixed(1)}%</div>
//             <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;">
//                 Distance: ${prod.distance.toFixed(4)}
//             </div>
//         </div>
//     `;
//     }).join('');
//     
//     // Add animation
//     animateSimilarProducts();
// }

/**
 * Show error message
 */
function showError(message) {
    errorMessage.textContent = `❌ ${message}`;
    errorMessage.style.display = 'flex';
}

/**
 * Hide error message
 */
function hideError() {
    errorMessage.style.display = 'none';
}

/**
 * Show results section
 */
function showResults() {
    resultsSection.style.display = 'block';
}

/**
 * Hide results section
 */
function hideResults() {
    resultsSection.style.display = 'none';
}

/**
 * Show loading spinner
 */
function showLoading() {
    loadingSpinner.style.display = 'block';
    searchBtn.disabled = true;
    searchBtn.querySelector('.btn-text').style.display = 'none';
    searchBtn.querySelector('.btn-loader').style.display = 'inline';
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    loadingSpinner.style.display = 'none';
    searchBtn.disabled = false;
    searchBtn.querySelector('.btn-text').style.display = 'inline';
    searchBtn.querySelector('.btn-loader').style.display = 'none';
}

// ## Handle user ID click - navigate to user history - COMMENTED OUT FOR PERFORMANCE
// function handleUserClick(event, userId) {
//     event.preventDefault();
//     // Store user ID in sessionStorage and navigate
//     sessionStorage.setItem('selectedUserId', userId);
//     window.location.href = `/static/user-history.html?user_id=${userId}`;
// }

// Temporary: Just prevent navigation
function handleUserClick(event, userId) {
    event.preventDefault();
    // User history feature is temporarily disabled for performance
    console.log(`User history for ${userId} is temporarily disabled`);
}

/**
 * Animate recommendation items
 */
function animateRecommendations(container) {
    const items = container.querySelectorAll('.recommendation-item');
    items.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateX(-20px)';
        setTimeout(() => {
            item.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            item.style.opacity = '1';
            item.style.transform = 'translateX(0)';
        }, index * 40);
    });
}

// ## Animate similar products - COMMENTED OUT FOR PERFORMANCE
// function animateSimilarProducts() {
//     const items = document.querySelectorAll('.similar-product-item');
//     items.forEach((item, index) => {
//         item.style.opacity = '0';
//         item.style.transform = 'scale(0.9) translateY(10px)';
//         setTimeout(() => {
//             item.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
//             item.style.opacity = '1';
//             item.style.transform = 'scale(1) translateY(0)';
//         }, index * 80);
//     });
// }

/**
 * Format number with commas (not used for user IDs anymore)
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Check API health on page load
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('API is healthy:', data);
        } else {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('Cannot connect to API. Make sure it is running on', API_BASE_URL);
        showError('Cannot connect to API. Please make sure the FastAPI server is running on ' + API_BASE_URL);
    }
}

// Check API health when page loads
window.addEventListener('load', checkAPIHealth);

