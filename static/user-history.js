// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Get user ID from URL parameter
const urlParams = new URLSearchParams(window.location.search);
const userId = urlParams.get('user_id') || sessionStorage.getItem('selectedUserId');

// DOM Elements
const userIdDisplay = document.getElementById('userIdDisplay');
const errorMessage = document.getElementById('errorMessage');
const historySection = document.getElementById('historySection');
const historyList = document.getElementById('historyList');
const loadingSpinner = document.getElementById('loadingSpinner');
const userStats = document.getElementById('userStats');
const eventFilter = document.getElementById('eventFilter');

// Store all history data
let allHistoryData = [];

/**
 * Initialize page
 */
async function init() {
    if (!userId) {
        showError('No user ID provided');
        return;
    }

    userIdDisplay.textContent = userId;
    await loadUserHistory(userId);
}

/**
 * Load user history from API
 */
async function loadUserHistory(userId) {
    showLoading();
    hideError();
    hideHistory();

    try {
        // Note: You'll need to add a user history endpoint to your FastAPI
        // For now, we'll use a placeholder approach
        const response = await fetch(`${API_BASE_URL}/user-history/${userId}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('User history not found. This user may not exist in the dataset.');
            }
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        allHistoryData = data.history || [];
        
        displayUserStats(data);
        displayHistory(allHistoryData);
        showHistory();

    } catch (error) {
        console.error('Error loading user history:', error);
        showError(`Failed to load user history: ${error.message}`);
    } finally {
        hideLoading();
    }
}

/**
 * Display user statistics
 */
function displayUserStats(data) {
    const stats = data.stats || {};
    document.getElementById('totalInteractions').textContent = stats.total_interactions || 0;
    document.getElementById('uniqueProducts').textContent = stats.unique_products || 0;
    document.getElementById('eventTypes').textContent = stats.event_types || 0;
    userStats.style.display = 'block';
}

/**
 * Display history list
 */
function displayHistory(historyData) {
    if (!historyData || historyData.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p>No interaction history found for this user</p>
            </div>
        `;
        return;
    }

    historyList.innerHTML = historyData.map((item, index) => {
        const event = item.event || 'interaction';
        const eventIcon = getEventIcon(event);
        const eventColor = getEventColor(event);
        
        return `
            <div class="history-item" data-index="${index}" data-event="${event}">
                <div class="history-item-header">
                    <span class="event-badge" style="background: ${eventColor}">
                        ${eventIcon} ${event.charAt(0).toUpperCase() + event.slice(1)}
                    </span>
                    ${item.event_date ? `
                        <span class="history-date">${formatDate(item.event_date)}</span>
                    ` : `
                        <span class="history-date">Score: ${item.score ? item.score.toFixed(4) : 'N/A'}</span>
                    `}
                </div>
                <div class="history-item-body">
                    <div class="history-product">
                        <strong>Product ID:</strong> ${item.product_id}
                    </div>
                    ${item.product_name ? `
                        <div class="history-product-name">
                            ${item.product_name}
                        </div>
                    ` : ''}
                    ${item.score && !item.event_date ? `
                        <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem;">
                            Interaction Score: <strong>${item.score.toFixed(4)}</strong>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
    
    // Animate items
    animateHistoryItems();
}

/**
 * Filter history by event type
 */
function filterHistory() {
    const selectedEvent = eventFilter.value;
    
    if (selectedEvent === 'all') {
        displayHistory(allHistoryData);
    } else {
        const filtered = allHistoryData.filter(item => {
            const event = (item.event || 'interaction').toLowerCase();
            return event === selectedEvent.toLowerCase();
        });
        displayHistory(filtered);
    }
}

/**
 * Get event icon
 */
function getEventIcon(event) {
    const icons = {
        'purchased': '🛒',
        'cart': '🛍️',
        'rating': '⭐',
        'wishlist': '❤️',
        'search_keyword': '🔍'
    };
    return icons[event.toLowerCase()] || '📋';
}

/**
 * Get event color
 */
function getEventColor(event) {
    const colors = {
        'purchased': '#10b981',
        'cart': '#3b82f6',
        'rating': '#f59e0b',
        'wishlist': '#ef4444',
        'search_keyword': '#8b5cf6'
    };
    return colors[event.toLowerCase()] || '#64748b';
}

/**
 * Format date
 */
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Animate history items
 */
function animateHistoryItems() {
    const items = document.querySelectorAll('.history-item');
    items.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(10px)';
        setTimeout(() => {
            item.style.transition = 'all 0.3s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, index * 30);
    });
}

/**
 * Go back to main page
 */
function goBack() {
    window.location.href = '/';
}

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
 * Show history section
 */
function showHistory() {
    historySection.style.display = 'block';
}

/**
 * Hide history section
 */
function hideHistory() {
    historySection.style.display = 'none';
}

/**
 * Show loading spinner
 */
function showLoading() {
    loadingSpinner.style.display = 'block';
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    loadingSpinner.style.display = 'none';
}

// Initialize when page loads
window.addEventListener('load', init);

