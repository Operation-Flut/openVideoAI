/**
 * Model Manager UI Logic for openVideoAI
 * Handles model listing, downloading, deletion, and detail views
 */

// API base URL
const API_BASE = '/api';

// State
let allModels = [];
let filteredModels = [];
let currentFilter = 'all';
let currentAiFilter = 'all';
let searchQuery = '';
let currentFeatureFilter = 'all';
let selectedModel = null;
let activeDownloadModel = null;
let progressPollInterval = null;
let isRecommendMode = false;
let implementedFeatures = {};
let uiSettings = {};

// DOM Elements
const modelGrid = document.getElementById('modelGrid');
const searchInput = document.getElementById('searchInput');
const statusFilter = document.getElementById('statusFilter');
const aiFilter = document.getElementById('aiFilter');
const featureFilter = document.getElementById('featureFilter');
const recommendButton = document.getElementById('recommendButton');
const sidebar = document.getElementById('modelDetailSidebar');
const modelDetail = document.getElementById('modelDetail');
const closeSidebar = document.getElementById('closeSidebar');
const downloadModal = document.getElementById('downloadModal');
const closeDownloadModal = document.getElementById('closeDownloadModal');
const cancelDownloadBtn = document.getElementById('cancelDownload');
const modalModelName = document.getElementById('modalModelName');
const progressFill = document.getElementById('progressFill');
const progressPercent = document.getElementById('progressPercent');
const progressSpeed = document.getElementById('progressSpeed');
const progressETA = document.getElementById('progressETA');
const progressText = document.getElementById('progressText');

// Status elements
const totalModelsEl = document.getElementById('totalModels');
const downloadedModelsEl = document.getElementById('downloadedModels');
const availableModelsEl = document.getElementById('availableModels');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadModels();
    await loadImplementedFeatures();
    await loadSettingsData();
    // Populate feature filter after both models and implemented features are loaded
    populateFeatureFilter();
    updateStatusSummary();
    applyFilters();
    setupEventListeners();
    startPeriodicRefresh();
});

// Load persisted UI settings (recommendation preference, compatibility filter)
async function loadSettingsData() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        uiSettings = await response.json();
    } catch (error) {
        console.warn('Could not load settings:', error);
        uiSettings = {};
    }
}

// Event Listeners
function setupEventListeners() {
    // Search
    searchInput.addEventListener('input', debounce(() => {
        searchQuery = searchInput.value.toLowerCase();
        refreshView();
    }, 300));

    // Status Filter
    statusFilter.addEventListener('change', () => {
        currentFilter = statusFilter.value;
        refreshView();
    });

    // AI Filter
    aiFilter.addEventListener('change', () => {
        currentAiFilter = aiFilter.value;
        refreshView();
    });

    // Feature Filter
    featureFilter.addEventListener('change', () => {
        currentFeatureFilter = featureFilter.value;
        refreshView();
    });

    // Recommend Button
    recommendButton.addEventListener('click', toggleRecommendMode);

    // Close sidebar
    closeSidebar.addEventListener('click', closeModelDetail);

    // Close download modal
    closeDownloadModal.addEventListener('click', hideDownloadModal);
    cancelDownloadBtn.addEventListener('click', cancelActiveDownload);

    // Close modal on overlay click
    downloadModal.addEventListener('click', (e) => {
        if (e.target === downloadModal) hideDownloadModal();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (sidebar.classList.contains('open')) closeModelDetail();
            if (downloadModal.style.display === 'flex') hideDownloadModal();
        }
    });
}

// Load models from API
async function loadModels() {
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) throw new Error('Failed to load models');
        allModels = await response.json();
        updateStatusSummary();
    } catch (error) {
        console.error('Error loading models:', error);
        showError('Failed to load models. Please refresh the page.');
    }
}

// Load implemented features from API
async function loadImplementedFeatures() {
    try {
        const response = await fetch(`${API_BASE}/features/implemented`);
        if (response.ok) {
            implementedFeatures = await response.json();
            // Re-populate feature filter with intersection
            populateFeatureFilter();
        }
    } catch (error) {
        console.warn('Could not load implemented features:', error);
    }
}

// Populate feature filter dynamically from loaded models (intersection with implemented)
function populateFeatureFilter() {
    // Extract unique features from all models
    const featureSet = new Set();
    allModels.forEach(model => {
        model.features.forEach(feature => featureSet.add(feature));
    });
    
    // Filter to only features that have UI implementations
    const implementedSet = new Set(Object.keys(implementedFeatures));
    const availableFeatures = Array.from(featureSet).filter(f => implementedSet.has(f)).sort();
    
    // Create slug mapping for values
    const slugMap = {};
    availableFeatures.forEach(feature => {
        // Create URL-friendly slug
        const slug = feature.toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-|-$/g, '');
        slugMap[feature] = slug;
    });
    
    // Store mapping for filter use
    window.featureSlugMap = slugMap;
    
    // Populate select
    const currentValue = featureFilter.value;
    featureFilter.innerHTML = '<option value="all">All Features</option>';
    
    availableFeatures.forEach(feature => {
        const option = document.createElement('option');
        option.value = slugMap[feature];
        option.textContent = feature;
        featureFilter.appendChild(option);
    });
    
    // Restore selection if valid
    if ([...featureFilter.options].some(opt => opt.value === currentValue)) {
        featureFilter.value = currentValue;
    } else {
        featureFilter.value = 'all';
        currentFeatureFilter = 'all';
    }
}

// Update status summary
function updateStatusSummary() {
    const downloaded = allModels.filter(m => m.downloaded).length;
    const available = allModels.filter(m => !m.downloaded).length;
    
    totalModelsEl.textContent = allModels.length;
    downloadedModelsEl.textContent = downloaded;
    availableModelsEl.textContent = available;
}

// Apply search / status / AI / feature / compatibility filters to a model list.
// Shared by the normal grid view AND the recommended view, so the filter
// options can be combined with recommend mode.
function getFilteredModels(models) {
    return models.filter(model => {
        // Hide models that are not compatible with this system (settings toggle)
        if (uiSettings.hideIncompatibleModels && model.compatible === false) return false;

        // Search filter
        if (searchQuery) {
            const searchable = `${model.name} ${model.description} ${model.features.join(' ')}`.toLowerCase();
            if (!searchable.includes(searchQuery)) return false;
        }

        // Status filter
        if (currentFilter === 'downloaded' && !model.downloaded) return false;
        if (currentFilter === 'available' && model.downloaded) return false;

        // AI / Non-AI filter
        if (currentAiFilter === 'ai' && !model.is_ai) return false;
        if (currentAiFilter === 'non-ai' && model.is_ai) return false;

        // Feature filter (uses dynamic slug map from populateFeatureFilter)
        if (currentFeatureFilter !== 'all') {
            // Reverse lookup: find the feature name from the slug
            const slugMap = window.featureSlugMap || {};
            const targetFeature = Object.keys(slugMap).find(key => slugMap[key] === currentFeatureFilter);
            if (targetFeature && !model.features.includes(targetFeature)) return false;
        }

        return true;
    });
}

// Apply filters and render
function applyFilters() {
    filteredModels = getFilteredModels(allModels);
    renderModelGrid();
}

// Render the current view. In recommend mode the active filters are applied on
// top of the scored recommendations; otherwise they filter the full catalog.
function refreshView() {
    if (isRecommendMode) {
        void showRecommendedModels();
    } else {
        applyFilters();
    }
}

// Render model grid
function renderModelGrid() {
    if (filteredModels.length === 0) {
        modelGrid.innerHTML = `
            <div class="empty-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <h3>No models found</h3>
                <p>${searchQuery || currentFilter !== 'all' ? 'Try adjusting your search or filter' : 'No models available'}</p>
            </div>
        `;
        return;
    }

    modelGrid.innerHTML = filteredModels.map(model => createModelCard(model)).join('');
    
    // Add click listeners
    modelGrid.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', (e) => {
            // Don't open detail if clicking a button
            if (e.target.closest('button')) return;
            openModelDetail(card.dataset.modelName);
        });
    });

    modelGrid.querySelectorAll('.btn-download').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            startDownload(btn.dataset.modelName);
        });
    });

    modelGrid.querySelectorAll('.btn-danger').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            if (confirm(`Delete ${btn.dataset.modelName}? This cannot be undone.`)) {
                await deleteModel(btn.dataset.modelName);
            }
        });
    });
}

// Create model card HTML
function createModelCard(model) {
    const weightClass = model.weight.toLowerCase().replace(/\s+/g, '-');
    const badgeClass = model.downloaded ? 'badge-downloaded' : 'badge-available';
    const badgeText = model.downloaded ? 'Downloaded' : 'Available';
    const cardClass = model.downloaded ? 'model-card downloaded' : 'model-card';
    const aiBadge = model.is_ai
        ? '<span class="badge-ai" title="Neural network / deep learning model">AI</span>'
        : '<span class="badge-nonai" title="Mathematical algorithm, no neural network">Non-AI</span>';

    let downloadAction;
    if (model.is_builtin) {
        downloadAction = `<span class="builtin-note">Built-in — no download needed</span>`;
    } else if (model.downloaded) {
        downloadAction = `<button class="btn-danger" data-model-name="${escapeHtml(model.name)}">Delete</button>`;
    } else if (!model.compatible) {
        downloadAction = `<button class="btn-download btn-disabled" disabled title="${escapeAttr(model.compatibility_reason || 'Incompatible with this system')}">Not compatible</button>`;
    } else {
        downloadAction = `<button class="btn-download" data-model-name="${escapeHtml(model.name)}">Download</button>`;
    }

    return `
        <div class="${cardClass}" data-model-name="${model.name}">
            <div class="model-card-header">
                <span class="model-card-name">${escapeHtml(model.name)}</span>
                <span class="model-card-badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="model-card-badges">
                ${aiBadge}
                ${getHardwareBadges(model)}
            </div>
            <p class="model-card-description">${escapeHtml(model.description)}</p>
            <div class="model-card-meta">
                <span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg> ${escapeHtml(model.size)}</span>
                <span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg> ${escapeHtml(model.weight)}</span>
            </div>
            <div class="model-card-features">
                ${model.features.slice(0, 4).map(f => `<span class="feature-tag">${escapeHtml(f)}</span>`).join('')}
                ${model.features.length > 4 ? `<span class="feature-tag">+${model.features.length - 4} more</span>` : ''}
            </div>
            <div class="model-card-actions">
                ${downloadAction}
                <button class="btn-secondary" data-model-name="${escapeHtml(model.name)}" onclick="event.stopPropagation(); openModelDetail('${escapeHtml(model.name)}')">Details</button>
            </div>
        </div>
    `;
}

// Hardware requirement badges (GPU / Metal / CPU)
function getHardwareBadges(model) {
    const badges = [];
    if (model.requires_cuda) {
        badges.push(`<span class="badge-hardware badge-gpu" title="Needs a GPU accelerator — runs on NVIDIA CUDA, AMD DirectML, or ROCm">GPU</span>`);
    } else if (model.requires_metal) {
        badges.push(`<span class="badge-hardware badge-metal" title="Requires Apple Silicon (Metal)">Metal</span>`);
    } else if (model.can_run_on_cpu !== false) {
        badges.push(`<span class="badge-hardware badge-cpu" title="Runs on CPU — no GPU required">CPU</span>`);
    }
    return badges.join('');
}

// HTML-escape attribute values safely
function escapeAttr(value) {
    return escapeHtml(String(value ?? '')).replace(/"/g, '&quot;');
}

// Open model detail sidebar
function openModelDetail(modelName) {
    const model = allModels.find(m => m.name === modelName);
    if (!model) return;
    
    selectedModel = model;
    renderModelDetail(model);
    sidebar.classList.add('open');
}

// Close model detail sidebar
function closeModelDetail() {
    sidebar.classList.remove('open');
    selectedModel = null;
}

// Render model detail
function renderModelDetail(model) {
    const weightClass = model.weight.toLowerCase().replace(/\s+/g, '-');
    const aiBadge = model.is_ai
        ? '<span class="badge-ai" title="Neural network / deep learning model">AI</span>'
        : '<span class="badge-nonai" title="Mathematical algorithm, no neural network">Non-AI (Pure Math)</span>';
    const hardwareBadges = getHardwareBadges(model);

    let downloadAction = '';
    if (model.is_builtin) {
        downloadAction = `<p class="builtin-note">Built-in algorithm — no download or installation required.</p>
                   <button class="btn-secondary" onclick="closeModelDetail()">Close</button>`;
    } else if (model.downloaded) {
        downloadAction = `<button class="btn-danger" onclick="deleteModel('${escapeHtml(model.name)}'); closeModelDetail();">Delete Model</button>
                   <button class="btn-secondary" onclick="closeModelDetail()">Close</button>`;
    } else if (!model.compatible) {
        downloadAction = `<button class="btn-download btn-disabled" disabled title="${escapeAttr(model.compatibility_reason || 'Incompatible with this system')}">Cannot Download</button>
                   <p class="compatibility-note">${escapeHtml(model.compatibility_reason || 'This model is not compatible with your hardware.')}</p>
                   <button class="btn-secondary" onclick="closeModelDetail()">Close</button>`;
    } else {
        downloadAction = `<button class="btn-download" onclick="startDownload('${escapeHtml(model.name)}'); closeModelDetail();" style="flex: 2;">Download Model</button>
                   <button class="btn-secondary" onclick="closeModelDetail()">Close</button>`;
    }

    modelDetail.innerHTML = `
        <div class="detail-header">
            <h3 class="detail-name">${escapeHtml(model.name)}</h3>
            <div class="detail-badges">
                <span class="model-card-badge ${model.downloaded ? 'badge-downloaded' : 'badge-available'}">
                    ${model.downloaded ? 'Downloaded' : 'Available'}
                </span>
                <span class="model-card-badge badge-${weightClass}">${escapeHtml(model.weight)}</span>
                ${aiBadge}
                ${hardwareBadges}
            </div>
        </div>
        
        <p class="detail-description">${escapeHtml(model.description)}</p>
        
        <div class="detail-section">
            <span class="detail-section-title">Properties</span>
            <div class="detail-meta-grid">
                <div class="detail-meta-item">
                    <span class="detail-meta-label">Size</span>
                    <span class="detail-meta-value">${escapeHtml(model.size)}</span>
                </div>
                <div class="detail-meta-item">
                    <span class="detail-meta-label">Weight</span>
                    <span class="detail-meta-value">${escapeHtml(model.weight)}</span>
                </div>
                <div class="detail-meta-item">
                    <span class="detail-meta-label">Type</span>
                    <span class="detail-meta-value">${model.is_ai ? 'AI Model' : 'Non-AI (Pure Math)'}</span>
                </div>
                <div class="detail-meta-item">
                    <span class="detail-meta-label">Compute</span>
                    <span class="detail-meta-value">${getComputeSummary(model)}</span>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <span class="detail-section-title">Features</span>
            <div class="detail-features">
                ${model.features.map(f => `<span class="detail-feature">${escapeHtml(f)}</span>`).join('')}
            </div>
        </div>
        
        <div class="detail-actions">
            ${downloadAction}
        </div>
    `;
}

// Summary of compute requirements for the detail panel
function getComputeSummary(model) {
    if (model.requires_cuda) return 'Requires a GPU accelerator (CUDA / DirectML / ROCm)';
    if (model.requires_metal) return 'Apple Silicon (Metal) required';
    if (model.can_run_on_cpu !== false) return 'CPU capable';
    return 'GPU required';
}

// Start download
async function startDownload(modelName) {
    const model = allModels.find(m => m.name === modelName);
    if (!model) return;
    
    if (model.compatible === false) {
        alert(model.compatibility_reason || 'This model is not compatible with your hardware.');
        return;
    }
    
    activeDownloadModel = modelName;
    modalModelName.textContent = `Downloading ${modelName}`;
    showDownloadModal();
    
    try {
        const response = await fetch(`${API_BASE}/models/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start download');
        }
        
        // Start polling for progress
        startProgressPolling(modelName);
        
    } catch (error) {
        console.error('Download error:', error);
        hideDownloadModal();
        alert(`Failed to start download: ${error.message}`);
    }
}

// Poll for download progress
function startProgressPolling(modelName) {
    if (progressPollInterval) clearInterval(progressPollInterval);
    
    progressPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/models/${modelName}/progress`);
            if (!response.ok) return;
            
            const progress = await response.json();
            updateDownloadProgress(progress);
            
            if (progress.status === 'completed') {
                clearInterval(progressPollInterval);
                progressPollInterval = null;
                hideDownloadModal();
                await loadModels(); // Refresh model list
            } else if (progress.status === 'failed' || progress.status === 'cancelled') {
                clearInterval(progressPollInterval);
                progressPollInterval = null;
                progressText.textContent = progress.error || 'Download failed';
                progressFill.style.backgroundColor = '#ef4444';
                cancelDownloadBtn.textContent = 'Close';
            }
        } catch (error) {
            console.error('Progress polling error:', error);
        }
    }, 500);
}

// Update download progress UI
function updateDownloadProgress(progress) {
    const percent = progress.progress_percent;
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${percent.toFixed(1)}%`;
    
    if (progress.speed_bps > 0) {
        progressSpeed.textContent = formatBytes(progress.speed_bps) + '/s';
    }
    
    if (progress.eta_seconds > 0) {
        progressETA.textContent = formatTime(progress.eta_seconds);
    }
    
    switch (progress.status) {
        case 'starting':
            progressText.textContent = 'Starting download...';
            break;
        case 'downloading':
            progressText.textContent = `Downloaded ${formatBytes(progress.downloaded_bytes)} of ${formatBytes(progress.total_bytes)}`;
            break;
        case 'verifying':
            progressText.textContent = 'Verifying download...';
            break;
        case 'completed':
            progressText.textContent = 'Download complete!';
            progressFill.style.backgroundColor = '#10b981';
            break;
        case 'failed':
            progressText.textContent = `Failed: ${progress.error}`;
            progressFill.style.backgroundColor = '#ef4444';
            break;
    }
}

// Cancel active download
async function cancelActiveDownload() {
    if (!activeDownloadModel) return;
    
    try {
        await fetch(`${API_BASE}/models/${activeDownloadModel}/cancel`, {
            method: 'POST'
        });
    } catch (error) {
        console.error('Cancel error:', error);
    }
    
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
    }
    
    hideDownloadModal();
    activeDownloadModel = null;
}

// Delete model
async function deleteModel(modelName) {
    try {
        const response = await fetch(`${API_BASE}/models/${modelName}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete model');
        }
        
        await loadModels();
        if (selectedModel?.name === modelName) {
            closeModelDetail();
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert(`Failed to delete model: ${error.message}`);
    }
}

// Show download modal
function showDownloadModal() {
    downloadModal.style.display = 'flex';
    // Reset progress UI
    progressFill.style.width = '0%';
    progressFill.style.backgroundColor = '';
    progressPercent.textContent = '0%';
    progressSpeed.textContent = '0 B/s';
    progressETA.textContent = '--:--';
    progressText.textContent = 'Starting download...';
    cancelDownloadBtn.textContent = 'Cancel';
}

// Hide download modal
function hideDownloadModal() {
    downloadModal.style.display = 'none';
    activeDownloadModel = null;
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
    }
}

// Show loading state
function showLoading() {
    modelGrid.innerHTML = `
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Loading models...</p>
        </div>
    `;
}

// Show error state
function showError(message) {
    modelGrid.innerHTML = `
        <div class="empty-state">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            <h3>Error</h3>
            <p>${escapeHtml(message)}</p>
        </div>
    `;
}

// Periodic refresh
function startPeriodicRefresh() {
    setInterval(async () => {
        // Only refresh if no active download
        if (!activeDownloadModel) {
            try {
                const response = await fetch(`${API_BASE}/models`);
                if (response.ok) {
                    const newModels = await response.json();
                    // Check if anything changed
                    const changed = JSON.stringify(newModels) !== JSON.stringify(allModels);
                    if (changed) {
                        allModels = newModels;
                        updateStatusSummary();
                        refreshView();
                        // Update detail view if open
                        if (selectedModel) {
                            const updated = allModels.find(m => m.name === selectedModel.name);
                            if (updated) renderModelDetail(updated);
                        }
                    }
                }
            } catch (error) {
                console.error('Refresh error:', error);
            }
        }
    }, 10000); // Every 10 seconds
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
}

// Toggle Recommend Mode — improved with state badge, keyboard shortcut, and smoother transition
function toggleRecommendMode() {
    isRecommendMode = !isRecommendMode;
    recommendButton.classList.toggle('active', isRecommendMode);

    // Update button label for clarity
    recommendButton.textContent = isRecommendMode ? 'Show All Models' : 'Recommended Models';

    // Show/hide a small badge in the button showing mode status
    let modeBadge = recommendButton.querySelector('.mode-badge');
    if (isRecommendMode) {
        if (!modeBadge) {
            modeBadge = document.createElement('span');
            modeBadge.className = 'mode-badge';
            modeBadge.textContent = 'ON';
            recommendButton.appendChild(modeBadge);
        } else {
            modeBadge.style.display = 'inline';
        }
        recommendButton.setAttribute('aria-pressed', 'true');
        recommendButton.title = 'Currently showing recommended models for your system. Click to see all.';
    } else {
        if (modeBadge) modeBadge.style.display = 'none';
        recommendButton.setAttribute('aria-pressed', 'false');
        recommendButton.title = 'Show only the best models for your system';
    }

    if (isRecommendMode) {
        void showRecommendedModels();
    } else {
        applyFilters();
    }
}

// Cached platform capabilities fetched from the backend (real detected hardware)
let cachedCapabilities = null;

async function loadCapabilities() {
    if (cachedCapabilities) return cachedCapabilities;
    try {
        const response = await fetch(`${API_BASE}/system/capabilities`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        cachedCapabilities = await response.json();
    } catch (error) {
        console.warn('Could not load system capabilities:', error);
        cachedCapabilities = null;
    }
    return cachedCapabilities;
}

// Compatibility-aware recommendation system.
// Candidates are drawn from ALL available models (AI + non-AI math alike), but
// only models that are actually usable on this system are shown:
//  - hardware-compatible per the backend's own compatibility check, AND
//  - obtainable (built-in, already downloaded, or has a download URL)
async function showRecommendedModels() {
    const caps = await loadCapabilities() || {};

    // In recommend mode, include only models that are actually usable on this system
    const candidates = allModels.filter(model => {
        // Skip explicitly incompatible models
        if (model.compatible === false) return false;
        // Only show obtainable models: built-in, downloaded, or with a download URL
        const obtainable = model.is_builtin || model.downloaded || model.has_download_url;
        return obtainable;
    });

    if (candidates.length === 0) {
        filteredModels = [];
        renderModelGrid();
        modelGrid.innerHTML = `
            <div class="empty-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 0 0-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 0 0-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 0 0-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 0 0-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 0 0 1.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                </svg>
                <h3>No compatible models</h3>
                <p>No models in the catalog are compatible with this system's hardware right now.</p>
                <p style="font-size: 0.875rem; color: var(--text-muted);">Try turning off the recommendation filter or downloading a model directly.</p>
            </div>
        `;
        return;
    }

    const preferAi = uiSettings.preferAi !== false;

    // Score all candidates — improved algorithm: weights based on user-system fit
    const scored = candidates.map(model => ({
        ...model,
        recommendScore: calculateModelScore(model, caps, preferAi),
    }));
    // Sort descending by score
    scored.sort((a, b) => b.recommendScore - a.recommendScore);

    // Apply active search/status/AI/feature filters on top of recommendations,
    // keeping recommendation ordering for anything that passes the filter.
    filteredModels = getFilteredModels(scored).slice(0, 6);
    renderModelGrid();

    if (filteredModels.length === 0) {
        // Show a helpful empty state when filters are too restrictive for recommendations
        modelGrid.innerHTML = `
            <div class="empty-state">
                <h3>No recommendations match your filters</h3>
                <p>Try loosening the search or filter settings above, or switch back to all models.</p>
            </div>
        `;
        return;
    }

    // Show concise reasoning for the top recommendation
    const reasons = buildRecommendationReasons(filteredModels[0], caps);
    if (reasons.length > 0) {
        showToast(
            `💡 Recommended: <strong>${filteredModels[0].name}</strong> — ${reasons[0]}` +
            (reasons.length > 1 ? ` (+${reasons.length - 1} more reasons)` : ''),
            8000
        );
    }
}

function calculateModelScore(model, caps, preferAi) {
    let score = 0;
    const sizeMB = parseModelSize(model.size);
    const hasGpu = !!(caps.has_cuda || caps.has_hip || caps.has_dml || caps.has_metal);
    const needsGpu = !!(model.requires_cuda || model.requires_metal);
    const ramGB = caps.total_ram_gb || 8;
    const vramGB = (caps.total_vram_gb || 0);

    // 1. Ready-to-use state (strongest signal — users prefer working out of the box)
    if (model.is_builtin) {
        score += 60; // highest bonus: zero-install built-in algorithm
    } else if (model.downloaded) {
        score += 55; // already downloaded — nearly as good
    } else {
        score += 10; // available but needs download
    }

    // 2. Hardware compatibility (critical: model must run on user's system)
    if (needsGpu && hasGpu) {
        score += 30; // GPU-accelerated and has GPU — best case
    } else if (!needsGpu) {
        score += 20; // runs everywhere including CPU-only
    } else {
        score -= 30; // requires GPU but user doesn't have one — strongly penalize
    }

    // 3. Hardware-specific bonuses
    if (hasGpu && model.is_ai) {
        score += 15; // AI model that leverages available GPU
    }
    if (!hasGpu && !model.is_ai) {
        score += 15; // pure-math model shines on CPU-only systems
    }

    // 4. AI preference from settings
    if (preferAi) {
        score += model.is_ai ? 20 : 0;
    } else {
        score += model.is_ai ? 5 : 15; // non-AI preferred when AI off
    }

    // 5. Size efficiency (lighter is generally preferred unless heavy has unique value)
    if (sizeMB < 30) score += 12;
    else if (sizeMB < 100) score += 6;
    else if (sizeMB < 300) score += 0;
    else if (sizeMB > 500) score -= 12;

    // 6. System memory fit
    if (model.weight === 'Heavy' && ramGB < 16) score -= 20;
    if (model.weight === 'Medium' && ramGB < 8) score -= 12;
    if (model.weight === 'Ultra-Lightweight') score += 10;

    // 7. VRAM awareness for GPU-accelerated AI models
    if (needsGpu && hasGpu) {
        if (vramGB >= 8) score += 8;
        else if (vramGB < 4) score -= 8; // very limited VRAM hurts large AI models
    }

    // 8. Versatility (more features = more useful for diverse tasks)
    const featureCount = (model.features || []).length;
    score += featureCount * 4;

    // 9. Popular use cases — weighted by common user needs
    const popularity = {
        'Upscale': 14,
        'Face Recovery': 10,
        'Frame Interpolation': 9,
        'Denoise': 7,
        'Sharpen / Deblur': 6,
        'Color Correction': 5,
        'SloMo': 4,
        'Optical Flow': 3,
        'Motion Estimation': 3,
        'Frame Recovery': 3
    };
    for (const feature of model.features || []) {
        score += popularity[feature] || 1; // unknown features get small positive weight
    }

    // 10. Recent user selection boost — if user frequently picks a feature, boost it
    // (would be read from localStorage; basic placeholder here)
    const userPreference = getUserPreferenceScore(model);
    score += userPreference;

    return Math.round(score);
}

function parseModelSize(sizeStr) {
    // Parse strings like "~65MB", "~150MB", "~15MB"
    const match = sizeStr.match(/~?(\d+(?:\.\d+)?)\s*(MB|GB)/i);
    if (!match) return 100;
    const value = parseFloat(match[1]);
    const unit = match[2].toUpperCase();
    return unit === 'GB' ? value * 1024 : value;
}

function getUserPreferenceScore(model) {
    try {
        const pref = localStorage.getItem(`pref_model_${model.name}`);
        if (pref === 'downloaded') return 5;
        if (pref === 'used') return 3;
    } catch (e) { }
    return 0;
}

function buildRecommendationReasons(model, caps) {
    const reasons = [];
    const hasGpu = !!(caps.has_cuda || caps.has_hip || caps.has_dml || caps.has_metal);
    const needsGpu = !!(model.requires_cuda || model.requires_metal);
    const ramGB = caps.total_ram_gb || 8;

    // Status / availability reason
    if (model.is_builtin) {
        reasons.push('Zero-install — runs instantly on this system');
    } else if (model.downloaded) {
        reasons.push('Already downloaded — ready to use right away');
    } else {
        reasons.push(`Available to download (${model.size})`);
    }

    // Hardware reason (most important for users)
    if (needsGpu && hasGpu) {
        const backend = caps.has_cuda ? 'CUDA' : (caps.has_hip ? 'ROCm/HIP' : (caps.has_dml ? 'DirectML' : 'Metal'));
        reasons.push(`GPU-accelerated via ${backend}`);
    } else if (!needsGpu) {
        reasons.push('Works on any hardware, CPU or GPU');
    } else {
        reasons.push('Requires a GPU — your system has one');
    }

    // Size / resource reason
    const sizeMB = parseModelSize(model.size);
    if (sizeMB < 30) {
        reasons.push('Lightweight — minimal download');
    } else if (sizeMB > 500) {
        reasons.push('Large model — high quality but big download');
    } else {
        reasons.push(`Moderate size (${model.size})`);
    }

    // System resources
    if (ramGB < 8 && model.weight === 'Heavy') {
        reasons.push('Caution: heavy on RAM — consider lighter alternative');
    }

    // Feature highlights
    if (model.features && model.features.length > 0) {
        const topFeatures = model.features.slice(0, 2);
        reasons.push(`Features: ${topFeatures.join(' + ')}`);
    }

    // Versatility
    const featureCount = (model.features || []).length;
    if (featureCount > 2) {
        reasons.push(`Multi-purpose (${featureCount} features)`);
    }

    return reasons;
}

function showToast(message, duration = 10000) {
    // Remove any existing toast
    const existingToast = document.querySelector('.recommendation-toast');
    if (existingToast) existingToast.remove();
    
    // Create toast element with better styling
    const toast = document.createElement('div');
    toast.className = 'recommendation-toast';
    toast.innerHTML = message;
    document.body.appendChild(toast);
    
    // Auto-remove after animation completes
    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, duration);
}

// Make functions globally available for inline handlers
window.openModelDetail = openModelDetail;
window.startDownload = startDownload;
window.deleteModel = deleteModel;