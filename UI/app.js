/**
 * Main UI Logic for openVideoAI
 * Feature-based enhancement pipeline with per-feature model selection
 */

const API_BASE = '/api';

// Feature definitions - maps feature to model filter and UI elements
const FEATURES = {
    base: {
        name: 'Upscale',
        featureKey: 'Upscale',
        required: true,
        modelSelector: 'baseModelSelect',
        toggleId: null, // Always enabled
        contentId: null,
        strengthId: null,
        badgeId: null
    },
    face: {
        name: 'Face Recovery',
        featureKey: 'Face Recovery',
        modelSelector: 'faceModelSelect',
        toggleId: 'faceToggle',
        contentId: 'faceContent',
        strengthId: null,
        badgeId: 'faceBadge'
    },
    sharpen: {
        name: 'Sharpen / Deblur',
        featureKey: 'Sharpen / Deblur',
        modelSelector: 'sharpenModelSelect',
        toggleId: 'sharpenToggle',
        contentId: 'sharpenContent',
        strengthId: 'sharpenStrength',
        badgeId: 'sharpenBadge'
    },
    denoise: {
        name: 'Denoise',
        featureKey: 'Denoise',
        modelSelector: 'denoiseModelSelect',
        toggleId: 'denoiseToggle',
        contentId: 'denoiseContent',
        strengthId: 'denoiseStrength',
        badgeId: 'denoiseBadge'
    },
    frame: {
        name: 'Frame Interpolation',
        featureKey: 'Frame Interpolation',
        modelSelector: 'frameModelSelect',
        toggleId: 'frameToggle',
        contentId: 'frameContent',
        strengthId: null,
        badgeId: 'frameBadge'
    },
    color: {
        name: 'Match Color',
        featureKey: 'Match Color',
        modelSelector: 'colorModelSelect',
        toggleId: 'colorToggle',
        contentId: 'colorContent',
        strengthId: null,
        badgeId: 'colorBadge'
    },
    log: {
        name: 'Log Restoration',
        featureKey: 'Log Restoration',
        modelSelector: 'logModelSelect',
        toggleId: 'logRescueToggle',
        contentId: 'logRescueContent',
        strengthId: null,
        badgeId: 'logRescueBadge'
    }
};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const enhanceBtn = document.getElementById('enhanceBtn');
const modelSelectors = {};

// State
let allModels = [];
let downloadedModels = [];
let selectedFile = null;
let selectedReferenceFile = null;
let featureStates = {};
let logRescueModule = null;
let systemSettings = {
    computeBackend: 'auto',
    gpuDevice: 'auto',
    defaultUpscale: 4,
    defaultTileSize: 400,
    outputFormat: 'jpg',
    jpgQuality: 95
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSystemSettings();
    await loadModels();
    await loadLogRescueModule();
    updateFeatureAvailability();
    loadPreferences();
    setupEventListeners();
});

async function loadSystemSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        systemSettings = { ...systemSettings, ...await response.json() };
        const upscale = document.getElementById('upscaleSelect');
        const outputFormat = document.getElementById('outputFormatSelect');
        const upscaleValue = `${systemSettings.defaultUpscale}x`;
        if ([...upscale.options].some(option => option.value === upscaleValue || option.text === upscaleValue)) {
            upscale.value = upscaleValue;
        }
        if ([...outputFormat.options].some(option => option.value === systemSettings.outputFormat)) {
            outputFormat.value = systemSettings.outputFormat;
        }
    } catch (error) {
        console.error('Could not load system settings:', error);
    }
}

// Load models from API
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) throw new Error('Failed to load models');
        
        allModels = await response.json();
        downloadedModels = allModels.filter(m => m.downloaded);
        
        // Initialize feature states
        Object.keys(FEATURES).forEach(key => {
            featureStates[key] = {
                enabled: FEATURES[key].required || false,
                model: null
            };
        });
        
        // Populate all model selectors
        populateAllModelSelectors();
        
    } catch (error) {
        console.error('Error loading models:', error);
        showError('Failed to load models');
    }
}

async function loadLogRescueModule() {
    const section = document.getElementById('section-log');
    try {
        const response = await fetch(`${API_BASE}/modules`);
        if (!response.ok) throw new Error('Failed to load processing modules');
        const modules = await response.json();
        logRescueModule = modules.find(module => module.module === 'Log_Rescue_and_Restoration');
        if (!logRescueModule) throw new Error('LOG Rescue module is missing from the catalog');

        document.getElementById('logRescueTitle').textContent = logRescueModule.display_name;
        document.getElementById('logRescueDescription').textContent = logRescueModule.description;

        const profileSelect = document.getElementById('logProfileSelect');
        profileSelect.replaceChildren(...logRescueModule.supported_input_profiles.map(profile => {
            const option = document.createElement('option');
            option.value = profile;
            option.textContent = profile;
            return option;
        }));
        profileSelect.disabled = false;

        const parameters = logRescueModule.configuration_parameters;
        const denoise = parameters.denoise_strength;
        const shadow = parameters.shadow_recovery;
        const denoiseInput = document.getElementById('logDenoiseStrength');
        const shadowInput = document.getElementById('logShadowRecovery');
        configureRangeInput(denoiseInput, denoise);
        configureRangeInput(shadowInput, shadow);
        document.getElementById('logDenoiseValue').textContent = Number(denoise.default).toFixed(2);
        document.getElementById('logShadowValue').textContent = Number(shadow.default).toFixed(2);

        const colorSpace = parameters.color_space_target;
        const colorSpaceSelect = document.getElementById('logColorSpace');
        colorSpaceSelect.replaceChildren(...colorSpace.options.map(target => {
            const option = document.createElement('option');
            option.value = target;
            option.textContent = target;
            return option;
        }));
        colorSpaceSelect.value = colorSpace.default;
        colorSpaceSelect.disabled = false;

        const pipelineList = document.getElementById('logPipelineSteps');
        pipelineList.replaceChildren(...Object.values(logRescueModule.processing_pipeline).map(step => {
            const item = document.createElement('li');
            item.textContent = `${step.process}: ${step.model}`;
            return item;
        }));

    } catch (error) {
        console.error('Error loading LOG Rescue module:', error);
        section.classList.add('feature-unavailable');
        document.getElementById('logRescueToggle').disabled = true;
        document.getElementById('logRescueDescription').textContent =
            'Could not load LOG Rescue module configuration.';
    }
}

function configureRangeInput(input, parameter) {
    input.min = parameter.range[0];
    input.max = parameter.range[1];
    input.step = 0.05;
    input.value = parameter.default;
    input.disabled = false;
}

function setLogRescueControlsEnabled(enabled) {
    const profileSelect = document.getElementById('logProfileSelect');
    const denoiseInput = document.getElementById('logDenoiseStrength');
    const shadowInput = document.getElementById('logShadowRecovery');
    const colorSpaceSelect = document.getElementById('logColorSpace');
    const modelSelect = modelSelectors.log;
    const enabledControls = Boolean(enabled && modelSelect && !modelSelect.disabled);

    [profileSelect, denoiseInput, shadowInput, colorSpaceSelect].forEach(control => {
        control.disabled = !enabledControls;
    });
}

// Populate all model selectors
function populateAllModelSelectors() {
    // Get model selectors
    Object.keys(FEATURES).forEach(key => {
        const selector = document.getElementById(FEATURES[key].modelSelector);
        if (selector) modelSelectors[key] = selector;
    });

    // Base model selector (required - gets all upscale models)
    populateSelector('base', downloadedModels.filter(m => m.features.includes('Upscale')));
    
    // Feature-specific selectors
    populateSelector('face', downloadedModels.filter(m => m.features.includes('Face Recovery')));
    populateSelector('sharpen', downloadedModels.filter(m => m.features.includes('Sharpen / Deblur')));
    populateSelector('denoise', downloadedModels.filter(m => m.features.includes('Denoise')));
    populateSelector('frame', downloadedModels.filter(m => m.features.includes('Frame Interpolation')));
    populateSelector('color', downloadedModels.filter(m => m.features.includes('Match Color')));
    populateSelector('log', downloadedModels.filter(m => m.features.includes('Log Restoration')));
}

// Populate a single selector
function populateSelector(featureKey, models) {
    const selector = modelSelectors[featureKey];
    if (!selector) return;

    const feature = FEATURES[featureKey];
    
    if (models.length === 0) {
        selector.innerHTML = `<option value="">No ${feature.name.toLowerCase()} models downloaded</option>`;
        selector.disabled = true;
    } else {
        selector.innerHTML = models.map(m => 
            `<option value="${m.name}">${m.name} (${m.size})</option>`
        ).join('');
        selector.disabled = false;
        
        // Restore saved selection
        const saved = localStorage.getItem(`model_${featureKey}`);
        if (saved && [...selector.options].some(o => o.value === saved)) {
            selector.value = saved;
            featureStates[featureKey].model = saved;
        } else if (models.length > 0) {
            selector.value = models[0].name;
            featureStates[featureKey].model = models[0].name;
        }
    }
}

// Update feature availability based on downloaded models
function updateFeatureAvailability() {
    Object.keys(FEATURES).forEach(key => {
        if (key === 'base') return; // Base is always required
        
        const feature = FEATURES[key];
        const selector = modelSelectors[key];
        const section = document.getElementById(`section-${key}`);
        const toggle = document.getElementById(feature.toggleId);
        const content = document.getElementById(feature.contentId);
        const badge = document.getElementById(feature.badgeId);
        
        const hasModels = selector && !selector.disabled &&
            (key !== 'log' || Boolean(logRescueModule));
        
        if (hasModels) {
            // Models available - enable feature
            section.classList.remove('feature-unavailable');
            if (toggle) toggle.disabled = false;
            if (content) content.style.opacity = featureStates[key].enabled ? '1' : '0.5';
            if (key === 'log') setLogRescueControlsEnabled(featureStates[key].enabled);
            if (badge) {
                badge.textContent = 'Available';
                badge.className = 'feature-badge available';
            }
        } else {
            // No models - disable feature
            section.classList.add('feature-unavailable');
            if (toggle) {
                toggle.checked = false;
                toggle.disabled = true;
            }
            featureStates[key].enabled = false;
            if (content) content.style.opacity = '0.4';
            if (key === 'log') setLogRescueControlsEnabled(false);
            if (badge) {
                badge.textContent = 'No model';
                badge.className = 'feature-badge unavailable';
            }
        }
    });
}

// Load saved preferences
function loadPreferences() {
    Object.keys(FEATURES).forEach(key => {
        if (key === 'base') return;
        
        const feature = FEATURES[key];
        const saved = localStorage.getItem(`enabled_${key}`);
        if (saved !== null) {
            featureStates[key].enabled = saved === 'true';
            const toggle = document.getElementById(feature.toggleId);
            if (toggle) toggle.checked = featureStates[key].enabled;
        }
        
        // Update content visibility
        const content = document.getElementById(feature.contentId);
        if (content) {
            content.style.opacity = featureStates[key].enabled ? '1' : '0.5';
            content.style.pointerEvents = featureStates[key].enabled ? 'auto' : 'none';
        }
        if (key === 'log') setLogRescueControlsEnabled(featureStates[key].enabled);
    });
}

// Save preferences
function savePreferences() {
    Object.keys(FEATURES).forEach(key => {
        if (key === 'base') return;
        localStorage.setItem(`enabled_${key}`, featureStates[key].enabled);
        const selector = modelSelectors[key];
        if (selector && selector.value) {
            localStorage.setItem(`model_${key}`, selector.value);
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    const logRescueToggle = document.getElementById('logRescueToggle');
    document.getElementById('logDenoiseStrength').addEventListener('input', event => {
        document.getElementById('logDenoiseValue').textContent =
            Number(event.target.value).toFixed(2);
    });
    document.getElementById('logShadowRecovery').addEventListener('input', event => {
        document.getElementById('logShadowValue').textContent =
            Number(event.target.value).toFixed(2);
    });

    // Feature toggles
    Object.keys(FEATURES).forEach(key => {
        if (key === 'base') return;
        
        const feature = FEATURES[key];
        const toggle = document.getElementById(feature.toggleId);
        const content = document.getElementById(feature.contentId);
        const selector = modelSelectors[key];
        
        if (toggle) {
            toggle.addEventListener('change', () => {
                featureStates[key].enabled = toggle.checked;
                if (content) {
                    content.style.opacity = toggle.checked ? '1' : '0.5';
                    content.style.pointerEvents = toggle.checked ? 'auto' : 'none';
                }
                if (key === 'log') setLogRescueControlsEnabled(toggle.checked);
                savePreferences();
            });
        }
        
        if (selector) {
            selector.addEventListener('change', () => {
                featureStates[key].model = selector.value;
                savePreferences();
            });
        }
    });
    
    // Base model selector
    if (modelSelectors.base) {
        modelSelectors.base.addEventListener('change', () => {
            featureStates.base.model = modelSelectors.base.value;
            savePreferences();
        });
    }

    const colorReferenceInput = document.getElementById('colorReferenceInput');
    const colorReferenceName = document.getElementById('colorReferenceName');
    if (colorReferenceInput && colorReferenceName) {
        colorReferenceInput.addEventListener('change', () => {
            const file = colorReferenceInput.files[0];
            selectedReferenceFile = file && file.type.startsWith('image/') ? file : null;
            colorReferenceName.textContent = selectedReferenceFile
                ? selectedReferenceFile.name
                : 'No reference image selected';
            if (file && !selectedReferenceFile) {
                colorReferenceInput.value = '';
                alert('Please select an image as the color reference.');
            }
        });
    }

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.backgroundColor = 'rgba(59, 130, 246, 0.05)';
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.backgroundColor = 'rgba(30, 30, 30, 0.5)';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.backgroundColor = 'rgba(30, 30, 30, 0.5)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // Click to select file
    dropZone.addEventListener('click', () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*,video/*';
        input.onchange = (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        };
        input.click();
    });
    
    // Enhance button
    enhanceBtn.addEventListener('click', startEnhancement);
    
    // Enter key on drop zone
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            dropZone.click();
        }
    });
    dropZone.setAttribute('tabindex', '0');
    dropZone.setAttribute('role', 'button');
    dropZone.setAttribute('aria-label', 'Drop zone for media files');
}

// Handle file selection
function handleFileSelect(file) {
    const validTypes = ['image/jpeg', 'image/png', 'video/mp4', 'video/webm', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
        alert('Unsupported file type. Please select JPG, PNG, or MP4.');
        return;
    }
    
    selectedFile = file;
    updateDropZoneUI(file);
}

// Update drop zone UI with selected file
function updateDropZoneUI(file) {
    const isVideo = file.type.startsWith('video/');
    const icon = isVideo 
        ? '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>'
        : '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>';
    
    const sizeStr = formatFileSize(file.size);
    
    dropZone.innerHTML = `
        ${icon}
        <h3>${file.name}</h3>
        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-muted);">${sizeStr} • ${file.type}</p>
        <button class="btn-secondary" style="margin-top: 1rem; width: auto; padding: 0.5rem 1.5rem;" onclick="event.stopPropagation(); clearFileSelection();">Remove</button>
    `;
    dropZone.style.borderColor = 'var(--primary)';
    dropZone.style.backgroundColor = 'rgba(59, 130, 246, 0.05)';
}

// Clear file selection
function clearFileSelection() {
    selectedFile = null;
    dropZone.innerHTML = `
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
        </svg>
        <h3>Drag & Drop Media Here</h3>
        <p style="margin-top: 0.5rem; font-size: 0.875rem;">Supports JPG, PNG, MP4</p>
    `;
    dropZone.style.borderColor = 'var(--border)';
    dropZone.style.backgroundColor = 'rgba(30, 30, 30, 0.5)';
}

// Start enhancement process
async function startEnhancement() {
    if (!selectedFile) {
        alert('Please select a media file first');
        return;
    }
    
    // Validate base model
    const baseModel = featureStates.base.model;
    const logRescueEnabled = featureStates.log.enabled;
    if (!baseModel && !logRescueEnabled) {
        alert('Please select a base upscale model');
        return;
    }

    if (logRescueEnabled && !selectedFile.type.startsWith('video/')) {
        alert('LOG Rescue is available for video files. Please select a video.');
        return;
    }
    if (logRescueEnabled && !logRescueModule) {
        alert('LOG Rescue configuration is unavailable. Reload the page and try again.');
        return;
    }

    if (featureStates.color.enabled && !selectedReferenceFile) {
        alert('Please select a reference image to use Match Color.');
        return;
    }
    
    // Gather all settings
    const settings = {
        inputFile: {
            name: selectedFile.name,
            type: selectedFile.type,
            size: selectedFile.size
        },
        baseModel: baseModel,
        upscale: document.getElementById('upscaleSelect').value,
        features: {}
    };

    if (logRescueEnabled) {
        settings.features.log = {
            model: modelSelectors.log.value,
            module: logRescueModule.module,
            inputProfile: document.getElementById('logProfileSelect').value,
            denoiseStrength: Number(document.getElementById('logDenoiseStrength').value),
            shadowRecovery: Number(document.getElementById('logShadowRecovery').value),
            colorSpaceTarget: document.getElementById('logColorSpace').value
        };
    }
    
    // Add enabled features
    Object.keys(FEATURES).forEach(key => {
        if (key === 'base') return;
        
        if (featureStates[key].enabled && featureStates[key].model) {
            settings.features[key] = {
                model: featureStates[key].model,
                strength: featureStates[key].strengthId 
                    ? parseInt(document.getElementById(featureStates[key].strengthId).value) 
                    : null
            };
        }
    });
    
    // Frame interpolation specific settings
    if (featureStates.frame.enabled) {
        settings.features.frame.fps = document.getElementById('fpsSelect').value;
        settings.features.frame.slowMotion = document.getElementById('slowmoSelect').value;
        settings.features.frame.sceneDetection = document.getElementById('sceneDetectToggle').checked;
    }

    if (featureStates.color.enabled) {
        settings.features.color.referenceFile = {
            name: selectedReferenceFile.name,
            type: selectedReferenceFile.type,
            size: selectedReferenceFile.size
        };
    }
    
    // Output settings
    settings.output = {
        format: document.getElementById('outputFormatSelect').value,
        quality: document.getElementById('qualitySelect').value
    };
    
    console.log('Enhancement settings:', settings);
    
    // Process using the actual API
    await processFile(settings);
}

// Process file via API
async function processFile(settings) {
    const isVideo = selectedFile.type.startsWith('video/');
    const scale = parseInt(settings.upscale.replace('x', '')) || 4;
    
    // Show processing UI
    showProcessingUI();
    
    try {
        // Upload file to processing endpoint
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model_name', settings.baseModel);
        formData.append('scale', scale.toString());
        formData.append('device', getProcessingDevice());
        formData.append('output_format', settings.output.format);
        formData.append('jpg_quality', String(systemSettings.jpgQuality));
        
        const uploadResponse = await fetch('/api/processing/upscale/file', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            const error = await uploadResponse.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const { task_id } = await uploadResponse.json();
        
        // Poll for progress
        const result = await pollProcessingTask(task_id, (progress) => {
            updateProcessingProgress(progress);
        });
        
        // Download result
        await downloadResult(task_id, settings.inputFile.name, scale, result);
        
    } catch (error) {
        hideProcessingUI();
        alert('Processing failed: ' + error.message);
        console.error(error);
    }
}

function getProcessingDevice() {
    if (systemSettings.computeBackend === 'cpu') return 'cpu';
    const selectedDevice = String(systemSettings.gpuDevice || 'auto');
    if (systemSettings.computeBackend === 'cuda') {
        return selectedDevice.startsWith('cuda:')
            ? selectedDevice
            : 'cuda';
    }
    if (systemSettings.computeBackend === 'metal') return 'mps';
    return selectedDevice !== 'auto' ? selectedDevice : 'auto';
}

// Poll processing task
async function pollProcessingTask(taskId, onProgress) {
    return new Promise((resolve, reject) => {
        const checkStatus = async () => {
            try {
                const response = await fetch(`/api/processing/task/${taskId}`);
                if (!response.ok) {
                    throw new Error('Failed to check status');
                }
                
                const status = await response.json();
                
                if (onProgress) onProgress(status.progress || 0);
                
                if (status.status === 'completed') {
                    resolve(status.result);
                } else if (status.status === 'failed') {
                    reject(new Error(status.error || 'Processing failed'));
                } else {
                    // Still processing
                    setTimeout(checkStatus, 1000);
                }
            } catch (error) {
                reject(error);
            }
        };
        
        checkStatus();
    });
}

// Download result file
async function downloadResult(taskId, originalName, scale, result) {
    try {
        const response = await fetch(`/api/processing/result/${taskId}`);
        if (!response.ok) {
            throw new Error('Failed to download result');
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        const outputPath = result && result.output_path;
        if (outputPath) {
            a.download = outputPath.split(/[\\/]/).pop();
        } else {
            const nameParts = originalName.split('.');
            const ext = nameParts.pop();
            const baseName = nameParts.join('.');
            a.download = `${baseName}_upscaled_${scale}x.${ext}`;
        }
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        hideProcessingUI();
        alert('Processing complete! File downloaded.');
        
    } catch (error) {
        hideProcessingUI();
        alert('Download failed: ' + error.message);
    }
}

// Processing UI
let processingModal = null;

function showProcessingUI() {
    if (processingModal) return;
    
    processingModal = document.createElement('div');
    processingModal.className = 'processing-modal-overlay';
    processingModal.innerHTML = `
        <div class="processing-modal">
            <div class="processing-header">
                <h3>Processing...</h3>
            </div>
            <div class="processing-body">
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="processingProgressFill"></div>
                    </div>
                    <div class="progress-text" id="processingProgressText">Starting...</div>
                </div>
                <div class="processing-stats">
                    <span id="processingTime">Elapsed: 0s</span>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(processingModal);
    
    // Add styles if not present
    if (!document.getElementById('processing-styles')) {
        const style = document.createElement('style');
        style.id = 'processing-styles';
        style.textContent = `
            .processing-modal-overlay {
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 2000;
                animation: fadeIn 0.2s ease;
            }
            .processing-modal {
                background: var(--panel-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                width: 90%;
                max-width: 480px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
                animation: slideUp 0.3s ease;
            }
            .processing-header {
                padding: 1rem 1.5rem;
                border-bottom: 1px solid var(--border);
            }
            .processing-header h3 { margin: 0; font-size: 1rem; }
            .processing-body { padding: 1.5rem; }
            .progress-bar {
                height: 10px;
                background: var(--bg-dark);
                border-radius: 5px;
                overflow: hidden;
                margin-bottom: 0.75rem;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--primary), #60a5fa);
                border-radius: 5px;
                transition: width 0.3s ease;
                width: 0%;
            }
            .progress-text {
                font-size: 0.875rem;
                color: var(--text-main);
                text-align: center;
                margin-bottom: 1rem;
            }
            .processing-stats {
                display: flex;
                justify-content: center;
                font-size: 0.8125rem;
                color: var(--text-muted);
            }
        `;
        document.head.appendChild(style);
    }
    
    const startTime = Date.now();
    processingModal._startTime = startTime;
    processingModal._interval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const timeEl = document.getElementById('processingTime');
        if (timeEl) timeEl.textContent = `Elapsed: ${elapsed}s`;
    }, 1000);
}

function updateProcessingProgress(progress) {
    if (!processingModal) return;
    
    const fill = document.getElementById('processingProgressFill');
    const text = document.getElementById('processingProgressText');
    
    if (fill) fill.style.width = `${progress}%`;
    if (text) {
        if (progress < 100) {
            text.textContent = `Processing... ${Math.round(progress)}%`;
        } else {
            text.textContent = 'Processing complete! Preparing download...';
        }
    }
}

function hideProcessingUI() {
    if (processingModal) {
        clearInterval(processingModal._interval);
        processingModal.remove();
        processingModal = null;
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showError(message) {
    console.error(message);
}

// Make clearFileSelection globally available for inline handler
window.clearFileSelection = clearFileSelection;