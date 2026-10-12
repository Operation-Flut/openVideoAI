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
        required: false,
        modelSelector: 'baseModelSelect',
        toggleId: 'baseToggle',
        contentId: 'baseContent',
        strengthId: null,
        badgeId: 'baseBadge'
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
        strengthId: 'colorStrength',
        badgeId: 'colorBadge',
        requiresReferenceImage: true
    },
    log: {
        name: 'Log Restoration',
        featureKey: 'Log Restoration',
        modelSelector: 'logModelSelect',
        toggleId: 'logRescueToggle',
        contentId: 'logRescueContent',
        strengthId: null,
        badgeId: 'logRescueBadge'
    },
    lowlight: {
        name: 'Low-Light Enhancement',
        featureKey: 'Low-Light Enhancement',
        modelSelector: 'lowlightModelSelect',
        toggleId: 'lowlightToggle',
        contentId: 'lowlightContent',
        strengthId: 'lowlightStrength',
        badgeId: 'lowlightBadge'
    },
    sdrtohdr: {
        name: 'SDR to HDR',
        featureKey: 'SDR to HDR',
        modelSelector: 'sdrtohdrModelSelect',
        toggleId: 'sdrtohdrToggle',
        contentId: 'sdrtohdrContent',
        strengthId: null,
        badgeId: 'sdrtohdrBadge'
    },
    flatprofile: {
        name: 'Flat Color Profile',
        featureKey: 'Flat Color Profile',
        modelSelector: 'flatprofileModelSelect',
        toggleId: 'flatprofileToggle',
        contentId: 'flatprofileContent',
        strengthId: null,
        badgeId: 'flatprofileBadge'
    },
    overexposure: {
        name: 'Overexposure & Contrast Fix',
        featureKey: 'Overexposure & Contrast Fix',
        modelSelector: 'overexposureModelSelect',
        toggleId: 'overexposureToggle',
        contentId: 'overexposureContent',
        strengthId: null,
        badgeId: 'overexposureBadge'
    },
    reencode: {
        name: 'Reencoding',
        featureKey: 'Reencoding',
        modelSelector: 'reencodeModelSelect',
        toggleId: 'reencodeToggle',
        contentId: 'reencodeContent',
        strengthId: null,
        badgeId: 'reencodeBadge'
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
let inputResolution = null; // { w, h } once a file is loaded
let logRescueModule = null;
let processingFeatures = null;
let systemSettings = {
    computeBackend: 'auto',
    gpuDevice: 'auto',
    defaultUpscale: 4,
    defaultTileSize: 400,
    outputFormat: 'jpg',
    jpgQuality: 95,
    sharpenOverlap: 128
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSystemSettings();
    await loadModels();
    await loadProcessingFeatures();
    await loadLogRescueModule();
    updateFeatureAvailability();
    loadPreferences();
    setupEventListeners();
    
    // Fix: ensure base content visible and dropdown populated
    const bc = document.getElementById('baseContent');
    if (bc) { bc.style.opacity = '1'; bc.style.pointerEvents = 'auto'; }
    if (downloadedModels.length > 0 && modelSelectors.base) {
        populateSelector('base', downloadedModels.filter(m => m.features && m.features.includes('Upscale')));
    }
    updateScaleDropdown();
});

async function loadSystemSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        systemSettings = { ...systemSettings, ...await response.json() };
        const outputFormat = document.getElementById('outputFormatSelect');
        if ([...outputFormat.options].some(option => option.value === systemSettings.outputFormat)) {
            outputFormat.value = systemSettings.outputFormat;
        }
        // Sync the default upscale scale from settings. updateScaleDropdown()
        // reads the current value later and preserves it when rebuilding options.
        const upscaleSelect = document.getElementById('upscaleSelect');
        const defaultScale = `${systemSettings.defaultUpscale || 4}x`;
        if (upscaleSelect && [...upscaleSelect.options].some(o => o.value === defaultScale)) {
            upscaleSelect.value = defaultScale;
        }
        // Scale dropdown is updated later once we know the input resolution
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
                enabled: key === 'base' ? true : false, // base defaults on, others off
                model: null,
                strength: null
            };
        });
        
        // Populate all model selectors
        populateAllModelSelectors();
        
    } catch (error) {
        console.error('Error loading models:', error);
        showError('Failed to load models');
    }
}

// Load available processing features from API
async function loadProcessingFeatures() {
    try {
        const response = await fetch(`${API_BASE}/processing/features`);
        if (!response.ok) throw new Error('Failed to load processing features');
        processingFeatures = await response.json();
        console.log('Processing features loaded:', Object.keys(processingFeatures));
    } catch (error) {
        console.error('Error loading processing features:', error);
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
    updateRangeFill(input);
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
    populateSelector('base', downloadedModels.filter(m => m.features && m.features.includes('Upscale')));
    
    // Feature-specific selectors
    populateSelector('face', downloadedModels.filter(m => m.features && m.features.includes('Face Recovery')));
    populateSelector('sharpen', downloadedModels.filter(m => m.features && m.features.includes('Sharpen / Deblur')));
    populateSelector('denoise', downloadedModels.filter(m => m.features && m.features.includes('Denoise')));
    populateSelector('frame', downloadedModels.filter(m => m.features && m.features.includes('Frame Interpolation')));
    populateSelector('color', downloadedModels.filter(m => m.features && m.features.includes('Match Color')));
    populateSelector('log', downloadedModels.filter(m => m.features && m.features.includes('Log Restoration')));
    
    // New feature selectors - populate from processing features API
    populateSelectorFromFeatures('lowlight', 'low_light', ['Zero-DCE']);
    populateSelectorFromFeatures('sdrtohdr', 'sdr_to_hdr', ['HDRNet']);
    populateSelectorFromFeatures('flatprofile', 'flat_color_profile', ['ColorGrade']);
    populateSelectorFromFeatures('overexposure', 'overexposure_fix', ['AutoEnhance']);
    populateSelectorFromFeatures('reencode', 'reencoding', ['ffmpeg']);
}

// Populate selector from processing features API
function populateSelectorFromFeatures(featureKey, apiFeatureKey, availableModels) {
    const selector = modelSelectors[featureKey];
    if (!selector) return;
    
    const feature = FEATURES[featureKey];
    
    if (availableModels.length === 0) {
        selector.innerHTML = `<option value="">No ${feature.name.toLowerCase()} models available</option>`;
        selector.disabled = true;
    } else {
        selector.innerHTML = availableModels.map(m => 
            `<option value="${m}">${m}</option>`
        ).join('');
        selector.disabled = false;
        
        // Restore saved selection
        const saved = localStorage.getItem(`model_${featureKey}`);
        if (saved && [...selector.options].some(o => o.value === saved)) {
            selector.value = saved;
            featureStates[featureKey].model = saved;
        } else if (availableModels.length > 0) {
            selector.value = availableModels[0];
            featureStates[featureKey].model = availableModels[0];
        }
    }
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
        const feature = FEATURES[key];
        const selector = modelSelectors[key];
        const section = document.getElementById(`section-${key}`);
        const toggle = document.getElementById(feature.toggleId);
        const content = document.getElementById(feature.contentId);
        const badge = document.getElementById(feature.badgeId);

        const hasModels = selector && !selector.disabled;
        const hasFeatures = processingFeatures && processingFeatures[feature.featureKey.toLowerCase().replace(' ', '_')];

        if (key === 'base') {
            // Upscale: available when models exist, but 1x works without a model
            if (toggle) toggle.disabled = false;
            if (section) section.classList.remove('feature-unavailable');
            if (content) content.style.opacity = featureStates.base.enabled ? '1' : '0.5';
            if (badge) {
                badge.textContent = featureStates.base.enabled ? 'Enabled' : 'Disabled';
                badge.className = featureStates.base.enabled ? 'feature-badge available' : 'feature-badge';
            }
            return;
        }

        if (hasModels || hasFeatures) {
            section.classList.remove('feature-unavailable');
            if (toggle) toggle.disabled = false;
            if (content) content.style.opacity = featureStates[key].enabled ? '1' : '0.5';
            if (key === 'log') setLogRescueControlsEnabled(featureStates[key].enabled);
            if (badge) {
                badge.textContent = 'Available';
                badge.className = 'feature-badge available';
            }
        } else {
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
        const feature = FEATURES[key];
        const saved = localStorage.getItem(`enabled_${key}`);
        if (saved !== null) {
            featureStates[key].enabled = saved === 'true';
        }
        const toggle = document.getElementById(feature.toggleId);
        if (toggle) toggle.checked = featureStates[key].enabled;

        // Update content visibility
        const content = document.getElementById(feature.contentId);
        if (content) {
            content.style.opacity = featureStates[key].enabled ? '1' : '0.5';
            content.style.pointerEvents = featureStates[key].enabled ? 'auto' : 'none';
        }
        if (key === 'log') setLogRescueControlsEnabled(featureStates[key].enabled);
    });

    // Restore face fidelity slider
    const savedFidelity = localStorage.getItem('faceFidelity');
    const fidRange = document.getElementById('faceFidelity');
    if (savedFidelity !== null && fidRange) {
        fidRange.value = savedFidelity;
        const fidLabel = document.getElementById('faceFidelityValue');
        if (fidLabel) fidLabel.textContent = savedFidelity;
        updateRangeFill(fidRange);
    }
}

// Save preferences
function savePreferences() {
    Object.keys(FEATURES).forEach(key => {
        localStorage.setItem(`enabled_${key}`, featureStates[key].enabled);
        const selector = modelSelectors[key];
        if (selector && selector.value) {
            localStorage.setItem(`model_${key}`, selector.value);
        }
    });
    const fidRange = document.getElementById('faceFidelity');
    if (fidRange) {
        localStorage.setItem('faceFidelity', fidRange.value);
    }
}

// Collapsible feature sections
// Collapsed state is persisted per section id (e.g. "collapsed_section-face").
function setSectionCollapsed(section, collapsed) {
    if (!section) return;
    section.classList.toggle('collapsed', Boolean(collapsed));
    if (section.id) {
        localStorage.setItem(`collapsed_${section.id}`, collapsed ? 'true' : 'false');
    }
}

function setupCollapsibleSections() {
    document.querySelectorAll('.sidebar .feature-section').forEach(section => {
        const header = section.querySelector(':scope > .feature-header');
        const content = section.querySelector(':scope > .feature-content');
        if (!header || !content) return;

        // Chevron indicator (points right when collapsed)
        const chevron = document.createElement('span');
        chevron.className = 'feature-chevron';
        chevron.textContent = '▾';
        chevron.setAttribute('aria-hidden', 'true');
        header.appendChild(chevron);

        // Clicking the header toggles collapse, unless a control is clicked.
        header.addEventListener('click', (event) => {
            if (event.target.closest('.feature-toggle, input, select, button, a, label')) return;
            const toggleBox = header.querySelector('.feature-toggle input[type=checkbox]');
            // Disabled features stay collapsed.
            if (toggleBox && !toggleBox.checked) return;
            setSectionCollapsed(section, !section.classList.contains('collapsed'));
        });

        // Restore persisted state; disabled/unavailable sections always collapse.
        const toggleBox = header.querySelector('.feature-toggle input[type=checkbox]');
        const restored = localStorage.getItem(`collapsed_${section.id}`);
        if (toggleBox && !toggleBox.checked) {
            section.classList.add('collapsed');
        } else if (restored === 'true') {
            section.classList.add('collapsed');
        }
    });
}

// Setup event listeners
function setupEventListeners() {
    setupCollapsibleSections();

    // Range sliders: keep the filled track in sync with the current value.
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', () => updateRangeFill(slider));
        updateRangeFill(slider);
    });

    // GFPGAN has no fidelity knob -> disable the CodeFormer-only slider
    const faceModelSelect = document.getElementById('faceModelSelect');
    if (faceModelSelect) {
        const syncFidelity = () => {
            const fid = document.getElementById('faceFidelity');
            if (fid) fid.disabled = faceModelSelect.value !== 'CodeFormer';
        };
        faceModelSelect.addEventListener('change', syncFidelity);
        syncFidelity();
    }
    const logRescueToggle = document.getElementById('logRescueToggle');
    document.getElementById('logDenoiseStrength').addEventListener('input', event => {
        document.getElementById('logDenoiseValue').textContent =
            Number(event.target.value).toFixed(2);
    });
    document.getElementById('logShadowRecovery').addEventListener('input', event => {
        document.getElementById('logShadowValue').textContent =
            Number(event.target.value).toFixed(2);
    });

    // Feature toggles (includes base toggle now)
    Object.keys(FEATURES).forEach(key => {
        const feature = FEATURES[key];
        const toggle = document.getElementById(feature.toggleId);
        const content = document.getElementById(feature.contentId);
        const selector = modelSelectors[key];
        const strengthId = feature.strengthId;
        
        if (toggle) {
            toggle.addEventListener('change', () => {
                featureStates[key].enabled = toggle.checked;
                // Auto-collapse when disabled, expand when enabled
                setSectionCollapsed(document.getElementById(`section-${key}`), !toggle.checked);
                if (content) {
                    content.style.opacity = toggle.checked ? '1' : '0.5';
                    content.style.pointerEvents = toggle.checked ? 'auto' : 'none';
                }
                if (key === 'log') setLogRescueControlsEnabled(toggle.checked);
                // Update base badge when upscale is toggled
                if (key === 'base') {
                    const badge = document.getElementById(FEATURES.base.badgeId);
                    if (badge) {
                        badge.textContent = toggle.checked ? 'Enabled' : 'Disabled';
                        badge.className = toggle.checked ? 'feature-badge available' : 'feature-badge';
                    }
                }
                savePreferences();
            });
        }
        
        if (selector) {
            selector.addEventListener('change', () => {
                featureStates[key].model = selector.value;
                savePreferences();
            });
        }
        
        // Strength control
        if (strengthId) {
            const strengthInput = document.getElementById(strengthId);
            if (strengthInput) {
                strengthInput.addEventListener('input', () => {
                    featureStates[key].strength = strengthInput.value;
                    // Labels use either `<id>Value` (e.g. lowlightStrengthValue)
                    // or `<feature>Val` (e.g. denoiseVal); update whichever exists.
                    const label = document.getElementById(strengthId + 'Value')
                        || document.getElementById(strengthId.replace(/Strength$/, '') + 'Val');
                    if (label) label.textContent = strengthInput.value;
                });
            }
        }
    });

    // Scale dropdown — re-render labels when selection changes
    const upscaleSelect = document.getElementById('upscaleSelect');
    if (upscaleSelect) {
        upscaleSelect.addEventListener('change', () => updateScaleDropdown());
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
        input.style.display = 'none';
        input.onchange = (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        };
        document.body.appendChild(input);
        input.click();
        setTimeout(() => document.body.removeChild(input), 1000);
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
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4', 'video/webm', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
        alert('Unsupported file type. Please select JPG, PNG, WEBP, MP4, or WEBM.');
        return;
    }

    selectedFile = file;
    inputResolution = null;
    updateDropZoneUI(file);

    // Read resolution then update the scale dropdown
    if (file.type.startsWith('image/')) {
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
            inputResolution = { w: img.naturalWidth, h: img.naturalHeight };
            URL.revokeObjectURL(url);
            updateScaleDropdown();
        };
        img.onerror = () => URL.revokeObjectURL(url);
        img.src = url;
    } else if (file.type.startsWith('video/')) {
        const video = document.createElement('video');
        const url = URL.createObjectURL(file);
        video.onloadedmetadata = () => {
            inputResolution = { w: video.videoWidth, h: video.videoHeight };
            URL.revokeObjectURL(url);
            updateScaleDropdown();
        };
        video.onerror = () => URL.revokeObjectURL(url);
        video.src = url;
    }
}

// Update scale dropdown options to show actual output resolutions
function updateScaleDropdown() {
    const sel = document.getElementById('upscaleSelect');
    if (!sel) return;

    const currentVal = sel.value;
    const scales = [1, 2, 4, 8];

    sel.innerHTML = scales.map(s => {
        let label;
        if (inputResolution) {
            const outW = inputResolution.w * s;
            const outH = inputResolution.h * s;
            label = s === 1
                ? `1x (${outW}×${outH} — no upscale)`
                : `${s}x (${outW}×${outH})`;
        } else {
            label = s === 1 ? '1x (original resolution)' : `${s}x`;
        }
        return `<option value="${s}x">${label}</option>`;
    }).join('');

    // Restore previous selection if still valid, else default to 4x
    if ([...sel.options].some(o => o.value === currentVal)) {
        sel.value = currentVal;
    } else {
        sel.value = '4x';
    }
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
    selectedReferenceFile = null;
    inputResolution = null;
    updateScaleDropdown(); // reset labels to plain "1x / 2x / 4x / 8x"
    dropZone.innerHTML = `
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
        </svg>
        <h3>Drag & Drop Media Here</h3>
        <p style="margin-top: 0.5rem; font-size: 0.875rem;">Supports JPG, PNG, WEBP, MP4, WEBM</p>
    `;
    dropZone.style.borderColor = 'var(--border)';
    dropZone.style.backgroundColor = 'rgba(30, 30, 30, 0.5)';
}

// Strength sliders are 0-100 (percent); the processing APIs/pipelines expect 0-1.
function readUIScale(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return 0.5;
    const value = parseFloat(el.value);
    if (isNaN(value)) return 0.5;
    return Math.max(0, Math.min(1, value / 100));
}

// Filled track for a range input: white up to the current value, dark after.
function updateRangeFill(input) {
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || 100;
    const value = parseFloat(input.value);
    const pct = (!isNaN(value) && max > min)
        ? ((value - min) / (max - min)) * 100
        : 0;
    input.style.background =
        `linear-gradient(to right, #ffffff ${pct}%, #333333 ${pct}%)`;
}

// Start enhancement process
async function startEnhancement() {
    if (!selectedFile) {
        alert('Please select a media file first');
        return;
    }

    const upscaleEnabled = featureStates.base.enabled;
    const rawScaleValue = document.getElementById('upscaleSelect').value;
    const scale = (upscaleEnabled && rawScaleValue) ? parseInt(rawScaleValue) || 1 : 1;
    const baseModel = featureStates.base.model;
    const reencodeEnabled = featureStates.reencode.enabled;
    const logRescueEnabled = featureStates.log.enabled;

    // Gather all settings
    const settings = {
        inputFile: {
            name: selectedFile.name,
            type: selectedFile.type,
            size: selectedFile.size
        },
        upscaleEnabled: upscaleEnabled,
        baseModel: upscaleEnabled && scale > 1 ? baseModel : null,
        reencodeEnabled: reencodeEnabled,
        reencodeEngine: reencodeEnabled ? 'ffmpeg' : null,
        reencodePreset: reencodeEnabled ? 'h264_fast' : null,
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
        if (key === 'base' || key === 'reencode') return;

        if (featureStates[key].enabled && featureStates[key].model) {
            const strengthId = FEATURES[key].strengthId;
            settings.features[key] = {
                model: featureStates[key].model,
                strength: strengthId ? readUIScale(strengthId) : null
            };
        }
    });

    // Face fidelity weight (CodeFormer w-Factor): UI slider 0-100 -> 0-1
    if (settings.features.face) {
        settings.features.face.fidelityWeight = readUIScale('faceFidelity');
    }

    // Frame interpolation specific settings
    if (featureStates.frame.enabled) {
        const fpsSelect = document.getElementById('fpsSelect');
        const slowmoSelect = document.getElementById('slowmoSelect');
        const targetFps = parseInt((fpsSelect ? fpsSelect.value : '60').replace(/\D/g, '')) || 30;
        const slowFactor = parseFloat(slowmoSelect ? slowmoSelect.value : '1.0') || 1; // 1.0 / 0.5 / 0.25
        // RIFE's multiplier raises the output frame rate (output_fps = input_fps * multiplier).
        // Assuming a 30 fps source and compensating for the slow-motion factor, this keeps
        // the effective playback speed close to the selected "Target FPS".
        const multiplier = Math.max(2, Math.round((targetFps / 30) / slowFactor));
        // Spread the existing feature state (keeps model/strength) instead of
        // replacing it, so the selected model still reaches the backend.
        settings.features.frame = {
            ...settings.features.frame,
            fps: fpsSelect ? fpsSelect.value : '30',
            slowMotion: slowmoSelect ? slowmoSelect.value : '1.0x',
            multiplier
        };
    }

    // Match Color specific settings
    if (settings.features.color) {
        if (!selectedReferenceFile) {
            alert('Please select a reference image for color matching first.');
            return;
        }
        const preserveLuminance = document.getElementById('colorPreserveLuminance');
        settings.features.color.preserveLuminance =
            preserveLuminance ? preserveLuminance.checked : true;
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
    
    // Always include the scale value (1x when disabled, otherwise selected)
    settings.upscale = `${scale}x`;

    console.log('Enhancement settings:', settings);

    // Process using the actual API
    await processFile(settings);
}

// Internal step type -> backend route slug. These differ for several features
// (the backend uses the full feature name, e.g. "color" -> "match-color"), so
// the request URL must be mapped instead of using the step type verbatim.
const PROCESSING_ROUTES = {
    upscale: 'upscale',
    sharpen: 'sharpen',
    denoise: 'denoise',
    face: 'face',
    frame: 'frame-interpolation',
    color: 'match-color',
    lowlight: 'low-light',
    sdrtohdr: 'sdr-to-hdr',
    flatprofile: 'flat-profile',
    overexposure: 'overexposure-fix',
    reencode: 'reencode'
};

// Process file via API
async function processFile(settings) {
    const scale = parseInt(settings.upscale) || 1;
    const upscaleEnabled = settings.upscaleEnabled && scale > 1 && settings.baseModel;

    // Show processing UI
    showProcessingUI();

    try {
        let currentFile = selectedFile;
        let currentName = settings.inputFile.name;
        let currentType = settings.inputFile.type;
        const isVideo = currentType.startsWith('video/');
        
        // Build list of processing steps in order
        const steps = [];
        
        // 1. Upscale (or passthrough at scale=1)
        if (upscaleEnabled) {
            steps.push({ type: 'upscale', settings });
        }
        
        // 2. Sharpen/Deblur
        if (settings.features.sharpen) {
            steps.push({ type: 'sharpen', settings: settings.features.sharpen });
        }
        
        // 3. Denoise
        if (settings.features.denoise) {
            steps.push({ type: 'denoise', settings: settings.features.denoise });
        }
        
        // 4. Face Recovery
        if (settings.features.face) {
            steps.push({ type: 'face', settings: settings.features.face });
        }
        
        // 5. Frame Interpolation (video only)
        if (settings.features.frame && isVideo) {
            steps.push({ type: 'frame', settings: settings.features.frame });
        }
        
        // 6. Match Color
        if (settings.features.color) {
            steps.push({ type: 'color', settings: settings.features.color });
        }
        
        // 7. Low-Light Enhancement
        if (settings.features.lowlight) {
            steps.push({ type: 'lowlight', settings: settings.features.lowlight });
        }
        
        // 8. SDR to HDR
        if (settings.features.sdrtohdr) {
            steps.push({ type: 'sdrtohdr', settings: settings.features.sdrtohdr });
        }
        
        // 9. Flat Color Profile
        if (settings.features.flatprofile) {
            steps.push({ type: 'flatprofile', settings: settings.features.flatprofile });
        }
        
        // 10. Overexposure & Contrast Fix
        if (settings.features.overexposure) {
            steps.push({ type: 'overexposure', settings: settings.features.overexposure });
        }
        
        // 11. Reencoding (last step if enabled)
        if (settings.reencodeEnabled) {
            steps.push({ type: 'reencode', settings });
        }
        
        if (steps.length === 0) {
            throw new Error('No processing steps selected');
        }
        
        // Process each step sequentially
        for (let i = 0; i < steps.length; i++) {
            const step = steps[i];
            const isLastStep = (i === steps.length - 1);
            
            updateProcessingProgress(0, `Step ${i + 1}/${steps.length}: ${step.type}...`);
            
            const formData = new FormData();
            formData.append('file', currentFile);
            
            const endpoint = `/api/processing/${PROCESSING_ROUTES[step.type] || step.type}/file`;
            
            if (step.type === 'upscale') {
                formData.append('model_name', step.settings.baseModel || '');
                formData.append('scale', scale.toString());
                formData.append('device', getProcessingDevice());
                formData.append('output_format', settings.output.format);
                formData.append('jpg_quality', String(systemSettings.jpgQuality));
            } else if (step.type === 'sharpen') {
                formData.append('model_name', step.settings.model);
                formData.append('strength', step.settings.strength ?? 0.5);
                if (systemSettings.sharpenOverlap) {
                    formData.append('overlap', String(systemSettings.sharpenOverlap));
                }
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'denoise') {
                formData.append('model_name', step.settings.model);
                formData.append('strength', step.settings.strength ?? 0.5);
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'face') {
                formData.append('model_name', step.settings.model);
                formData.append('fidelity_weight', step.settings.fidelityWeight ?? 0.5);
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'frame') {
                formData.append('model_name', step.settings.model);
                formData.append('device', getProcessingDevice());
                formData.append('multiplier', step.settings.multiplier || 2);
            } else if (step.type === 'color') {
                formData.append('model_name', step.settings.model);
                formData.append('strength', step.settings.strength ?? 1.0);
                formData.append('preserve_luminance', step.settings.preserveLuminance ? 'true' : 'false');
                formData.append('device', getProcessingDevice());
                if (selectedReferenceFile) {
                    formData.append('reference', selectedReferenceFile);
                }
            } else if (step.type === 'lowlight') {
                formData.append('model_name', step.settings.model);
                formData.append('strength', step.settings.strength ?? 0.5);
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'sdrtohdr') {
                formData.append('model_name', step.settings.model);
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'flatprofile') {
                formData.append('model_name', step.settings.model);
                formData.append('strength', step.settings.strength ?? 0.5);
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'overexposure') {
                formData.append('model_name', step.settings.model);
                formData.append('strength', step.settings.strength ?? 0.5);
                formData.append('device', getProcessingDevice());
            } else if (step.type === 'reencode') {
                formData.append('engine', step.settings.reencodeEngine || 'ffmpeg');
                formData.append('preset', step.settings.reencodePreset || 'h264_fast');
            }
            
            const uploadResponse = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            if (!uploadResponse.ok) {
                const error = await uploadResponse.json();
                throw new Error(`${step.type}: ${error.detail || 'Upload failed'}`);
            }
            
            const { task_id } = await uploadResponse.json();
            const result = await pollProcessingTask(task_id, (p) => {
                const overall = ((i / steps.length) * 100) + (p / steps.length);
                updateProcessingProgress(overall, `Step ${i + 1}/${steps.length}: ${step.type}...`);
            });
            
            // Download intermediate result
            const resultResponse = await fetch(`/api/processing/result/${task_id}`);
            const resultBlob = await resultResponse.blob();
            
            if (!isLastStep) {
                // Create file object for next step
                currentFile = new File([resultBlob], 'intermediate.' + (isVideo ? 'mp4' : 'png'), {
                    type: isVideo ? 'video/mp4' : 'image/png'
                });
            } else {
                // Last step - download final result
                const url = URL.createObjectURL(resultBlob);
                const a = document.createElement('a');
                a.href = url;
                const ext = currentName.split('.').pop();
                const baseName = currentName.split('.').slice(0, -1).join('.');
                a.download = `${baseName}_processed.${ext}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        }
        
        hideProcessingUI();
        alert('Processing complete! File downloaded.');
        
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
async function downloadResult(taskId, originalName, scale, result, isReencode = false) {
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
            a.download = isReencode 
                ? `${baseName}_reencoded.${ext}`
                : `${baseName}_upscaled_${scale}x.${ext}`;
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