/**
 * Settings Page Logic for openVideoAI
 */
const API_BASE = '/api';

// DOM Elements
const settingsSaveStatus = document.getElementById('settingsSaveStatus');
const tabs = document.querySelectorAll('.settings-tab');
const tabPanels = document.querySelectorAll('.tab-panel');

// State
let settings = {};
let isDirty = false;
let aboutPopulated = false;

// Form fields that map 1:1 to settings keys (element id === settings key)
const SETTING_FIELDS = [
    // Performance
    'maxRamUsage', 'maxVramUsage', 'cpuThreads', 'tileSize',
    // GPU Acceleration
    'gpuDevice', 'fp16Enabled', 'tensorrtEnabled',
    // Processing defaults
    'defaultUpscale', 'defaultTileSize', 'outputFormat', 'jpgQuality', 'sharpenOverlap', 'sharpenUnsharpAmount',
    // Model management
    'modelDir', 'autoDownloadModels', 'verifyModels', 'hideIncompatibleModels', 'preferAi', 'advancedMode',
    // Appearance
    'theme', 'accentColor', 'fontSize', 'density',
    // Advanced
    'debugLogging', 'showMetrics', 'experimentalFeatures'
];

// Frontend defaults (mirror src/core/config.py DEFAULT_SETTINGS)
const SETTING_DEFAULTS = {
    maxRamUsage: 85, maxVramUsage: 95, cpuThreads: 8, tileSize: 400,
    computeBackend: 'auto', gpuDevice: 'auto',
    fp16Enabled: true, tensorrtEnabled: false,
    defaultUpscale: 4, defaultTileSize: 400,
    outputFormat: 'jpg', jpgQuality: 95, sharpenOverlap: 128, sharpenUnsharpAmount: 0.5,
    modelDir: './models',
    autoDownloadModels: true, verifyModels: true,
    hideIncompatibleModels: false, preferAi: true, advancedMode: false,
    theme: 'dark', accentColor: '#3b82f6',
    fontSize: 'medium', density: 'comfortable',
    debugLogging: false, showMetrics: true, experimentalFeatures: false
};

// Suffixes shown next to range sliders
const SLIDER_SUFFIX = {
    maxRamUsage: '%', maxVramUsage: '%', cpuThreads: ' threads',
    tileSize: ' px', jpgQuality: '%', sharpenOverlap: ' px', sharpenUnsharpAmount: '%'
};

function applySetting(key, value) {
    const el = document.getElementById(key);
    if (!el) return;
    if (el.type === 'checkbox') {
        el.checked = Boolean(value);
    } else {
        el.value = value;
    }
    if (el.type === 'range') {
        updateSliderLabel(el);
        updateRangeFill(el);
    }
}

function readSetting(key) {
    const el = document.getElementById(key);
    if (!el) return undefined;
    if (el.type === 'checkbox') return el.checked;
    if (el.type === 'range' || el.type === 'number') return Number(el.value);
    // Numeric select fields (defaultUpscale, defaultTileSize) must be sent as ints
    if (key === 'defaultUpscale' || key === 'defaultTileSize') return Number(el.value);
    return el.value;
}

function getComputeBackend() {
    const checked = document.querySelector('input[name="computeBackend"]:checked');
    return checked ? checked.value : 'auto';
}

function setComputeBackend(value) {
    document.querySelectorAll('input[name="computeBackend"]').forEach(radio => {
        radio.checked = radio.value === value;
    });
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

// Re-read one range slider's live value label.
function updateSliderLabel(input) {
    const label = document.getElementById(input.id + 'Value');
    if (!label) return;
    const suffix = SLIDER_SUFFIX[input.id] || '';
    const raw = Number(input.value);
    // sharpenUnsharpAmount is stored 0..2 but shown as a percentage.
    const display = (input.id === 'sharpenUnsharpAmount') ? Math.round(raw * 100) : raw;
    label.textContent = `${display}${suffix}`;
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSettings();
    setupTabNavigation();
    setupEventListeners();
    setupSettingsButtons();

    // Restore last active tab (defaults to first tab)
    const savedTab = localStorage.getItem('activeSettingsTab');
    if (savedTab && document.querySelector(`[data-tab="${savedTab}"]`)) {
        activateTab(savedTab);
    } else {
        // Activate the first tab
        const firstTab = tabs[0];
        if (firstTab) activateTab(firstTab.getAttribute('data-tab'));
    }
});

// Load settings from API
async function loadSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        settings = await response.json();
        
        // Update form values
        updateFormValues();
        // Paint the initial slider fills
        document.querySelectorAll('input[type="range"]').forEach(updateRangeFill);
        
        // Load system info
        await loadSystemInfo();
        
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

// Update form values from settings object
function updateFormValues() {
    const merged = { ...SETTING_DEFAULTS, ...settings };
    SETTING_FIELDS.forEach(key => applySetting(key, merged[key]));
    setComputeBackend(merged.computeBackend);
}

// Read every setting back out of the form controls (form -> settings object)
function collectFormValues() {
    const collected = { ...SETTING_DEFAULTS };
    SETTING_FIELDS.forEach(key => {
        const value = readSetting(key);
        if (value !== undefined) collected[key] = value;
    });
    collected.computeBackend = getComputeBackend();
    return collected;
}

// Setup tab navigation
function setupTabNavigation() {
    // Tab click handlers
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            activateTab(tabName);
        });
    });
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.altKey && e.key >= '1' && e.key <= '5') {
            const tabIndex = parseInt(e.key) - 1;
            if (tabIndex < tabs.length) {
                tabs[tabIndex].click();
            }
        }
    });
}

// Activate a tab
function activateTab(tabName) {
    // Update tab states
    tabs.forEach(tab => {
        const isActive = tab.getAttribute('data-tab') === tabName;
        tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
        tab.classList.toggle('active', isActive);
    });

    // Update panel states
    tabPanels.forEach(panel => {
        const isActive = panel.getAttribute('data-tab') === tabName;
        panel.hidden = !isActive;
    });

    // Scroll to top when tab changes
    document.querySelector('.settings-container').scrollTop = 0;

    // Save preference
    localStorage.setItem('activeSettingsTab', tabName);

    // Lazy-load About tab content the first time it's opened
    if (tabName === 'about' && !aboutPopulated) {
        aboutPopulated = true;
        populateFeatureStatus();
        populateModelCatalog();
    }
}

// Setup event listeners for form inputs
function setupEventListeners() {
    // Form inputs
    const formInputs = document.querySelectorAll('input, select');
    formInputs.forEach(input => {
        input.addEventListener('change', () => {
            isDirty = true;
            updateSaveStatus('You have unsaved changes', 'warning');
            if (input.type === 'range') updateRangeFill(input);
        });
        
        input.addEventListener('input', () => {
            isDirty = true;
            if (input.type === 'range') {
                updateSliderLabel(input);
                updateRangeFill(input);
            }
        });
    });
    
    // Save button
    const saveButton = document.getElementById('saveSettings');
    if (saveButton) {
        saveButton.addEventListener('click', saveSettings);
    }
}

// Setup settings buttons
function setupSettingsButtons() {
    const buttons = {
        clearModelCache: '/api/models/cache',
        verifyAllModels: '/api/models/verify',
        browseModelDir: null,
        exportSettings: null,
        importSettings: null,
        resetSettings: null
    };
    
    Object.entries(buttons).forEach(([id, endpoint]) => {
        const btn = document.getElementById(id);
        if (!btn) return;
        
        btn.addEventListener('click', async () => {
            if (id === 'clearModelCache') {
                await clearModelCache(btn);
            } else if (id === 'verifyAllModels') {
                await verifyAllModels(btn);
            } else if (id === 'resetSettings') {
                resetSettings();
            }
        });
    });
}

// Save settings to API
async function saveSettings() {
    try {
        const payload = collectFormValues();
        const response = await fetch(`${API_BASE}/settings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error('Failed to save settings');
        
        settings = payload;
        isDirty = false;
        updateSaveStatus('Settings saved', 'success');
        
        setTimeout(() => updateSaveStatus('', ''), 3000);
        
    } catch (error) {
        console.error('Error saving settings:', error);
        updateSaveStatus('save failed', 'error');
    }
}

// Update save status display
function updateSaveStatus(message, state) {
    if (!settingsSaveStatus) return;
    
    settingsSaveStatus.textContent = message;
    settingsSaveStatus.setAttribute('data-state', state);
}

// Clear model cache
async function clearModelCache(btn) {
    const originalText = btn.textContent;
    btn.textContent = 'Clearing...';
    btn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/models/cache`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to clear cache');
        
        updateSaveStatus('Model cache cleared', 'success');
    } catch (error) {
        console.error('Error clearing cache:', error);
        updateSaveStatus('Failed to clear cache', 'error');
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// Verify all models
async function verifyAllModels(btn) {
    const originalText = btn.textContent;
    btn.textContent = 'Verifying...';
    btn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/models/verify`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to verify models');
        
        const result = await response.json();
        updateSaveStatus(`Verified: ${result.verified} passed, ${result.failed} failed`, 'success');
    } catch (error) {
        console.error('Error verifying models:', error);
        updateSaveStatus('Failed to verify models', 'error');
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// Reset settings to defaults
function resetSettings() {
    if (!confirm('Reset all settings to defaults? This cannot be undone.')) return;

    // Apply the frontend defaults to the form, then persist them
    settings = { ...SETTING_DEFAULTS };
    updateFormValues();
    saveSettings();
}

// Load system information
async function loadSystemInfo() {
    try {
        const [infoResp, capsResp] = await Promise.all([
            fetch(`${API_BASE}/system/info`),
            fetch(`${API_BASE}/system/capabilities`)
        ]);
        const info = infoResp.ok ? await infoResp.json() : {};
        const caps = capsResp.ok ? await capsResp.json() : {};

        const cpu = info.cpu_details || {};
        const ram = info.ram_details || {};
        const os = info.os_details || {};
        const py = info.python_details || {};
        const gpus = Array.isArray(info.gpus) ? info.gpus : [];

        // Live hints next to the resource sliders
        if (document.getElementById('systemCpuInfo')) {
            document.getElementById('systemCpuInfo').textContent = cpu.cores ? `${cpu.cores} cores` : 'N/A';
        }
        if (document.getElementById('systemRamInfo')) {
            document.getElementById('systemRamInfo').textContent = ram.total_gb ? `${ram.total_gb} GB total` : 'N/A';
        }

        // First usable accelerator, falling back to an OS-level adapter
        const gpuEntry = gpus.find(g => g.selectable && g.device && g.backend !== 'cpu')
            || gpus.find(g => g.backend === 'hardware');

        // System info cards in the Advanced tab
        if (document.getElementById('cpuInfo')) {
            document.getElementById('cpuInfo').textContent = cpu.name || info.cpu || 'Unknown';
        }
        if (document.getElementById('ramInfo')) {
            document.getElementById('ramInfo').textContent = ram.total_gb
                ? `${ram.total_gb} GB (${ram.available_gb} GB free)`
                : (info.ram || 'N/A');
        }
        if (document.getElementById('gpuInfo')) {
            document.getElementById('gpuInfo').textContent = gpuEntry?.name || 'None detected';
        }
        if (document.getElementById('vramInfo')) {
            document.getElementById('vramInfo').textContent = gpuEntry?.vram ? `${gpuEntry.vram} GB` : 'N/A';
        }
        if (document.getElementById('systemVramInfo')) {
            document.getElementById('systemVramInfo').textContent =
                gpuEntry?.vram ? `${gpuEntry.vram} GB available` : 'N/A';
        }
        if (document.getElementById('osInfo')) {
            document.getElementById('osInfo').textContent = os.platform
                ? `${os.platform} ${os.release || os.version || ''}`
                : (info.os || 'Unknown');
        }
        if (document.getElementById('pythonInfo')) {
            document.getElementById('pythonInfo').textContent = py.version
                ? `${py.version} (${py.implementation || ''})`
                : (info.python || 'Unknown');
        }

        // Disable compute backends that do not exist on this machine
        applyBackendAvailability(caps, gpus);

        // GPU device selector: only real, selectable torch devices
        const gpuDeviceSelect = document.getElementById('gpuDevice');
        if (gpuDeviceSelect) {
            const selectable = gpus.filter(g => g.selectable && g.device && g.backend !== 'hardware');
            gpuDeviceSelect.innerHTML = '<option value="auto">Auto Select</option>';
            selectable.forEach(gpu => {
                const option = document.createElement('option');
                option.value = gpu.device;
                option.textContent = gpu.vram ? `${gpu.name} (${gpu.vram} GB)` : gpu.name;
                gpuDeviceSelect.appendChild(option);
            });
            if (selectable.length > 0) {
                gpuDeviceSelect.disabled = false;

                // Restore the saved GPU selection now that real options exist
                const savedDevice = settings.gpuDevice || 'auto';
                gpuDeviceSelect.value = [...gpuDeviceSelect.options].some(o => o.value === savedDevice)
                    ? savedDevice
                    : 'auto';
            }
        }

        // Expose the real system thread count as the CPU slider maximum
        const cpuThreadsInput = document.getElementById('cpuThreads');
        if (cpuThreadsInput && caps.total_cpu_threads) {
            cpuThreadsInput.max = Math.max(Number(cpuThreadsInput.max) || 100, caps.total_cpu_threads);
        }

    } catch (error) {
        console.error('Error loading system info:', error);
    }
}

// Gray out compute backends that are not available on this machine.
function applyBackendAvailability(caps, gpus) {
    const accelerator = caps || {};
    const gpuBackendHint = document.getElementById('gpuBackendHint');
    if (gpuBackendHint) {
        const active = gpus.find(g => g.selectable && g.device && g.backend !== 'cpu');
        gpuBackendHint.textContent = active
            ? `Detected: ${active.name} via ${backendLabel(active.backend)}`
            : 'No GPU accelerator detected — processing will use the CPU';
    }

    const availability = {
        cuda: !!accelerator.has_cuda,
        amd: !!(accelerator.has_hip || accelerator.has_dml),
        metal: !!accelerator.has_metal
    };

    document.querySelectorAll('.radio-option[data-requires]').forEach(option => {
        const requires = option.getAttribute('data-requires');
        const available = availability[requires] !== false;
        option.classList.toggle('backend-unavailable', !available);
        const input = option.querySelector('input[type="radio"]');
        if (input) input.disabled = !available;
        if (!available) {
            const desc = option.querySelector('.radio-desc');
            if (desc && !desc.textContent.includes('not detected')) {
                desc.textContent = `${desc.textContent.trim()} — not detected on this system`;
            }
        }
    });

    // If the saved backend is unavailable here, fall back to auto detect.
    const current = getComputeBackend();
    if (current !== 'auto' && current !== 'cpu' && availability[current] === false) {
        setComputeBackend('auto');
        isDirty = true;
        updateSaveStatus(' backend not available — switched to Auto Detect', 'warning');
    }
}

function backendLabel(backend) {
    const labels = {
        cuda: 'CUDA (NVIDIA)',
        hip: 'ROCm/HIP (AMD)',
        dml: 'DirectML (AMD/Intel)',
        metal: 'Metal (Apple)',
        cpu: 'CPU'
    };
    return labels[backend] || backend;
}

// Feature Status Table
async function populateFeatureStatus() {
    const tbody = document.getElementById('featureStatusBody');
    if (!tbody) return;
    
    // Get implemented features from API
    let implementedFeatures = {};
    try {
        const response = await fetch(`${API_BASE}/features/implemented`);
        if (response.ok) {
            implementedFeatures = await response.json();
        }
    } catch (e) {
        console.error('Could not load implemented features:', e);
    }
    
    // Feature definitions from models.json
    const features = [
        { name: 'Upscale', status: 'implemented', backend: 'CUDA/Metal/Python', notes: 'Real-ESRGAN, Real-CUGAN' },
        { name: 'Face Recovery', status: 'implemented', backend: 'CUDA', notes: 'CodeFormer' },
        { name: 'Sharpen / Deblur', status: 'implemented', backend: 'CPU/CUDA', notes: 'NAFNet-GoPro, NAFNet-REDS' },
        { name: 'Denoise', status: 'implemented', backend: 'CPU/CUDA', notes: 'NAFNet-SIDD, SCUNet' },
        { name: 'Frame Interpolation', status: 'implemented', backend: 'CPU/CUDA', notes: 'RIFE, FILM' },
        { name: 'SloMo', status: 'implemented', backend: 'CPU/CUDA', notes: 'RIFE-SloMo' },
        { name: 'Frame Recovery', status: 'implemented', backend: 'CUDA', notes: 'BasicVSR, EDVR' },
        { name: 'Multipurpose Restoration', status: 'implemented', backend: 'CUDA', notes: 'RealBasicVSR' },
        { name: 'Low-Light Enhancement', status: 'implemented', backend: 'CPU', notes: 'Zero-DCE' },
        { name: 'Correct Distortion', status: 'ui-only', backend: 'OpenCV', notes: 'Lens correction only' },
        { name: 'Colorization', status: 'ui-only', backend: 'OpenCV', notes: 'Basic channel mixing' },
        { name: 'Object Removal', status: 'ui-only', backend: 'OpenCV', notes: 'Inpainting only' },
        { name: 'Match Color', status: 'implemented', backend: 'CPU (Reinhard + WCT2 neural)', notes: 'Reinhard + WCT2; NLUT/DCT broken-checkpoint (fail-fast)' },
        { name: 'Log Restoration', status: 'implemented', backend: 'CPU', notes: 'ITMLUT, FastLLVE' },
        { name: 'SDR to HDR', status: 'ui-only', backend: 'OpenCV', notes: 'Simulated HDR effect' },
        { name: 'Focus Fix', status: 'ui-only', backend: 'OpenCV', notes: 'Wiener deconvolution' },
        { name: 'Overexposure & Contrast Fix', status: 'implemented', backend: 'OpenCV', notes: 'CLAHE, histogram equalization' },
        { name: 'Flat Color Profile', status: 'implemented', backend: 'OpenCV', notes: 'Gray World + saturation' }
    ];
    
    // Count by status
    const counts = { implemented: 0, partial: 0, planned: 0, uiOnly: 0 };
    
    tbody.innerHTML = features.map((feature, i) => {
        let statusClass = feature.status;
        if (statusClass === 'ui-only') {
            counts.uiOnly++;
            statusClass = 'ui-only';
        } else {
            counts.implemented++;
        }
        
        return `
            <tr>
                <td><strong>${feature.name}</strong></td>
                <td><span class="status-dot ${statusClass}"></span>${statusLabel(feature.status)}</td>
                <td>${feature.backend}</td>
                <td>${feature.notes}</td>
            </tr>
        `;
    }).join('');
    
    // Update stats
    document.getElementById('implementedCount').textContent = counts.implemented;
    document.getElementById('partialCount').textContent = counts.partial;
    document.getElementById('plannedCount').textContent = counts.planned;
    document.getElementById('uiOnlyCount').textContent = counts.uiOnly;
}

// Model Catalog Summary
async function populateModelCatalog() {
    // Total models - from models.json
    const totalModels = 45;
    const totalFeatures = 18;
    const totalModules = 2;
    
    document.getElementById('totalModels').textContent = totalModels;
    document.getElementById('totalFeatures').textContent = totalFeatures;
    document.getElementById('totalModules').textContent = totalModules;
    
    // Model counts by feature
    const modelCounts = [
        { feature: 'Upscale', count: 9, backend: 'CUDA' },
        { feature: 'Face Recovery', count: 5, backend: 'CUDA' },
        { feature: 'Sharpen', count: 8, backend: 'CPU/CUDA' },
        { feature: 'Denoise', count: 3, backend: 'CPU/CUDA' },
        { feature: 'Frame Interp', count: 6, backend: 'CPU/CUDA' },
        { feature: 'SlowMo', count: 3, backend: 'CPU/CUDA' }
    ];
    
    const grid = document.getElementById('modelCountsGrid');
    if (grid) {
        grid.innerHTML = modelCounts.map(mc => `
            <div class="model-count-card">
                <div class="count">${mc.count}</div>
                <div class="label">${mc.feature}</div>
                <div class="backend" style="font-size: 0.75rem; color: var(--text-muted);">${mc.backend}</div>
            </div>
        `).join('');
    }
}

// Utility functions
const STATUS_LABELS = {
    'implemented': 'Implemented',
    'partial':     'Partial',
    'planned':     'Planned',
    'ui-only':     'UI Only',
};

function statusLabel(status) {
    return STATUS_LABELS[status] ?? status.charAt(0).toUpperCase() + status.slice(1);
}