/**
 * Settings Page Logic for openVideoAI
 * Handles tab navigation, system resources, GPU acceleration, processing defaults,
 * model management, appearance, and advanced settings.
 */

const API_BASE = '/api';

// Default settings - matches the new HTML structure
const DEFAULT_SETTINGS = {
    // System Resources (RAM/VRAM as percentages, CPU as thread count)
    maxRamUsage: 85,
    maxVramUsage: 95,
    cpuThreads: 8,
    tileSize: 400,
    
    // GPU Acceleration
    computeBackend: 'auto', // auto, cuda, metal, cpu
    gpuDevice: 'auto',
    fp16Enabled: true,
    tensorrtEnabled: false,
    
    // Processing Defaults
    defaultUpscale: 4,
    defaultTileSize: 400,
    outputFormat: 'jpg',
    jpgQuality: 95,
    
    // Model Management
    modelDir: './models',
    autoDownloadModels: true,
    verifyModels: true,
    hideIncompatibleModels: false,
    preferAi: true,
    
    // Appearance
    theme: 'dark',
    accentColor: '#3b82f6',
    fontSize: 'medium',
    density: 'comfortable',
    
    // Advanced
    debugLogging: false,
    showMetrics: true,
    experimentalFeatures: false
};

// State
let settings = { ...DEFAULT_SETTINGS };
let saveTimer;

// DOM Elements
let elements = {};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initElements();
    setupTabNavigation();
    await loadSettings();
    applySettings();
    setupEventListeners();
    await loadSystemCapabilities();
    await detectSystemInfo();
});

// Initialize DOM element references
function initElements() {
    // Tab navigation
    elements.tabButtons = document.querySelectorAll('.settings-tab');
    elements.tabPanels = document.querySelectorAll('.tab-panel');
    
    // System Resources
    elements.maxRamUsage = document.getElementById('maxRamUsage');
    elements.maxRamUsageValue = document.getElementById('maxRamUsageValue');
    elements.maxVramUsage = document.getElementById('maxVramUsage');
    elements.maxVramUsageValue = document.getElementById('maxVramUsageValue');
    elements.cpuThreads = document.getElementById('cpuThreads');
    elements.cpuThreadsValue = document.getElementById('cpuThreadsValue');
    elements.tileSize = document.getElementById('tileSize');
    elements.tileSizeValue = document.getElementById('tileSizeValue');
    
    // GPU Acceleration
    elements.computeBackendRadios = document.querySelectorAll('input[name="computeBackend"]');
    elements.gpuDevice = document.getElementById('gpuDevice');
    elements.gpuDeviceHint = document.getElementById('gpuDeviceHint');
    elements.fp16Enabled = document.getElementById('fp16Enabled');
    elements.tensorrtEnabled = document.getElementById('tensorrtEnabled');
    
    // Processing Defaults
    elements.defaultUpscale = document.getElementById('defaultUpscale');
    elements.defaultTileSize = document.getElementById('defaultTileSize');
    elements.outputFormat = document.getElementById('outputFormat');
    elements.jpgQuality = document.getElementById('jpgQuality');
    elements.jpgQualityValue = document.getElementById('jpgQualityValue');
    
    // Model Management
    elements.modelDir = document.getElementById('modelDir');
    elements.browseModelDir = document.getElementById('browseModelDir');
    elements.autoDownloadModels = document.getElementById('autoDownloadModels');
    elements.verifyModels = document.getElementById('verifyModels');
    elements.hideIncompatibleModels = document.getElementById('hideIncompatibleModels');
    elements.preferAi = document.getElementById('preferAi');
    elements.clearModelCache = document.getElementById('clearModelCache');
    elements.verifyAllModels = document.getElementById('verifyAllModels');
    
    // Appearance
    elements.theme = document.getElementById('theme');
    elements.accentColor = document.getElementById('accentColor');
    elements.fontSize = document.getElementById('fontSize');
    elements.density = document.getElementById('density');
    
    // Advanced
    elements.debugLogging = document.getElementById('debugLogging');
    elements.showMetrics = document.getElementById('showMetrics');
    elements.experimentalFeatures = document.getElementById('experimentalFeatures');
    elements.exportSettings = document.getElementById('exportSettings');
    elements.importSettings = document.getElementById('importSettings');
    elements.resetSettings = document.getElementById('resetSettings');
    elements.saveStatus = document.getElementById('settingsSaveStatus');
    
    // System Info
    elements.cpuInfo = document.getElementById('cpuInfo');
    elements.ramInfo = document.getElementById('ramInfo');
    elements.gpuInfo = document.getElementById('gpuInfo');
    elements.vramInfo = document.getElementById('vramInfo');
    elements.osInfo = document.getElementById('osInfo');
    elements.pythonInfo = document.getElementById('pythonInfo');
}

// Tab Navigation
function setupTabNavigation() {
    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
        
        // Keyboard navigation
        button.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                button.click();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                focusNextTab(button);
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                focusPrevTab(button);
            }
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    elements.tabButtons.forEach(btn => {
        const isActive = btn.dataset.tab === tabName;
        btn.setAttribute('aria-selected', isActive);
        btn.setAttribute('aria-controls', `tab-${tabName}`);
    });
    
    // Update panels
    elements.tabPanels.forEach(panel => {
        const isActive = panel.dataset.tab === tabName;
        panel.hidden = !isActive;
    });
    
    // Update URL hash for deep linking (optional)
    history.replaceState(null, '', `#${tabName}`);
}

function focusNextTab(currentButton) {
    const buttons = Array.from(elements.tabButtons);
    const currentIndex = buttons.indexOf(currentButton);
    const nextIndex = (currentIndex + 1) % buttons.length;
    buttons[nextIndex].focus();
}

function focusPrevTab(currentButton) {
    const buttons = Array.from(elements.tabButtons);
    const currentIndex = buttons.indexOf(currentButton);
    const prevIndex = (currentIndex - 1 + buttons.length) % buttons.length;
    buttons[prevIndex].focus();
}

// Check URL hash on load and activate corresponding tab
function activateTabFromHash() {
    const hash = window.location.hash.slice(1);
    if (hash && document.querySelector(`.settings-tab[data-tab="${hash}"]`)) {
        switchTab(hash);
    }
}

// Load persisted server settings
async function loadSettings() {
    try {
        const stored = localStorage.getItem('openvideoai_settings');
        if (stored) {
            settings = { ...settings, ...JSON.parse(stored) };
        }
    } catch (e) {
        console.warn('Failed to load legacy browser settings:', e);
        setSaveStatus('Could not read browser-saved settings.', 'error');
    }

    try {
        const response = await fetch(`${API_BASE}/settings`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const persistedSettings = await response.json();
        settings = { ...DEFAULT_SETTINGS, ...settings, ...persistedSettings };
    } catch (e) {
        console.error('Failed to load settings from server:', e);
        setSaveStatus('Server settings unavailable; using local preferences.', 'error');
    }
}

function setSaveStatus(message, state = '') {
    if (!elements.saveStatus) return;
    elements.saveStatus.textContent = message;
    elements.saveStatus.dataset.state = state;
}

// Save locally and persist to backend
function saveSettings() {
    try {
        localStorage.setItem('openvideoai_settings', JSON.stringify(settings));
    } catch (e) {
        console.error('Failed to save browser settings:', e);
        setSaveStatus(`Browser save failed: ${e.message}`, 'error');
    }

    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
        setSaveStatus('Saving settings…');
        try {
            const response = await fetch(`${API_BASE}/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            if (!response.ok) {
                const detail = await response.json().catch(() => ({}));
                throw new Error(detail.detail || `Server returned ${response.status}`);
            }
            setSaveStatus('Settings saved.', 'success');
        } catch (e) {
            console.error('Failed to save settings to server:', e);
            setSaveStatus(`Settings were not saved on server: ${e.message}`, 'error');
        }
    }, 250);
}

// Apply settings to UI elements
function applySettings() {
    // System Resources
    setSliderValue(elements.maxRamUsage, elements.maxRamUsageValue, settings.maxRamUsage, '%');
    setSliderValue(elements.maxVramUsage, elements.maxVramUsageValue, settings.maxVramUsage, '%');
    setSliderValue(elements.cpuThreads, elements.cpuThreadsValue, settings.cpuThreads, ' threads');
    setSliderValue(elements.tileSize, elements.tileSizeValue, settings.tileSize, ' px');
    
    // GPU Acceleration
    elements.computeBackendRadios.forEach(radio => {
        radio.checked = radio.value === settings.computeBackend;
    });
    void updateGpuDeviceOptions();
    elements.fp16Enabled.checked = settings.fp16Enabled;
    elements.tensorrtEnabled.checked = settings.tensorrtEnabled;
    
    // Processing Defaults
    elements.defaultUpscale.value = settings.defaultUpscale;
    elements.defaultTileSize.value = settings.defaultTileSize;
    elements.outputFormat.value = settings.outputFormat;
    setSliderValue(elements.jpgQuality, elements.jpgQualityValue, settings.jpgQuality, '%');
    
    // Model Management
    elements.modelDir.value = settings.modelDir;
    elements.autoDownloadModels.checked = settings.autoDownloadModels;
    elements.verifyModels.checked = settings.verifyModels;
    elements.hideIncompatibleModels.checked = settings.hideIncompatibleModels;
    elements.preferAi.checked = settings.preferAi;
    
    // Appearance
    elements.theme.value = settings.theme;
    elements.accentColor.value = settings.accentColor;
    document.querySelector('.color-value').textContent = settings.accentColor;
    elements.fontSize.value = settings.fontSize;
    elements.density.value = settings.density;
    
    // Advanced
    elements.debugLogging.checked = settings.debugLogging;
    elements.showMetrics.checked = settings.showMetrics;
    elements.experimentalFeatures.checked = settings.experimentalFeatures;
    
    // Apply styling immediately
    applyTheme(settings.theme);
    applyAccentColor(settings.accentColor);
    applyFontSize(settings.fontSize);
    applyDensity(settings.density);
    
    // Activate tab from URL hash
    activateTabFromHash();
}

// Helper to set slider value and display
function setSliderValue(slider, display, value, suffix) {
    if (slider && display && value !== undefined && value !== null) {
        slider.value = value;
        display.textContent = value + suffix;
    }
}

// Setup all event listeners
function setupEventListeners() {
    // System Resources sliders
    setupSlider(elements.maxRamUsage, elements.maxRamUsageValue, 'maxRamUsage', '%');
    setupSlider(elements.maxVramUsage, elements.maxVramUsageValue, 'maxVramUsage', '%');
    setupSlider(elements.cpuThreads, elements.cpuThreadsValue, 'cpuThreads', ' threads');
    setupSlider(elements.tileSize, elements.tileSizeValue, 'tileSize', ' px');
    setupSlider(elements.jpgQuality, elements.jpgQualityValue, 'jpgQuality', '%');
    
    // GPU Acceleration radios
    elements.computeBackendRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            settings.computeBackend = e.target.value;
            settings.gpuDevice = 'auto';
            saveSettings();
            void updateGpuDeviceOptions();
        });
    });
    
    elements.gpuDevice.addEventListener('change', (e) => {
        settings.gpuDevice = e.target.value;
        saveSettings();
    });
    
    elements.fp16Enabled.addEventListener('change', (e) => {
        settings.fp16Enabled = e.target.checked;
        saveSettings();
    });
    
    elements.tensorrtEnabled.addEventListener('change', (e) => {
        settings.tensorrtEnabled = e.target.checked;
        saveSettings();
    });
    
    // Processing Defaults
    elements.defaultUpscale.addEventListener('change', (e) => {
        settings.defaultUpscale = parseInt(e.target.value);
        saveSettings();
    });
    elements.defaultTileSize.addEventListener('change', (e) => {
        settings.defaultTileSize = parseInt(e.target.value);
        saveSettings();
    });
    elements.outputFormat.addEventListener('change', (e) => {
        settings.outputFormat = e.target.value;
        saveSettings();
    });
    
    // Model Management
    elements.browseModelDir.addEventListener('click', browseModelDirectory);
    elements.autoDownloadModels.addEventListener('change', (e) => {
        settings.autoDownloadModels = e.target.checked;
        saveSettings();
    });
    elements.verifyModels.addEventListener('change', (e) => {
        settings.verifyModels = e.target.checked;
        saveSettings();
    });
    elements.hideIncompatibleModels.addEventListener('change', (e) => {
        settings.hideIncompatibleModels = e.target.checked;
        saveSettings();
    });
    elements.preferAi.addEventListener('change', (e) => {
        settings.preferAi = e.target.checked;
        saveSettings();
    });
    elements.clearModelCache.addEventListener('click', clearModelCache);
    elements.verifyAllModels.addEventListener('click', verifyAllModels);
    
    // Appearance
    elements.theme.addEventListener('change', (e) => {
        settings.theme = e.target.value;
        saveSettings();
        applyTheme(settings.theme);
    });
    
    elements.accentColor.addEventListener('input', (e) => {
        settings.accentColor = e.target.value;
        document.querySelector('.color-value').textContent = settings.accentColor;
        saveSettings();
        applyAccentColor(settings.accentColor);
    });
    
    elements.fontSize.addEventListener('change', (e) => {
        settings.fontSize = e.target.value;
        saveSettings();
        applyFontSize(settings.fontSize);
    });
    
    elements.density.addEventListener('change', (e) => {
        settings.density = e.target.value;
        saveSettings();
        applyDensity(settings.density);
    });
    
    // Advanced
    elements.debugLogging.addEventListener('change', (e) => {
        settings.debugLogging = e.target.checked;
        saveSettings();
    });
    
    elements.showMetrics.addEventListener('change', (e) => {
        settings.showMetrics = e.target.checked;
        saveSettings();
    });
    
    elements.experimentalFeatures.addEventListener('change', (e) => {
        settings.experimentalFeatures = e.target.checked;
        saveSettings();
    });
    
    // Advanced Actions
    elements.exportSettings.addEventListener('click', exportSettings);
    elements.importSettings.addEventListener('click', importSettings);
    elements.resetSettings.addEventListener('click', resetSettings);
}

// Helper to setup slider with display update
function setupSlider(slider, display, key, suffix) {
    if (!slider || !display) return;
    
    slider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        settings[key] = value;
        display.textContent = value + suffix;
    });
    
    slider.addEventListener('change', (e) => {
        settings[key] = parseInt(e.target.value);
        saveSettings();
    });
}

// Update GPU device dropdown based on backend
async function updateGpuDeviceOptions() {
    const backend = settings.computeBackend;
    const select = elements.gpuDevice;
    const hint = elements.gpuDeviceHint;
    
    if (backend === 'cpu') {
        select.disabled = true;
        select.innerHTML = '<option value="cpu">CPU Only</option>';
        hint.textContent = 'CPU mode selected - no GPU device needed';
        return;
    }
    
    select.disabled = false;
    select.replaceChildren(new Option('Auto Select', 'auto'));
    hint.textContent = 'Detecting available GPU devices…';

    try {
        const response = await fetch(`${API_BASE}/system/gpus`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const data = await response.json();
        // Backend aliases: 'amd' covers DirectML and ROCm/HIP accelerators.
        // OS-level adapter entries (selectable === false) carry no torch
        // device string and are never offered in this dropdown.
        const allowedBackends = backend === 'auto'
            ? ['cuda', 'hip', 'dml', 'metal']
            : backend === 'amd'
                ? ['hip', 'dml']
                : [backend];
        const devices = (data.gpus || []).filter(gpu =>
            gpu.selectable !== false &&
            gpu.device && gpu.device !== 'cpu' &&
            allowedBackends.includes(gpu.backend)
        );

        for (const gpu of devices) {
            const option = new Option(
                `${gpu.name} (${gpu.vram ? `${gpu.vram} GB VRAM` : 'shared memory'})`,
                gpu.device
            );
            select.add(option);
        }

        if ([...select.options].some(option => option.value === settings.gpuDevice)) {
            select.value = settings.gpuDevice;
        } else {
            select.value = 'auto';
            settings.gpuDevice = 'auto';
        }
        hint.textContent = devices.length > 0
            ? 'Auto Select chooses the best available device; choose a device to pin it.'
            : `No ${backend.toUpperCase()} devices detected.`;
    } catch (e) {
        console.error('GPU detection failed:', e);
        hint.textContent = `GPU detection failed: ${e.message}`;
    }
}

// Load system capabilities and configure resource sliders + backend availability
async function loadSystemCapabilities() {
    try {
        const response = await fetch(`${API_BASE}/system/capabilities`);
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const caps = await response.json();

        // RAM / VRAM sliders are percentages of system max (0-100)
        const totalRamGb = caps.total_ram_gb || 0;
        const totalVramGb = caps.total_vram_gb || 0;

        if (elements.maxRamUsage) {
            elements.maxRamUsage.min = 1;
            elements.maxRamUsage.max = 100;
            elements.maxRamUsage.step = 1;
        }
        const ramHint = document.getElementById('maxRamUsageHint');
        if (ramHint) {
            ramHint.innerHTML = `Percentage of system RAM allocated for processing. System RAM: <strong>${totalRamGb} GB</strong>`;
        }

        if (elements.maxVramUsage) {
            elements.maxVramUsage.min = 0;
            elements.maxVramUsage.max = 100;
            elements.maxVramUsage.step = 1;
        }
        const vramHint = document.getElementById('maxVramUsageHint');
        if (vramHint) {
            vramHint.innerHTML = `Percentage of GPU VRAM allocated for processing. Set to 0 for no limit. ${totalVramGb > 0 ? `Total VRAM: <strong>${totalVramGb} GB</strong>` : 'No dedicated GPU VRAM detected.'}`;
        }

        // CPU slider is in units of the maximum threads
        const totalThreads = Math.max(1, caps.total_cpu_threads || 1);
        if (elements.cpuThreads) {
            elements.cpuThreads.min = 1;
            elements.cpuThreads.max = totalThreads;
            elements.cpuThreads.step = 1;
        }
        const cpuHint = document.getElementById('cpuThreadsHint');
        if (cpuHint) {
            cpuHint.innerHTML = `CPU threads (of a maximum <strong>${totalThreads}</strong>) for preprocessing.`;
        }
        // Clamp persisted value to the real hardware thread count
        if (settings.cpuThreads > totalThreads) {
            settings.cpuThreads = totalThreads;
        }
        setSliderValue(elements.cpuThreads, elements.cpuThreadsValue, settings.cpuThreads, ' threads');

        // Gate GPU backends to hardware that actually supports them
        const hasCuda = !!caps.has_cuda;
        const hasHip = !!caps.has_hip;
        const hasDml = !!caps.has_dml;
        const hasAmd = hasHip || hasDml;
        const hasMetal = !!caps.has_metal;

        const cudaOption = document.getElementById('cudaBackendOption');
        const amdOption = document.getElementById('amdBackendOption');
        const metalOption = document.getElementById('metalBackendOption');
        const backendHint = document.getElementById('gpuBackendHint');

        if (cudaOption) {
            const cudaRadio = cudaOption.querySelector('input');
            cudaOption.classList.toggle('backend-unavailable', !hasCuda);
            cudaRadio.disabled = !hasCuda;
            const desc = cudaOption.querySelector('.radio-desc');
            if (desc) desc.textContent = hasCuda
                ? 'Force NVIDIA CUDA acceleration'
                : 'Unavailable: no NVIDIA CUDA-capable GPU detected';
        }

        if (amdOption) {
            const amdRadio = amdOption.querySelector('input');
            amdOption.classList.toggle('backend-unavailable', !hasAmd);
            amdRadio.disabled = !hasAmd;
            const desc = amdOption.querySelector('.radio-desc');
            const amdHardware = !!caps.amd_hardware_detected;
            if (desc) desc.textContent = hasAmd
                ? `Force AMD GPU acceleration (${hasHip ? 'ROCm/HIP' : 'DirectML'})`
                : amdHardware
                    ? 'AMD GPU detected — install torch-directml (pip install torch-directml) and restart to enable acceleration'
                    : 'Unavailable: no AMD GPU / DirectML backend detected. Requires torch-directml or a ROCm PyTorch build.';
        }

        if (metalOption) {
            const metalRadio = metalOption.querySelector('input');
            metalOption.classList.toggle('backend-unavailable', !hasMetal);
            metalRadio.disabled = !hasMetal;
            const desc = metalOption.querySelector('.radio-desc');
            if (desc) desc.textContent = hasMetal
                ? 'Force Apple Metal acceleration'
                : 'Unavailable: only supported on Apple Silicon devices';
        }

        // If the persisted backend is not available on this machine, fall back to auto
        let backendChanged = false;
        if (!hasCuda && settings.computeBackend === 'cuda') {
            settings.computeBackend = 'auto';
            backendChanged = true;
        }
        if (!hasAmd && ['amd', 'hip', 'dml'].includes(settings.computeBackend)) {
            settings.computeBackend = 'auto';
            backendChanged = true;
        }
        if (!hasMetal && settings.computeBackend === 'metal') {
            settings.computeBackend = 'auto';
            backendChanged = true;
        }
        if (backendChanged) {
            settings.gpuDevice = 'auto';
        }
        elements.computeBackendRadios.forEach(radio => {
            radio.checked = radio.value === settings.computeBackend;
        });

        if (backendHint) {
            const detected = [];
            if (hasCuda) detected.push('CUDA');
            if (hasAmd) detected.push(`AMD ${hasHip ? '(ROCm/HIP)' : '(DirectML)'}`);
            if (hasMetal) detected.push('Metal');
            const amdHardware = !!caps.amd_hardware_detected;
            backendHint.textContent = detected.length > 0
                ? `Detected acceleration: ${detected.join(', ')}. Unsupported backends are disabled.`
                : amdHardware
                    ? 'AMD GPU detected but no DirectML backend is installed. Install torch-directml and restart to enable GPU acceleration.'
                    : 'No GPU acceleration detected on this system. Only CPU processing is available.';
        }

        if (backendChanged) {
            setSaveStatus('Backend not available on this system — switched to Auto Detect.', 'success');
            saveSettings();
            void updateGpuDeviceOptions();
        }
    } catch (e) {
        console.error('Failed to load system capabilities:', e);
    }
}

// Theme application
function applyTheme(theme) {
    const root = document.documentElement;
    if (theme === 'system') {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    } else {
        root.setAttribute('data-theme', theme);
    }
}

function applyAccentColor(color) {
    const root = document.documentElement;
    root.style.setProperty('--primary', color);
    const darker = darkenColor(color, 0.15);
    root.style.setProperty('--primary-hover', darker);
}

function applyFontSize(size) {
    const root = document.documentElement;
    const sizes = {
        small: '12px',
        medium: '14px',
        large: '16px'
    };
    root.style.setProperty('--base-font-size', sizes[size] || sizes.medium);
}

function applyDensity(density) {
    const root = document.documentElement;
    root.setAttribute('data-density', density);
}

// Color utility
function darkenColor(hex, amount) {
    const num = parseInt(hex.replace('#', ''), 16);
    const r = Math.max(0, Math.floor(((num >> 16) & 0xFF) * (1 - amount)));
    const g = Math.max(0, Math.floor(((num >> 8) & 0xFF) * (1 - amount)));
    const b = Math.max(0, Math.floor((num & 0xFF) * (1 - amount)));
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Hardware detection
async function detectSystemInfo() {
    try {
        const response = await fetch(`${API_BASE}/system/info`);
        if (response.ok) {
            const data = await response.json();
            updateSystemInfo(data);
        } else {
            detectClientSide();
        }
    } catch (e) {
        console.warn('System info detection failed:', e);
        detectClientSide();
    }
}

function detectClientSide() {
    const cpuCores = navigator.hardwareConcurrency || 'Unknown';
    if (elements.cpuInfo) elements.cpuInfo.textContent = cpuCores + ' cores';
    
    const ramGB = navigator.deviceMemory ? navigator.deviceMemory + ' GB' : 'Unknown';
    if (elements.ramInfo) elements.ramInfo.textContent = ramGB;
    
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    let gpuInfo = 'Unknown';
    let vramInfo = 'Unknown';
    
    if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
            gpuInfo = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        }
    }
    
    if (elements.gpuInfo) elements.gpuInfo.textContent = gpuInfo;
    if (elements.vramInfo) elements.vramInfo.textContent = vramInfo;
    if (elements.osInfo) elements.osInfo.textContent = navigator.platform;
    if (elements.pythonInfo) elements.pythonInfo.textContent = 'Unknown (check backend)';
}

function updateSystemInfo(data) {
    if (data.cpu && elements.cpuInfo) elements.cpuInfo.textContent = data.cpu;
    if (data.ram && elements.ramInfo) elements.ramInfo.textContent = data.ram;
    if (data.gpu && elements.gpuInfo) elements.gpuInfo.textContent = data.gpu;
    if (data.vram && elements.vramInfo) elements.vramInfo.textContent = data.vram;
    if (data.os && elements.osInfo) elements.osInfo.textContent = data.os;
    if (data.python && elements.pythonInfo) elements.pythonInfo.textContent = 'Python ' + data.python;
}

// Model Management Actions
async function browseModelDirectory() {
    const newDir = prompt('Enter model directory path:', settings.modelDir);
    if (newDir && newDir.trim()) {
        settings.modelDir = newDir.trim();
        elements.modelDir.value = settings.modelDir;
        saveSettings();
        try {
            await fetch(API_BASE + '/settings/model-dir', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: settings.modelDir })
            });
        } catch (e) {
            console.warn('Failed to update model dir on backend:', e);
        }
    }
}

async function clearModelCache() {
    if (!confirm('Clear model cache? This will not delete downloaded models, only temporary files.')) return;
    
    try {
        const response = await fetch(API_BASE + '/models/cache', { method: 'DELETE' });
        if (!response.ok) {
            const result = await response.json().catch(() => ({}));
            throw new Error(result.detail || `Server returned ${response.status}`);
        }
        alert('Model cache cleared');
    } catch (e) {
        alert('Failed to clear cache: ' + e.message);
    }
}

async function verifyAllModels() {
    if (!confirm('Verify all downloaded models? This may take a moment.')) return;
    
    try {
        const response = await fetch(API_BASE + '/models/verify', { method: 'POST' });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Server returned ${response.status}`);
        }
        const result = await response.json();
        alert('Verification complete: ' + result.verified + ' verified, ' + result.failed + ' failed');
    } catch (e) {
        alert('Verification failed: ' + e.message);
    }
}

// Settings Import/Export
function exportSettings() {
    const data = JSON.stringify(settings, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'openvideoai-settings.json';
    a.click();
    URL.revokeObjectURL(url);
}

function importSettings() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const imported = JSON.parse(e.target.result);
                settings = { ...DEFAULT_SETTINGS, ...imported };
                saveSettings();
                applySettings();
                alert('Settings imported successfully');
            } catch (e) {
                alert('Failed to import settings: ' + e.message);
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

function resetSettings() {
    if (!confirm('Reset all settings to defaults? This cannot be undone.')) return;
    
    settings = { ...DEFAULT_SETTINGS };
    saveSettings();
    applySettings();
    alert('Settings reset to defaults');
}

// Listen for system theme changes
if (window.matchMedia) {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', (e) => {
        if (settings.theme === 'system') {
            applyTheme('system');
        }
    });
}