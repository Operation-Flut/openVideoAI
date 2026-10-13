import React, { useState, useEffect } from 'react';
import { Settings, defaultSettings } from '../../types/settings';
import { settingsAPI } from '../../services/api';

type TabId = 'general' | 'gpu' | 'output' | 'advanced';

export const SettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [activeTab, setActiveTab] = useState<TabId>('general');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await settingsAPI.getSettings();
        setSettings({ ...defaultSettings, ...response });
      } catch (err) {
        console.error('Failed to load settings:', err);
      }
    };
    loadSettings();
  }, []);
  
  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await settingsAPI.updateSettings(settings);
      setMessage({ type: 'success', text: 'Settings saved successfully' });
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to save settings' });
    } finally {
      setSaving(false);
    }
  };
  
  const handleChange = (key: keyof Settings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };
  
  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: '⚙️' },
    { id: 'gpu' as TabId, label: 'GPU / Compute', icon: '🎮' },
    { id: 'output' as TabId, label: 'Output', icon: '💾' },
    { id: 'advanced' as TabId, label: 'Advanced', icon: '🔧' },
  ];
  
  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Tab Navigation */}
      <div className="border-b bg-gray-50">
        <nav className="flex" aria-label="Settings tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors
                ${activeTab === tab.id 
                  ? 'border-blue-500 text-blue-600 bg-white' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }
              `}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>
      
      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'general' && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h3 className="font-medium text-gray-900 mb-4">General Settings</h3>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium mb-1">Model Directory</label>
                  <input
                    type="text"
                    value={settings.modelDir}
                    onChange={(e) => handleChange('modelDir', e.target.value)}
                    className="w-full p-2 border rounded"
                    placeholder="models"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Export Directory</label>
                  <input
                    type="text"
                    value={settings.exportDir}
                    onChange={(e) => handleChange('exportDir', e.target.value)}
                    className="w-full p-2 border rounded"
                    placeholder="exports"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">CPU Threads</label>
                  <input
                    type="number"
                    value={settings.cpuThreads}
                    onChange={(e) => handleChange('cpuThreads', parseInt(e.target.value))}
                    min="1"
                    max="64"
                    className="w-full p-2 border rounded"
                  />
                </div>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'gpu' && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h3 className="font-medium text-gray-900 mb-4">Compute Backend</h3>
              <div className="space-y-3">
                {[
                  { value: 'auto', label: 'Auto', desc: 'Automatically select best available' },
                  { value: 'cuda', label: 'CUDA (NVIDIA)', desc: 'NVIDIA GPU acceleration' },
                  { value: 'metal', label: 'Metal (Apple)', desc: 'Apple Silicon GPU' },
                  { value: 'hip', label: 'HIP (AMD)', desc: 'AMD GPU on Linux' },
                  { value: 'dml', label: 'DirectML (Windows)', desc: 'Microsoft DirectML' },
                  { value: 'cpu', label: 'CPU', desc: 'CPU-only processing' },
                ].map((opt) => (
                  <label key={opt.value} className="flex items-center gap-3 p-3 border rounded hover:bg-gray-50 cursor-pointer">
                    <input
                      type="radio"
                      name="computeBackend"
                      value={opt.value}
                      checked={settings.computeBackend === opt.value}
                      onChange={(e) => handleChange('computeBackend', e.target.value as Settings['computeBackend'])}
                      className="text-blue-600"
                    />
                    <div>
                      <div className="font-medium">{opt.label}</div>
                      <div className="text-sm text-gray-500">{opt.desc}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
            
            <div>
              <h3 className="font-medium text-gray-900 mb-4">GPU Device</h3>
              <select
                value={settings.gpuDevice}
                onChange={(e) => handleChange('gpuDevice', e.target.value)}
                className="w-full md:w-1/2 p-2 border rounded"
              >
                <option value="auto">Auto</option>
                <option value="cuda:0">CUDA:0</option>
                <option value="cuda:1">CUDA:1</option>
                <option value="mps">Metal (MPS)</option>
                <option value="hip:0">HIP:0</option>
                <option value="dml">DirectML</option>
              </select>
            </div>
            
            <div>
              <h3 className="font-medium text-gray-900 mb-4">VRAM Usage Limit</h3>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={settings.maxVramUsage}
                  onChange={(e) => handleChange('maxVramUsage', parseInt(e.target.value))}
                  className="flex-1"
                />
                <span className="font-mono text-lg">{settings.maxVramUsage}%</span>
              </div>
              <p className="text-sm text-gray-500 mt-1">Maximum percentage of GPU VRAM to use</p>
            </div>
            
            <div>
              <h3 className="font-medium text-gray-900 mb-4">FP16 Precision</h3>
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={settings.fp16Enabled}
                  onChange={(e) => handleChange('fp16Enabled', e.target.checked)}
                  className="rounded text-blue-600"
                />
                <span>Enable half-precision (FP16) for faster inference on supported GPUs</span>
              </label>
            </div>
          </div>
        )}
        
        {activeTab === 'output' && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h3 className="font-medium text-gray-900 mb-4">Output Format</h3>
              <select
                value={settings.outputFormat}
                onChange={(e) => handleChange('outputFormat', e.target.value as Settings['outputFormat'])}
                className="w-full md:w-1/2 p-2 border rounded"
              >
                <option value="source">Keep source format</option>
                <option value="jpg">JPG</option>
                <option value="png">PNG</option>
                <option value="webp">WebP</option>
                <option value="tiff">TIFF</option>
                <option value="mp4">MP4 (video only)</option>
              </select>
            </div>
            
            <div>
              <h3 className="font-medium text-gray-900 mb-4">JPG Quality</h3>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={settings.jpgQuality}
                  onChange={(e) => handleChange('jpgQuality', parseInt(e.target.value))}
                  className="flex-1"
                />
                <span className="font-mono text-lg">{settings.jpgQuality}</span>
              </div>
            </div>
            
            <div>
              <h3 className="font-medium text-gray-900 mb-4">Default Tile Size</h3>
              <input
                type="number"
                value={settings.defaultTileSize}
                onChange={(e) => handleChange('defaultTileSize', parseInt(e.target.value))}
                min="64"
                max="2048"
                step="64"
                className="w-full md:w-1/2 p-2 border rounded"
              />
              <p className="text-sm text-gray-500 mt-1">Larger tiles use more VRAM but may be faster</p>
            </div>
          </div>
        )}
        
        {activeTab === 'advanced' && (
          <div className="space-y-6 max-w-2xl">
            <div>
              <h3 className="font-medium text-gray-900 mb-4">Sharpening Settings</h3>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium mb-1">Overlap</label>
                  <input
                    type="number"
                    value={settings.sharpenOverlap}
                    onChange={(e) => handleChange('sharpenOverlap', parseInt(e.target.value))}
                    min="32"
                    max="256"
                    step="32"
                    className="w-full p-2 border rounded"
                  />
                  <p className="text-sm text-gray-500 mt-1">Window overlap for tile processing</p>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Unsharp Amount</label>
                  <input
                    type="number"
                    value={settings.sharpenUnsharpAmount}
                    onChange={(e) => handleChange('sharpenUnsharpAmount', parseFloat(e.target.value))}
                    min="0"
                    max="2"
                    step="0.1"
                    className="w-full p-2 border rounded"
                  />
                  <p className="text-sm text-gray-500 mt-1">Unsharp mask strength</p>
                </div>
              </div>
            </div>
            
            <div className="border-t pt-6">
              <h3 className="font-medium text-gray-900 mb-4">Danger Zone</h3>
              <button
                onClick={() => { 
                  if (confirm('Reset all settings to defaults?')) {
                    setSettings(defaultSettings);
                  }
                }}
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
              >
                Reset to Defaults
              </button>
            </div>
          </div>
        )}
      </div>
      
      {/* Save Button & Message */}
      <div className="border-t bg-gray-50 p-4 flex justify-between items-center">
        {message && (
          <div className={`px-4 py-2 rounded text-sm ${message.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
            {message.text}
          </div>
        )}
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
};