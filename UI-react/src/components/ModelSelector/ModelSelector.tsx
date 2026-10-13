import React, { useState } from 'react';

interface ModelSelectorProps {
  feature: string;
  onModelSelect: (model: any) => void;
}

const featureModels: Record<string, any[]> = {
  'Upscale': [
    { id: 'Real-ESRGAN', name: 'Real-ESRGAN', description: 'General purpose upscaling', size: '65MB', weight: 'Lightweight', icon: '⬆️' },
    { id: 'RealBasicVSR', name: 'RealBasicVSR', description: 'Video upscaling with temporal consistency', size: '200MB', weight: 'Medium', icon: '🎬' },
    { id: 'Real-CUGAN', name: 'Real-CUGAN', description: 'Anime/multipurpose upscaling', size: '40MB', weight: 'Lightweight', icon: '🎨' },
  ],
  'Face Recovery': [
    { id: 'CodeFormer', name: 'CodeFormer', description: 'Advanced face restoration', size: '360MB', weight: 'Medium', icon: '👤' },
    { id: 'GFPGAN', name: 'GFPGAN', description: 'Face restoration for old photos', size: '333MB', weight: 'Medium', icon: '👴' },
    { id: 'RestoreFormer', name: 'RestoreFormer', description: 'High-quality face restoration', size: '443MB', weight: 'Heavy', icon: '✨' },
  ],
  'Sharpen / Deblur': [
    { id: 'NAFNet', name: 'NAFNet', description: 'General deblurring', size: '260MB', weight: 'Ultra-Lightweight', icon: '🔍' },
    { id: 'MPRNet', name: 'MPRNet', description: 'Multi-stage progressive restoration', size: '57MB', weight: 'Lightweight', icon: '🔧' },
    { id: 'FFTformer', name: 'FFTformer', description: 'Frequency domain transformer', size: '66MB', weight: 'Lightweight', icon: '📊' },
  ],
  'Denoise': [
    { id: 'SCUNet', name: 'SCUNet', description: 'Swin Transformer + UNet denoising', size: '45MB', weight: 'Lightweight', icon: '🧼' },
    { id: 'NAFNet-SIDD', name: 'NAFNet-SIDD', description: 'Sensor noise removal', size: '464MB', weight: 'Ultra-Lightweight', icon: '📷' },
  ],
  'Frame Interpolation': [
    { id: 'RIFE', name: 'RIFE', description: 'Real-time intermediate flow estimation', size: '20MB', weight: 'Ultra-Lightweight', icon: '🎞️' },
    { id: 'FILM', name: 'FILM', description: 'Frame interpolation for large motion', size: '130MB', weight: 'Medium', icon: '🎬' },
  ],
  'SloMo': [
    { id: 'RIFE-SloMo', name: 'RIFE-SloMo', description: 'Slow motion video generation', size: '20MB', weight: 'Ultra-Lightweight', icon: '🐢' },
  ],
  'Match Color': [
    { id: 'Reinhard Transfer', name: 'Reinhard Transfer', description: 'Classical color transfer', size: '< 1MB', weight: 'Ultra-Lightweight', icon: '🎨' },
    { id: 'WCT2', name: 'WCT2', description: 'Wavelet color transfer', size: '40MB', weight: 'Lightweight', icon: '🌊' },
    { id: 'NLUT', name: 'NLUT', description: 'Neural lookup table color transfer', size: '225MB', weight: 'Lightweight', icon: '🧠' },
  ],
  'Low Light': [
    { id: 'Zero-DCE', name: 'Zero-DCE', description: 'Zero-reference low-light enhancement', size: '5MB', weight: 'Lightweight', icon: '🌙' },
  ],
};

export const ModelSelector: React.FC<ModelSelectorProps> = ({ feature, onModelSelect }) => {
  const [search, setSearch] = useState('');
  const models = featureModels[feature] || [];
  const filteredModels = models.filter(model =>
    model.name.toLowerCase().includes(search.toLowerCase()) ||
    model.description.toLowerCase().includes(search.toLowerCase())
  );
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">{feature}</h3>
        <input
          type="text"
          placeholder="Search models..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1 border rounded text-sm w-48"
        />
      </div>
      
      {filteredModels.length === 0 ? (
        <div className="text-center text-gray-500 py-8">
          {search ? 'No models match your search' : 'No models available for this feature'}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {filteredModels.map((model) => (
            <button
              key={model.id}
              onClick={() => onModelSelect(model)}
              className="p-3 text-left border rounded hover:bg-gray-50 transition-colors group"
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl">{model.icon || '📦'}</span>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm group-hover:text-blue-600 transition-colors">
                    {model.name}
                  </div>
                  <div className="text-xs text-gray-600 truncate">{model.description}</div>
                  <div className="text-xs text-gray-400 mt-1">
                    Size: {model.size} | Weight: {model.weight}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

interface ModelSelectorProps {
  feature: string;
  onModelSelect: (model: any) => void;
}