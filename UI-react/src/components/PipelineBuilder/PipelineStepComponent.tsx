import React from 'react';
import { DraggableProvided } from 'react-beautiful-dnd';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '../../store';

interface PipelineStepComponentProps {
  step: {
    id: string;
    name: string;
    description: string;
    icon?: string;
    config: Record<string, any>;
  };
  index: number;
  innerRef: (element: HTMLDivElement | null) => void;
  draggableProps: DraggableProvided['draggableProps'];
  dragHandleProps: DraggableProvided['dragHandleProps'];
}

export const PipelineStepComponent: React.FC<PipelineStepComponentProps> = ({
  step,
  index,
  innerRef,
  draggableProps,
  dragHandleProps,
}) => {
  const dispatch = useDispatch<AppDispatch>();
  
  const getDefaultConfig = (): Record<string, any> => {
    const configs: Record<string, any> = {
      'Real-ESRGAN': { model_name: 'Real-ESRGAN', scale: 4 },
      'RealBasicVSR': { model_name: 'RealBasicVSR', scale: 4 },
      'Real-CUGAN': { model_name: 'Real-CUGAN', scale: 3 },
      'CodeFormer': { model_name: 'CodeFormer', fidelity_weight: 0.5 },
      'GFPGAN': { model_name: 'GFPGAN', fidelity_weight: 0.5 },
      'RestoreFormer': { model_name: 'RestoreFormer', fidelity_weight: 0.5 },
      'NAFNet': { model_name: 'NAFNet', strength: 1.0 },
      'MPRNet': { model_name: 'MPRNet', strength: 1.0 },
      'FFTformer': { model_name: 'FFTformer', strength: 1.0 },
      'SCUNet': { model_name: 'SCUNet', strength: 0.5 },
      'NAFNet-SIDD': { model_name: 'NAFNet-SIDD', strength: 0.5 },
      'RIFE': { model_name: 'RIFE', multiplier: 2 },
      'FILM': { model_name: 'FILM', multiplier: 2 },
      'RIFE-SloMo': { model_name: 'RIFE-SloMo', slowdown_factor: 4.0 },
      'Reinhard Transfer': { model_name: 'Reinhard Transfer', strength: 1.0, preserve_luminance: true },
      'WCT2': { model_name: 'WCT2', strength: 1.0 },
      'NLUT': { model_name: 'NLUT', strength: 1.0 },
      'Zero-DCE': { model_name: 'Zero-DCE', strength: 0.7 },
    };
    return configs[step.id] || {};
  };
  
  const renderStepSpecificConfig = () => {
    const config = step.config || getDefaultConfig();
    
    switch (step.id) {
      case 'Real-ESRGAN':
      case 'RealBasicVSR':
      case 'Real-CUGAN':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="Real-ESRGAN">Real-ESRGAN</option>
                <option value="RealBasicVSR">RealBasicVSR</option>
                <option value="Real-CUGAN">Real-CUGAN</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Scale</label>
              <select
                value={config.scale || 4}
                onChange={(e) => handleConfigChange('scale', parseInt(e.target.value))}
                className="w-full p-2 border rounded text-sm"
              >
                <option value={2}>2x</option>
                <option value={4}>4x</option>
                <option value={8}>8x</option>
              </select>
            </div>
          </div>
        );
      
      case 'CodeFormer':
      case 'GFPGAN':
      case 'RestoreFormer':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="CodeFormer">CodeFormer</option>
                <option value="GFPGAN">GFPGAN</option>
                <option value="RestoreFormer">RestoreFormer</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Fidelity Weight</label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={config.fidelity_weight || 0.5}
                onChange={(e) => handleConfigChange('fidelity_weight', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600">Current: {config.fidelity_weight || 0.5}</div>
            </div>
          </div>
        );
      
      case 'NAFNet':
      case 'MPRNet':
      case 'FFTformer':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="NAFNet">NAFNet</option>
                <option value="MPRNet">MPRNet</option>
                <option value="FFTformer">FFTformer</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Strength</label>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={config.strength || 1.0}
                onChange={(e) => handleConfigChange('strength', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600">Current: {config.strength || 1.0}</div>
            </div>
          </div>
        );
      
      case 'SCUNet':
      case 'NAFNet-SIDD':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="SCUNet">SCUNet</option>
                <option value="NAFNet-SIDD">NAFNet-SIDD</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Strength</label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={config.strength || 0.5}
                onChange={(e) => handleConfigChange('strength', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600">Current: {config.strength || 0.5}</div>
            </div>
          </div>
        );
      
      case 'RIFE':
      case 'FILM':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="RIFE">RIFE</option>
                <option value="FILM">FILM</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Multiplier</label>
              <select
                value={config.multiplier || 2}
                onChange={(e) => handleConfigChange('multiplier', parseInt(e.target.value))}
                className="w-full p-2 border rounded text-sm"
              >
                <option value={1}>1x (60fps → 60fps)</option>
                <option value={2}>2x (60fps → 120fps)</option>
                <option value={4}>4x (30fps → 120fps)</option>
              </select>
            </div>
          </div>
        );
      
      case 'RIFE-SloMo':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="RIFE-SloMo">RIFE-SloMo</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Slowdown Factor</label>
              <input
                type="range"
                min="1.0"
                max="10.0"
                step="0.5"
                value={config.slowdown_factor || 4.0}
                onChange={(e) => handleConfigChange('slowdown_factor', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600">Current: {config.slowdown_factor || 4.0}x</div>
            </div>
          </div>
        );
      
      case 'Reinhard Transfer':
      case 'WCT2':
      case 'NLUT':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="Reinhard Transfer">Reinhard Transfer</option>
                <option value="WCT2">WCT2</option>
                <option value="NLUT">NLUT</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Strength</label>
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.1"
                value={config.strength || 1.0}
                onChange={(e) => handleConfigChange('strength', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600">Current: {config.strength || 1.0}</div>
            </div>
            {step.id === 'Reinhard Transfer' && (
              <div>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.preserve_luminance !== false}
                    onChange={(e) => handleConfigChange('preserve_luminance', e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Preserve Luminance</span>
                </label>
              </div>
            )}
          </div>
        );
      
      case 'Zero-DCE':
        return (
          <div className="space-y-2 mt-2">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model_name || step.id}
                onChange={(e) => handleConfigChange('model_name', e.target.value)}
                className="w-full p-2 border rounded text-sm"
              >
                <option value="Zero-DCE">Zero-DCE</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Strength</label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={config.strength || 0.7}
                onChange={(e) => handleConfigChange('strength', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600">Current: {config.strength || 0.7}</div>
            </div>
          </div>
        );
      
      default:
        return (
          <div className="text-sm text-gray-500 mt-2">
            No configuration available for {step.name}
          </div>
        );
    }
  };
  
  const handleConfigChange = (key: string, value: any) => {
    dispatch({
      type: 'pipeline/updatePipelineStepConfig',
      payload: { index, config: { ...step.config, [key]: value } }
    });
  };
  
  return (
    <div
      ref={innerRef}
      {...draggableProps}
      className="bg-blue-50 border border-blue-200 rounded p-3 flex items-start justify-between"
    >
      <div {...dragHandleProps} className="flex items-center mr-3 cursor-grab">
        <span className="text-xl mr-3">{step.icon || '📦'}</span>
        <div className="flex flex-col">
          <div className="font-medium">{step.name}</div>
          <div className="text-xs text-gray-500">Step {index + 1}</div>
        </div>
      </div>
      
      <div className="flex-1 min-w-0">
        {renderStepSpecificConfig()}
      </div>
      
      <button
        onClick={() => dispatch({ type: 'pipeline/removePipelineStep', payload: index })}
        className="text-red-500 hover:text-red-700 font-bold ml-2"
      >
        ×
      </button>
    </div>
  );
};