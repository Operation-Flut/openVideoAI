import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';
import { RootState, AppDispatch } from '../../store';
import { PipelineStepComponent } from './PipelineStepComponent';
import { addPipelineStep, clearPipeline } from '../../store/pipelineStore';

interface Model {
  id: string;
  name: string;
  description: string;
  size: string;
  weight: string;
  icon?: string;
}

interface FeatureModels {
  [key: string]: Model[];
}

const featureModels: FeatureModels = {
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

export const PipelineBuilder: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { steps, loading, error } = useSelector((state: RootState) => state.pipeline);
  
  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;
    
    const sourceIndex = result.source.index;
    const destinationIndex = result.destination.index;
    
    if (sourceIndex === destinationIndex) return;
    
    const reorderedSteps = Array.from(steps);
    const [movedStep] = reorderedSteps.splice(sourceIndex, 1);
    reorderedSteps.splice(destinationIndex, 0, movedStep);
    
    dispatch(clearPipeline());
    reorderedSteps.forEach(step => dispatch(addPipelineStep(step)));
  };
  
  const getDefaultConfig = (modelId: string): Record<string, any> => {
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
    return configs[modelId] || {};
  };
  
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Pipeline Builder</h2>
        <div className="text-gray-500">Loading models...</div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Pipeline Builder</h2>
        <div className="text-red-500">Error: {error}</div>
      </div>
    );
  }
  
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 h-full">
      <h2 className="text-xl font-bold mb-4">Pipeline Builder</h2>
      
      {/* Available Models Sidebar */}
      <div className="mb-6">
        <h3 className="font-semibold mb-2">Available Features</h3>
        <div className="space-y-2">
          {Object.keys(featureModels).map((feature) => (
            <details key={feature} className="border rounded p-2 bg-gray-50">
              <summary className="font-medium cursor-pointer flex items-center justify-between">
                <span>{feature}</span>
                <span className="text-sm text-gray-500">{featureModels[feature]?.length || 0} models</span>
              </summary>
              <div className="mt-2 grid grid-cols-2 gap-2">
                {featureModels[feature]?.map((model: Model) => (
                  <button
                    key={model.id}
                    onClick={() => dispatch(addPipelineStep({ ...model, config: getDefaultConfig(model.id) }))}
                    className="border rounded p-2 cursor-pointer hover:bg-white transition-colors text-left"
                  >
                    <div className="font-medium text-sm">{model.name}</div>
                    <div className="text-xs text-gray-600">{model.description}</div>
                    <div className="text-xs text-gray-400">
                      Size: {model.size} | Weight: {model.weight}
                    </div>
                  </button>
                ))}
              </div>
            </details>
          ))}
        </div>
      </div>
      
      {/* Pipeline Stages */}
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="pipeline-stages">
          {(provided) => (
            <div
              ref={provided.innerRef as any}
              {...provided.droppableProps}
              className="space-y-2 min-h-32 border-2 border-dashed border-gray-300 rounded p-4"
            >
              {steps.length === 0 && (
                <div className="text-center text-gray-400 py-8">
                  <p>Drag features here to build your pipeline</p>
                  <p className="text-sm mt-2">Click on a feature above to add it</p>
                </div>
              )}
              {steps.map((step, index) => (
                <Draggable key={step.id} draggableId={step.id} index={index}>
                  {(provided) => (
                    <PipelineStepComponent
                      step={step}
                      index={index}
                      innerRef={provided.innerRef}
                      draggableProps={provided.draggableProps}
                      dragHandleProps={provided.dragHandleProps}
                    />
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
      
      {/* Pipeline Actions */}
      {steps.length > 0 && (
        <div className="mt-4 flex justify-between">
          <button
            onClick={() => dispatch(clearPipeline())}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Clear Pipeline
          </button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
            Export Pipeline Config
          </button>
        </div>
      )}
    </div>
  );
};