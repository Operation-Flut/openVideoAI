export interface PipelineStepConfig {
  model_name?: string;
  scale?: number;
  fidelity_weight?: number;
  strength?: number;
  multiplier?: number;
  steps?: number;
}

export interface PipelineStep {
  id: string;
  name: string;
  description: string;
  icon?: string;
  config: PipelineStepConfig;
}

export interface PipelineConfig {
  features: string[];
  [key: string]: any;
}

export interface PipelineRequest {
  file: File;
  pipeline_config: PipelineConfig;
  device?: string;
}

export interface PipelineResponse {
  task_id: string;
  status: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'processing' | 'completed' | 'failed' | 'aborted';
  progress: number;
  currentStage?: string;
  error?: string;
  result?: any;
}

export interface ProcessingFeature {
  name: string;
  models: string[];
}