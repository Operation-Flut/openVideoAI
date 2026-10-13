export interface Model {
  id: string;
  name: string;
  description: string;
  size: string;
  weight: string;
  features: string[];
  is_ai: boolean;
  is_builtin: boolean;
  requires_cuda: boolean;
  requires_metal: boolean;
  can_run_on_cpu: boolean;
  has_download_url: boolean;
  compatible: boolean;
  compatibility_reason: string | null;
  download_url?: string;
  icon?: string;
}

export interface ModelFeature {
  name: string;
  description: string;
}

export interface ModelsResponse {
  models: Model[];
}