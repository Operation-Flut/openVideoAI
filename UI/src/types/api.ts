// API Types matching backend schemas

export interface ModelFeature {
  name: string
  description: string
}

export interface ModelInfo {
  name: string
  description: string
  description_short: string
  size: string
  weight: string
  url: string
  download_url: string | null
  filename: string | null
  features: ModelFeature[]
  is_ai: boolean
  requires_cuda: boolean
  requires_metal: boolean
  can_run_on_cpu: boolean
  is_builtin: boolean
  module: string | null
}

export interface ModelResponse {
  name: string
  description: string
  size: string
  weight: string
  downloaded: boolean
  implemented: boolean
  features: string[]
  is_ai: boolean
  is_builtin: boolean
  requires_cuda: boolean
  requires_metal: boolean
  can_run_on_cpu: boolean
  has_download_url: boolean
  compatible: boolean
  compatibility_reason: string | null
  local_path: string | null
  // Extended detail fields (shown in the model details popup)
  full_description?: string | null
  filename?: string | null
  module?: string | null
  source_url?: string | null
  download_url?: string | null
  verified?: boolean
  error?: string | null
  additional_paths?: Record<string, string>
}

export interface DownloadProgress {
  model_name: string
  total_bytes: number
  downloaded_bytes: number
  speed_bps: number
  eta_seconds: number
  status: string
  error: string | null
  progress_percent: number
}

export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'aborted'

export interface ProcessingTask {
  task_id: string
  status: TaskStatus
  progress: number
  message: string
  current_step: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  error: string | null
  result: Record<string, any> | null
  output_path: string | null
}

export interface PipelineStep {
  feature: string
  model: string
  parameters: Record<string, any>
}

export interface ProcessRequest {
  input_path: string
  output_path?: string
  steps: PipelineStep[]
  output_format: OutputFormat
  quality: string
}

export type OutputFormat = 'source' | 'jpg' | 'png' | 'webp' | 'tiff' | 'mp4'

export interface FeatureInfo {
  name: string
  description: string
  implemented: boolean
  available_models: string[]
  default_model: string | null
  parameters_schema: Record<string, any>
}

/** Feature metadata declared in models.json (exposed by GET /models/features). */
export interface FeatureMetadata {
  name: string
  displayName: string
  description: string
}

/** Simple-Mode sidebar group declared in models.json (GET /models/groups). */
export interface FeatureGroupDefinition {
  key: string
  label: string
  description: string
  /** Icon component name resolved by the UI (presentation concern). */
  icon: string
  features: string[]
}

/** Shape returned by GET /models/features/implemented. */
export interface ImplementedFeature {
  key: string
  displayName: string
  description: string
  required: boolean
  hasToggle: boolean
  hasStrength: boolean
  requiresReferenceImage: boolean
  /** Set when the catalog marks the feature itself as experimental. */
  experimental?: boolean
  availableModels: string[]
  defaultModel: string | null
}

/** An experimental feature or model, as marked in models.json. */
export interface ExperimentalItem {
  kind: 'feature' | 'model'
  name: string
  displayName: string
  description: string
  /** Features a model belongs to; empty for feature entries. */
  features: string[]
}

/** Shape returned by GET /models/experimental. */
export interface ExperimentalResponse {
  /** Whether the user currently has experimental features enabled. */
  enabled: boolean
  /** Empty when nothing in the catalog is marked experimental. */
  items: ExperimentalItem[]
}

export interface ProcessingFeaturesResponse {
  features: Record<string, FeatureInfo>
}

export interface SystemStatus {
  total_models: number
  downloaded: number
  available: number
  models_dir: string
  gpu_info: GpuInfo | null
  cpu_threads: number
  ram_gb: number
  vram_gb: number | null
}

export interface GpuInfo {
  name: string
  vram_gb: number
  backend: string
  device_id: number
}

export interface SettingsModel {
  system: SystemResources
  gpu: GPUSettings
  export: ExportSettings
  models: ModelSettings
  appearance: AppearanceSettings
  advanced: AdvancedSettings
}

export interface SystemResources {
  max_ram_usage: number
  max_vram_usage: number
  cpu_threads: number
  tile_size: number
}

export interface GPUSettings {
  compute_backend: ComputeBackend
  gpu_device: string
  fp16_enabled: boolean
  tensorrt_enabled: boolean
}

export type ComputeBackend = 'auto' | 'cuda' | 'amd' | 'hip' | 'dml' | 'metal' | 'cpu'

export interface ExportSettings {
  default_upscale: number
  default_tile_size: number
  output_format: OutputFormat
  jpg_quality: number
  sharpen_overlap: number
  sharpen_unsharp_amount: number
  export_dir: string
  export_naming: string
  overwrite_existing: boolean
  video_codec: string
  video_crf: number
}

export interface ModelSettings {
  model_dir: string
  auto_download_models: boolean
  verify_models: boolean
  hide_incompatible_models: boolean
  prefer_ai: boolean
  advanced_mode: boolean
}

export interface AppearanceSettings {
  theme: 'light' | 'dark' | 'system'
  accent_color: string
  font_size: 'small' | 'medium' | 'large'
  density: 'compact' | 'comfortable' | 'spacious'
}

export interface AdvancedSettings {
  debug_logging: boolean
  show_metrics: boolean
  experimental_features: boolean
}

export interface ComputeBackendInfo {
  value: ComputeBackend
  label: string
  available: boolean
  recommended: boolean
  reason: string | null
}

export interface SettingsResponse {
  settings: SettingsModel
  max_values: Record<string, any>
  /** Machine-derived: the compute backends this install can actually use. */
  available_backends: ComputeBackendInfo[]
}

export interface HealthResponse {
  status: string
  version: string
  timestamp: string
  uptime_seconds: number
}