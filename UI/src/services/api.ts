import axios, { AxiosInstance, AxiosError } from 'axios'
import type {
  ModelResponse,
  DownloadProgress,
  ProcessingTask,
  ProcessRequest,
  PipelineStep,
  FeatureMetadata,
  FeatureGroupDefinition,
  ImplementedFeature,
  ExperimentalResponse,
  ModelRecommendationsResponse,
  SystemStatus,
  SettingsModel,
  SettingsResponse,
  HealthResponse,
  OutputFormat,
  ProcessingFeaturesResponse,
  ProcessResponse,
  ExportOptionsResponse,
  BrowseResponse,
  ExportOptions,
  UpdateCheckResponse,
  PreReleaseCheckResponse,
  UpdateInstallResponse,
} from '../types/api'

const API_BASE = '/api'

class ApiService {
  public client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    })

    this.client.interceptors.request.use(
      (config) => config,
      (error) => Promise.reject(error)
    )

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const message = (error.response?.data as any)?.detail || error.message
        return Promise.reject(new Error(message))
      }
    )
  }

  async getModels(params?: {
    feature?: string
    downloaded_only?: boolean
    compatible_only?: boolean
    is_ai?: boolean
    page?: number
    page_size?: number
  }): Promise<ModelResponse[]> {
    // The catalog is small, but request a generous page so the client always
    // receives the complete list and can filter locally.
    const response = await this.client.get<ModelResponse[]>('/models', {
      params: { page_size: 500, ...params },
    })
    return response.data
  }

  async getModel(name: string): Promise<ModelResponse> {
    const response = await this.client.get<ModelResponse>(`/models/${name}`)
    return response.data
  }

  async downloadModel(name: string, force = false): Promise<DownloadProgress> {
    const response = await this.client.post<DownloadProgress>('/models/download', {
      model_name: name,
      force,
    })
    return response.data
  }

  async cancelDownload(name: string): Promise<void> {
    await this.client.post(`/models/${name}/cancel`)
  }

  async deleteModel(name: string): Promise<void> {
    await this.client.delete(`/models/${name}`)
  }

  async getFeatures(): Promise<FeatureMetadata[]> {
    const response = await this.client.get('/models/features')
    return response.data
  }

  async getFeatureGroups(): Promise<FeatureGroupDefinition[]> {
    const response = await this.client.get('/models/groups')
    return response.data
  }

  async getImplementedFeatures(): Promise<Record<string, ImplementedFeature>> {
    const response = await this.client.get('/models/features/implemented')
    return response.data
  }

  /** Features/models the catalog marks as experimental (GET /models/experimental). */
  async getExperimental(): Promise<ExperimentalResponse> {
    const response = await this.client.get<ExperimentalResponse>('/models/experimental')
    return response.data
  }

  /**
   * Which model Smart Select would use for each feature (GET /models/recommendations).
   * Passing the media of the loaded file lets the backend rank video models for
   * video and image models for stills.
   */
  async getModelRecommendations(
    media?: 'image' | 'video' | null,
  ): Promise<ModelRecommendationsResponse> {
    const response = await this.client.get<ModelRecommendationsResponse>('/models/recommendations', {
      params: media ? { media } : undefined,
    })
    return response.data
  }

  async getSystemStatus(): Promise<SystemStatus> {
    const response = await this.client.get<SystemStatus>('/models/status')
    return response.data
  }

  async getProcessingFeatures(): Promise<ProcessingFeaturesResponse> {
    const response = await this.client.get<ProcessingFeaturesResponse>('/processing/features')
    return response.data
  }

  async startProcessing(request: ProcessRequest): Promise<ProcessingTask> {
    const response = await this.client.post<ProcessingTask>('/processing', request)
    return response.data
  }

  async startProcessingWithFile(
    file: File,
    steps: PipelineStep[],
    outputFormat: OutputFormat,
    quality: string,
    exportOptions?: ExportOptions,
    onUploadProgress?: (percent: number) => void
  ): Promise<ProcessResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('steps', JSON.stringify(steps))
    formData.append('output_format', outputFormat)
    formData.append('quality', quality)
    if (exportOptions) {
      formData.append('export_options', JSON.stringify(exportOptions))
    }

    const response = await this.client.post<ProcessResponse>('/processing/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // Large sources (multi-GB video) can take far longer than the default
      // 30 s request timeout; the upload is local, so allow up to 30 minutes.
      timeout: 30 * 60 * 1000,
      // Uploading a large source can take a while; report it so the queue can
      // show progress before the render itself starts.
      onUploadProgress: (event) => {
        if (onUploadProgress && event.total) {
          onUploadProgress(Math.min(100, Math.max(0, (event.loaded / event.total) * 100)))
        }
      },
    })
    return response.data
  }

  async getTaskStatus(taskId: string): Promise<ProcessingTask> {
    const response = await this.client.get<ProcessingTask>(`/processing/task/${taskId}`)
    return response.data
  }

  /**
   * Run the pipeline on a short sample of the file (POST /processing/preview/file).
   * Videos are trimmed to `durationSeconds` and images downscaled by the backend.
   */
  async startPreviewWithFile(
    file: File,
    steps: PipelineStep[],
    outputFormat: OutputFormat,
    quality: string,
    durationSeconds = 2
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('steps', JSON.stringify(steps))
    formData.append('output_format', outputFormat)
    formData.append('quality', quality)
    formData.append('duration', String(durationSeconds))

    const response = await this.client.post<ProcessingTask>('/processing/preview/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  /** URL a viewer can load a completed task's result from. */
  taskResultUrl(taskId: string): string {
    return `${API_BASE}/processing/task/${taskId}/result`
  }

  async getTaskResult(taskId: string): Promise<Blob> {
    const response = await this.client.get(`/processing/task/${taskId}/result`, {
      responseType: 'blob',
    })
    return response.data
  }

  async cancelTask(taskId: string): Promise<void> {
    await this.client.post(`/processing/task/${taskId}/cancel`)
  }

  async listTasks(): Promise<ProcessingTask[]> {
    const response = await this.client.get<ProcessingTask[]>('/processing/tasks')
    return response.data
  }

  /**
   * The schema the Export dialog renders (GET /processing/export/options).
   */
  async getExportOptions(
    inputName?: string,
    scale?: number
  ): Promise<ExportOptionsResponse> {
    const params = new URLSearchParams()
    if (inputName) params.set('input_name', inputName)
    if (scale !== undefined) params.set('scale', String(scale))
    const response = await this.client.get<ExportOptionsResponse>(`/processing/export/options?${params}`)
    return response.data
  }

  /**
   * Ask the machine for a native file/folder picker (POST /processing/system/browse).
   */
  async systemBrowse(
    kind: 'file' | 'folder',
    suffixes?: string[],
    startDir?: string
  ): Promise<BrowseResponse> {
    const response = await this.client.post<BrowseResponse>('/processing/system/browse', {
      kind,
      suffixes,
      start_dir: startDir,
    })
    return response.data
  }

  async upscaleFile(
    file: File,
    options: {
      model: string
      scale: number
      device?: string
      tile_size?: number
      fp16?: boolean
      output_format?: OutputFormat
    }
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', options.model)
    formData.append('scale', options.scale.toString())
    if (options.device) formData.append('device', options.device)
    if (options.tile_size) formData.append('tile_size', options.tile_size.toString())
    if (options.fp16 !== undefined) formData.append('fp16', options.fp16.toString())
    if (options.output_format) formData.append('output_format', options.output_format)

    const response = await this.client.post<ProcessingTask>('/processing/upscale/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async faceRecoveryFile(
    file: File,
    options: {
      model: string
      fidelity_weight: number
      device?: string
    }
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', options.model)
    formData.append('fidelity_weight', options.fidelity_weight.toString())
    if (options.device) formData.append('device', options.device)

    const response = await this.client.post<ProcessingTask>('/processing/face-recovery/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async sharpenFile(
    file: File,
    options: {
      model: string
      strength: number
      overlap?: number
      unsharp_amount?: number
      device?: string
    }
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', options.model)
    formData.append('strength', options.strength.toString())
    if (options.overlap) formData.append('overlap', options.overlap.toString())
    if (options.unsharp_amount) formData.append('unsharp_amount', options.unsharp_amount.toString())
    if (options.device) formData.append('device', options.device)

    const response = await this.client.post<ProcessingTask>('/processing/sharpen/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async denoiseFile(
    file: File,
    options: {
      model: string
      strength: number
      device?: string
    }
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', options.model)
    formData.append('strength', options.strength.toString())
    if (options.device) formData.append('device', options.device)

    const response = await this.client.post<ProcessingTask>('/processing/denoise/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async frameInterpolationFile(
    file: File,
    options: {
      model: string
      multiplier: number
      slowmo_factor?: number
      device?: string
    }
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', options.model)
    formData.append('multiplier', options.multiplier.toString())
    if (options.slowmo_factor) formData.append('slowmo_factor', options.slowmo_factor.toString())
    if (options.device) formData.append('device', options.device)

    const response = await this.client.post<ProcessingTask>('/processing/frame-interpolation/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async matchColorFile(
    file: File,
    reference: File,
    options: {
      model: string
      strength: number
      preserve_luminance: boolean
      device?: string
    }
  ): Promise<ProcessingTask> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('reference', reference)
    formData.append('model', options.model)
    formData.append('strength', options.strength.toString())
    formData.append('preserve_luminance', options.preserve_luminance.toString())
    if (options.device) formData.append('device', options.device)

    const response = await this.client.post<ProcessingTask>('/processing/match-color/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  }

  async getSettings(): Promise<SettingsResponse> {
    const response = await this.client.get<SettingsResponse>('/settings')
    return response.data
  }

  async updateSettings(updates: Partial<SettingsModel>): Promise<SettingsModel> {
    const response = await this.client.patch<SettingsModel>('/settings', updates)
    return response.data
  }

  async resetSettings(): Promise<SettingsModel> {
    const response = await this.client.post<SettingsModel>('/settings/reset')
    return response.data
  }

  async exportSettings(): Promise<Blob> {
    const response = await this.client.get('/settings/export/file', {
      responseType: 'blob',
    })
    return response.data
  }

  async importSettings(file: File): Promise<void> {
    const formData = new FormData()
    formData.append('file', file)
    await this.client.post('/settings/import/file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  }

  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health')
    return response.data
  }

  async getSystemInfo(): Promise<SystemStatus> {
    const response = await this.client.get<SystemStatus>('/health/system')
    return response.data
  }

  async getVersion(): Promise<{ version: string; name: string; description: string }> {
    const response = await this.client.get('/health/version')
    return response.data
  }

  /**
   * Check for a new stable release (GET /update/check).
   */
  async checkForUpdate(): Promise<UpdateCheckResponse> {
    const response = await this.client.get<UpdateCheckResponse>('/update/check')
    return response.data
  }

  /**
   * Check for a new pre-release (GET /update/pre-release).
   */
  async checkPreRelease(): Promise<PreReleaseCheckResponse> {
    const response = await this.client.get<PreReleaseCheckResponse>('/update/pre-release')
    return response.data
  }

  /**
   * Download a release asset (POST /update/install).
   */
  async installUpdate(releaseTag: string): Promise<UpdateInstallResponse> {
    const response = await this.client.post<UpdateInstallResponse>('/update/install', {
      release_tag: releaseTag,
    })
    return response.data
  }
}

export const api = new ApiService()