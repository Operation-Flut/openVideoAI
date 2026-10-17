import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ModelResponse, ProcessingTask, PipelineStep, OutputFormat, ProcessingFeaturesResponse } from '../types/api'

// Model store
interface ModelState {
  models: ModelResponse[]
  downloadedModels: ModelResponse[]
  features: ProcessingFeaturesResponse['features']
  isLoading: boolean
  error: string | null
  setModels: (models: ModelResponse[]) => void
  setDownloadedModels: (models: ModelResponse[]) => void
  setFeatures: (features: ProcessingFeaturesResponse['features']) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  updateModelDownloadStatus: (name: string, downloaded: boolean, path?: string) => void
}

export const useModelStore = create<ModelState>()(
  persist(
    (set) => ({
      models: [],
      downloadedModels: [],
      features: {},
      isLoading: false,
      error: null,
      setModels: (models) => set({ models, downloadedModels: models.filter(m => m.downloaded) }),
      setDownloadedModels: (downloadedModels) => set({ downloadedModels }),
      setFeatures: (features) => set({ features }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
      updateModelDownloadStatus: (name, downloaded, path) =>
        set((state) => ({
          models: state.models.map((m) =>
            m.name === name ? { ...m, downloaded, local_path: path || m.local_path } : m
          ),
          downloadedModels: downloaded
            ? [...state.downloadedModels.filter(m => m.name !== name), state.models.find(m => m.name === name)!].filter(Boolean)
            : state.downloadedModels.filter(m => m.name !== name),
        })),
    }),
    {
      name: 'openvideoai-models',
      partialize: (state) => ({
        // Don't persist loading/error states
        models: state.models,
        downloadedModels: state.downloadedModels,
        features: state.features,
      }),
    }
  )
)

// Processing store
interface ProcessingState {
  currentTask: ProcessingTask | null
  taskHistory: ProcessingTask[]
  isProcessing: boolean
  progress: number
  message: string
  setCurrentTask: (task: ProcessingTask | null) => void
  updateTaskProgress: (taskId: string, progress: number, message: string, status?: ProcessingTask['status']) => void
  applyTaskStatus: (task: ProcessingTask) => void
  addToHistory: (task: ProcessingTask) => void
  clearHistory: () => void
  setProcessing: (isProcessing: boolean) => void
}

export const useProcessingStore = create<ProcessingState>()(
  persist(
    (set) => ({
      currentTask: null,
      taskHistory: [],
      isProcessing: false,
      progress: 0,
      message: '',
      setCurrentTask: (currentTask) => set({ currentTask, isProcessing: !!currentTask }),
      updateTaskProgress: (taskId, progress, message, status) =>
        set((state) => ({
          currentTask: state.currentTask?.task_id === taskId
            ? { ...state.currentTask, progress, message, status: status || state.currentTask.status }
            : state.currentTask,
          taskHistory: state.taskHistory.map((t) =>
            t.task_id === taskId ? { ...t, progress, message, status: status || t.status } : t
          ),
        })),
      // Merge a full task payload (status poll) so finished runs carry their
      // output_path/result, which the canvas needs to show the enhanced media.
      applyTaskStatus: (task) =>
        set((state) => ({
          currentTask: state.currentTask?.task_id === task.task_id
            ? { ...state.currentTask, ...task }
            : state.currentTask,
          taskHistory: state.taskHistory.map((t) => (t.task_id === task.task_id ? { ...t, ...task } : t)),
        })),
      addToHistory: (task) =>
        set((state) => ({
          taskHistory: [task, ...state.taskHistory.slice(0, 49)], // Keep last 50
        })),
      clearHistory: () => set({ taskHistory: [] }),
      setProcessing: (isProcessing) => set({ isProcessing }),
    }),
    {
      name: 'openvideoai-processing',
      partialize: (state) => ({
        taskHistory: state.taskHistory,
      }),
    }
  )
)

// Pipeline store - for building enhancement pipelines
interface PipelineState {
  steps: PipelineStep[]
  availableFeatures: ProcessingFeaturesResponse['features']
  selectedFiles: File[] | null
  referenceFile: File | null
  outputFormat: OutputFormat
  quality: string
  isExporting: boolean
  addStep: (step: PipelineStep) => void
  removeStep: (feature: string) => void
  updateStep: (feature: string, updates: Partial<PipelineStep>) => void
  reorderSteps: (fromIndex: number, toIndex: number) => void
  clearSteps: () => void
  setAvailableFeatures: (features: ProcessingFeaturesResponse['features']) => void
  setSelectedFiles: (files: File[] | null) => void
  setReferenceFile: (file: File | null) => void
  setOutputFormat: (format: OutputFormat) => void
  setQuality: (quality: string) => void
  setIsExporting: (isExporting: boolean) => void
  reset: () => void
}

const defaultPipelineState = {
  steps: [],
  availableFeatures: {} as ProcessingFeaturesResponse['features'],
  selectedFiles: null as File[] | null,
  referenceFile: null,
  outputFormat: 'source' as OutputFormat,
  quality: 'medium',
}

export const usePipelineStore = create<PipelineState>()(
  persist(
    (set) => ({
      ...defaultPipelineState,
      isExporting: false,
      addStep: (step) =>
        set((state) => ({
          steps: [...state.steps.filter(s => s.feature !== step.feature), step],
        })),
      removeStep: (feature) =>
        set((state) => ({
          steps: state.steps.filter(s => s.feature !== feature),
        })),
      updateStep: (feature, updates) =>
        set((state) => ({
          steps: state.steps.map((s) =>
            s.feature === feature ? { ...s, ...updates, parameters: { ...s.parameters, ...updates.parameters } } : s
          ),
        })),
      reorderSteps: (fromIndex, toIndex) =>
        set((state) => {
          const newSteps = [...state.steps]
          const [removed] = newSteps.splice(fromIndex, 1)
          newSteps.splice(toIndex, 0, removed)
          return { steps: newSteps }
        }),
      clearSteps: () => set({ steps: [] }),
      setAvailableFeatures: (availableFeatures) =>
        set((state) => {
          // Persisted steps can outlive a catalog change (e.g. a model whose
          // backend is gated out). Drop steps for unknown features and
          // re-point models that are no longer offered at the feature default,
          // so a stale selection can never silently no-op at export time.
          if (Object.keys(availableFeatures).length === 0) {
            return { availableFeatures }
          }
          const steps = state.steps.flatMap((step) => {
            const info = availableFeatures[step.feature]
            if (!info) return []
            if (step.model && info.available_models?.includes(step.model)) return [step]
            const fallback = info.default_model || info.available_models?.[0]
            return fallback ? [{ ...step, model: fallback }] : []
          })
          return { availableFeatures, steps }
        }),
      setSelectedFiles: (selectedFiles) => set({ selectedFiles }),
      setReferenceFile: (referenceFile) => set({ referenceFile }),
      setOutputFormat: (outputFormat) => set({ outputFormat }),
      setQuality: (quality) => set({ quality }),
      setIsExporting: (isExporting: boolean) => set({ isExporting }),
      reset: () => set({ ...defaultPipelineState, isExporting: false }),
    }),
    {
      name: 'openvideoai-pipeline',
      partialize: (state) => ({
        steps: state.steps,
        outputFormat: state.outputFormat,
        quality: state.quality,
      }),
    }
  )
)

// UI store
interface UIState {
  theme: 'light' | 'dark' | 'system'
  sidebarOpen: boolean
  density: 'compact' | 'comfortable' | 'spacious'
  fontSize: 'small' | 'medium' | 'large'
  accentColor: string
  notifications: Notification[]
  simpleMode: boolean
  setTheme: (theme: 'light' | 'dark' | 'system') => void
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setDensity: (density: 'compact' | 'comfortable' | 'spacious') => void
  setFontSize: (size: 'small' | 'medium' | 'large') => void
  setAccentColor: (color: string) => void
  setSimpleMode: (enabled: boolean) => void
  addNotification: (notification: Omit<Notification, 'id'>) => void
  removeNotification: (id: string) => void
}

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message?: string
  duration?: number
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: 'dark',
      sidebarOpen: true,
      density: 'comfortable',
      fontSize: 'medium',
      accentColor: '#3b82f6',
      notifications: [],
      simpleMode: true,
      setTheme: (theme) => set({ theme }),
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setDensity: (density) => set({ density }),
      setFontSize: (fontSize) => set({ fontSize }),
      setAccentColor: (accentColor) => set({ accentColor }),
      setSimpleMode: (enabled: boolean) => set({ simpleMode: enabled }),
      addNotification: (notification) =>
        set((state) => ({
          notifications: [...state.notifications, { ...notification, id: crypto.randomUUID() }],
        })),
      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),
    }),
    {
      name: 'openvideoai-ui',
    }
  )
)