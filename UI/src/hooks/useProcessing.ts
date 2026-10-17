import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import { api } from '../services/api'
import { useSSE, type ProcessingProgressEvent } from '../services/sse'
import { useProcessingStore, usePipelineStore } from '../store'
import type { ProcessRequest, OutputFormat } from '../types/api'

export function useProcessingFeatures() {
  return useQuery({
    queryKey: ['processing', 'features'],
    queryFn: () => api.getProcessingFeatures(),
  })
}

export function useStartProcessing() {
  const queryClient = useQueryClient()
  const { setCurrentTask, setProcessing } = useProcessingStore()

  return useMutation({
    mutationFn: (request: ProcessRequest) => api.startProcessing(request),
    onMutate: () => {
      setProcessing(true)
    },
    onSuccess: (task) => {
      setCurrentTask(task)
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
    onError: (error) => {
      setProcessing(false)
      setCurrentTask(null)
      console.error('Processing failed to start:', error)
    },
  })
}

export function useTaskStatus(taskId: string | null) {
  const { updateTaskProgress, applyTaskStatus, addToHistory } = useProcessingStore()
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ['task', taskId],
    queryFn: () => api.getTaskStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      if (query.state.data?.status === 'processing' || query.state.data?.status === 'pending') {
        return 1000
      }
      return false
    },
  })

  useSSE(
    taskId ? `${api['client'].defaults.baseURL}/processing/task/${taskId}/stream` : null,
    (data: ProcessingProgressEvent) => {
      updateTaskProgress(data.task_id, data.progress, data.message)
      if (data.type === 'processing.completed' || data.type === 'processing.failed' || data.type === 'processing.aborted') {
        queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      }
    }
  )

  useEffect(() => {
    if (query.data) {
      // Full payload (includes output_path once the task completes); the SSE
      // handler above only carries progress.
      applyTaskStatus(query.data)
      if (['completed', 'failed', 'aborted'].includes(query.data.status)) {
        addToHistory(query.data)
      }
    }
  }, [query.data, applyTaskStatus, addToHistory])

  return query
}

export function useCancelTask() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (taskId: string) => api.cancelTask(taskId),
    onSuccess: (_data, taskId) => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })
}

export function useTasks() {
  return useQuery({
    queryKey: ['tasks'],
    queryFn: () => api.listTasks(),
    refetchInterval: 5000,
  })
}

export function useUpscaleFile() {
  return useMutation({
    mutationFn: (params: {
      file: File
      model: string
      scale: number
      device?: string
      tile_size?: number
      fp16?: boolean
      output_format?: OutputFormat
    }) => api.upscaleFile(params.file, params),
  })
}

export function useFaceRecoveryFile() {
  return useMutation({
    mutationFn: (params: {
      file: File
      model: string
      fidelity_weight: number
      device?: string
    }) => api.faceRecoveryFile(params.file, params),
  })
}

export function useSharpenFile() {
  return useMutation({
    mutationFn: (params: {
      file: File
      model: string
      strength: number
      overlap?: number
      unsharp_amount?: number
      device?: string
    }) => api.sharpenFile(params.file, params),
  })
}

export function useDenoiseFile() {
  return useMutation({
    mutationFn: (params: {
      file: File
      model: string
      strength: number
      device?: string
    }) => api.denoiseFile(params.file, params),
  })
}

export function useFrameInterpolationFile() {
  return useMutation({
    mutationFn: (params: {
      file: File
      model: string
      multiplier: number
      slowmo_factor?: number
      device?: string
    }) => api.frameInterpolationFile(params.file, params),
  })
}

export function useMatchColorFile() {
  return useMutation({
    mutationFn: (params: {
      file: File
      reference: File
      model: string
      strength: number
      preserve_luminance: boolean
      device?: string
    }) => api.matchColorFile(params.file, params.reference, params),
  })
}

export function usePipeline() {
  const {
    steps,
    availableFeatures,
    outputFormat,
    quality,
    addStep,
    removeStep,
    updateStep,
    reorderSteps,
    clearSteps,
    setAvailableFeatures,
    setOutputFormat,
    setQuality,
  } = usePipelineStore()

  const { data: features } = useProcessingFeatures()

  useEffect(() => {
    if (features) {
      setAvailableFeatures(features.features)
    }
  }, [features, setAvailableFeatures])

  const getAvailableModels = (feature: string): string[] => {
    return availableFeatures[feature]?.available_models || []
  }

  const getDefaultModel = (feature: string): string | null => {
    return availableFeatures[feature]?.default_model || null
  }

  const getParametersSchema = (feature: string): Record<string, any> => {
    return availableFeatures[feature]?.parameters_schema || {}
  }

  const canAddFeature = (feature: string): boolean => {
    return !steps.some(s => s.feature === feature)
  }

  const buildRequest = (inputPath: string, outputPath?: string): ProcessRequest => {
    return {
      input_path: inputPath,
      output_path: outputPath,
      steps: steps.map(s => ({
        feature: s.feature,
        model: s.model,
        parameters: s.parameters,
      })),
      output_format: outputFormat,
      quality,
    }
  }

  return {
    steps,
    availableFeatures,
    outputFormat,
    quality,
    addStep,
    removeStep,
    updateStep,
    reorderSteps,
    clearSteps,
    setOutputFormat,
    setQuality,
    getAvailableModels,
    getDefaultModel,
    getParametersSchema,
    canAddFeature,
    buildRequest,
    stepCount: steps.length,
  }
}