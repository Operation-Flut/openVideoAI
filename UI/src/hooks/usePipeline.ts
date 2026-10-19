import { useCallback, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import { usePipelineStore, useProcessingStore } from '../store'
import { useRenderQueueStore, type RenderJob } from '../store/renderQueue'
import type { ExportOptions } from '../types/api'

export function usePipeline() {
  const {
    steps,
    availableFeatures,
    outputFormat,
    quality,
    selectedFiles,
    isExporting,
    addStep,
    removeStep,
    updateStep,
    reorderSteps,
    clearSteps,
    setAvailableFeatures,
    setOutputFormat,
    setQuality,
    setSelectedFiles,
    setIsExporting,
  } = usePipelineStore()

  // Populate the store with the backend feature/model registry. Without this,
  // availableFeatures stays empty and every pipeline section reports "No models".
  const { data: features } = useQuery({
    queryKey: ['processing', 'features'],
    queryFn: () => api.getProcessingFeatures(),
  })

  useEffect(() => {
    if (features) {
      setAvailableFeatures(features.features)
    }
  }, [features, setAvailableFeatures])

  const getAvailableModels = useCallback((feature: string): string[] => {
    return availableFeatures[feature]?.available_models || []
  }, [availableFeatures])

  const getDefaultModel = useCallback((feature: string): string | null => {
    return availableFeatures[feature]?.default_model || null
  }, [availableFeatures])

  const getParametersSchema = useCallback((feature: string): Record<string, any> => {
    return availableFeatures[feature]?.parameters_schema || {}
  }, [availableFeatures])

  const stepCount = steps.length

  const canAddFeature = useCallback((feature: string): boolean => {
    return !steps.some(s => s.feature === feature)
  }, [steps])

  // Convenience: queue an export of the currently selected file. The
  // RenderQueueRunner picks it up and runs it through the backend's single
  // render slot, so several exports can be lined up back to back.
  const exportCurrentFile = useCallback(async (exportOptions?: ExportOptions): Promise<RenderJob | null> => {
    const file = selectedFiles?.[0] || null
    if (!file || steps.length === 0) return null

    setIsExporting(true)
    try {
      const job = useRenderQueueStore.getState().enqueue({
        file,
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type,
        steps,
        outputFormat,
        quality,
        exportOptions,
      })
      return job
    } finally {
      setIsExporting(false)
    }
  }, [selectedFiles, steps, outputFormat, quality, setIsExporting])

  // Run the same pipeline on a short sample so the result can be judged before
  // committing to a full export.
  const previewCurrentFile = useCallback(async (durationSeconds = 2) => {
    const file = selectedFiles?.[0] || null
    if (!file || steps.length === 0) return

    try {
      const task = await api.startPreviewWithFile(file, steps, outputFormat, quality, durationSeconds)
      useProcessingStore.getState().setCurrentTask(task)
    } catch (err) {
      console.error('Preview failed:', err)
    }
  }, [selectedFiles, steps, outputFormat, quality])

  return {
    steps,
    availableFeatures,
    outputFormat,
    quality,
    selectedFiles,
    isExporting,
    addStep,
    removeStep,
    updateStep,
    reorderSteps,
    clearSteps,
    setAvailableFeatures,
    setOutputFormat,
    setQuality,
    setSelectedFiles,
    getAvailableModels,
    getDefaultModel,
    getParametersSchema,
    canAddFeature,
    stepCount,
    startProcessing: exportCurrentFile,
    previewCurrentFile,
  }
}