import { useCallback, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../services/api'
import { usePipelineStore } from '../store'
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

  // Convenience: start processing with the currently selected file
  const exportCurrentFile = useCallback(async (exportOptions?: ExportOptions) => {
    const file = selectedFiles?.[0] || null
    if (!file || steps.length === 0) return

    setIsExporting(true)
    try {
      const task = await api.startProcessingWithFile(
        file,
        steps,
        outputFormat,
        quality,
        exportOptions
      )
      // Update processing store with new task
      const { useProcessingStore } = await import('../store')
      useProcessingStore.getState().setCurrentTask(task)
    } catch (err) {
      console.error('Export failed:', err)
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
      const { useProcessingStore } = await import('../store')
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