import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useModelStore } from '../store'
import { setFeatureDisplayNames } from '../utils/displayNames'
import { useSettings } from './useSettings'
import type { DownloadProgress } from '../types/api'

export function useModels() {
  const { setModels, setDownloadedModels, setLoading, setError } = useModelStore()

  // Always fetch the FULL model catalog. Filtering happens client-side in the
  // Models page so the global store keeps every model available to the
  // Enhance pipeline (which previously lost models when a filtered query
  // overwrote the store).
  const query = useQuery({
    queryKey: ['models'],
    queryFn: () => api.getModels(),
    staleTime: 1000 * 60 * 5,
  })

  useEffect(() => {
    if (query.data) {
      setModels(query.data)
      setDownloadedModels(query.data.filter(m => m.downloaded))
    }
    setLoading(query.isLoading)
    setError(query.error?.message || null)
  }, [query.data, query.isLoading, query.error, setModels, setDownloadedModels, setLoading, setError])

  return query
}

export function useModel(name: string) {
  return useQuery({
    queryKey: ['model', name],
    queryFn: () => api.getModel(name),
    enabled: !!name,
  })
}

export function useDownloadModel() {
  const queryClient = useQueryClient()
  const { updateModelDownloadStatus } = useModelStore()

  return useMutation({
    mutationFn: ({ name, force }: { name: string; force?: boolean }) => api.downloadModel(name, force),
    onSuccess: (data, variables) => {
      if (data.status === 'completed') {
        updateModelDownloadStatus(variables.name, true)
      }
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
    onError: (error) => {
      console.error('Download failed:', error)
    },
  })
}

export function useDeleteModel() {
  const queryClient = useQueryClient()
  const { updateModelDownloadStatus } = useModelStore()

  return useMutation({
    mutationFn: (name: string) => api.deleteModel(name),
    onSuccess: (_data, name) => {
      updateModelDownloadStatus(name, false)
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

export function useFeatures() {
  const query = useQuery({
    queryKey: ['features'],
    queryFn: () => api.getFeatures(),
  })

  // Feature labels come from the catalog; publish them so `displayFeature()`
  // resolves everywhere without a hardcoded map.
  useEffect(() => {
    if (query.data) {
      setFeatureDisplayNames(
        Object.fromEntries(query.data.map(f => [f.name, f.displayName])),
      )
    }
  }, [query.data])

  return query
}

export function useFeatureGroups() {
  return useQuery({
    queryKey: ['feature-groups'],
    queryFn: () => api.getFeatureGroups(),
    staleTime: 1000 * 60 * 30,
  })
}

export function useImplementedFeatures() {
  return useQuery({
    queryKey: ['features', 'implemented'],
    queryFn: () => api.getImplementedFeatures(),
  })
}

/** Catalog-declared experimental features/models (Settings -> Advanced info). */
export function useExperimental(enabled = true) {
  return useQuery({
    queryKey: ['features', 'experimental'],
    queryFn: () => api.getExperimental(),
    enabled,
  })
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ['system', 'status'],
    queryFn: () => api.getSystemStatus(),
    refetchInterval: 30000,
  })
}

/**
 * Smart Select recommendations (GET /models/recommendations).
 *
 * The rules live in the backend (catalog metadata plus detected hardware), so
 * this hook only supplies the media of the loaded file. The settings that change
 * the answer (the memory budget and the experimental toggle) are part of the
 * query key so editing them in Settings refreshes the ranking instead of serving
 * a stale one. `recommendations` is empty while the setting is off - a disabled
 * react-query keeps its last data, so the flag is applied here rather than left
 * to each caller.
 */
export function useModelRecommendations(media: 'image' | 'video' | null) {
  const { data } = useSettings()
  const settings = data?.settings
  const smartSelect = settings?.models.smart_select ?? false

  const query = useQuery({
    queryKey: [
      'models',
      'recommendations',
      media ?? 'both',
      settings?.system.max_ram_usage ?? null,
      settings?.system.max_vram_usage ?? null,
      settings?.advanced.experimental_features ?? null,
    ],
    queryFn: () => api.getModelRecommendations(media),
    enabled: smartSelect,
    staleTime: 1000 * 60 * 5,
  })

  return {
    smartSelect,
    recommendations: smartSelect ? query.data?.recommendations : undefined,
  }
}

export function useModelDownloadProgress(modelName: string | null) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null)

  useEffect(() => {
    if (!modelName) return

    const eventSource = new EventSource(`${api['client'].defaults.baseURL}/models/${modelName}/progress/stream`)

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setProgress(data)
        if (data.status === 'completed' || data.status === 'failed') {
          eventSource.close()
        }
      } catch (e) {
        console.warn('Failed to parse download progress:', event.data)
      }
    }

    eventSource.onerror = () => {
      eventSource.close()
    }

    return () => eventSource.close()
  }, [modelName])

  return progress
}