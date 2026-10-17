import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../services/api'
import type { SettingsModel, SettingsResponse } from '../types/api'

export function useSettings() {
  return useQuery<SettingsResponse>({
    queryKey: ['settings'],
    queryFn: () => api.getSettings(),
    staleTime: 1000 * 60 * 10,
  })
}

export function useUpdateSettings() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (updates: Partial<SettingsModel>) => api.updateSettings(updates),
    onSuccess: (settings) => {
      queryClient.setQueryData<SettingsResponse>(['settings'], (current) =>
        current ? { ...current, settings } : current,
      )
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      // Settings such as "Experimental Features" change which features/models
      // the backend advertises, so refresh the catalog-derived queries too.
      queryClient.invalidateQueries({ queryKey: ['processing', 'features'] })
      queryClient.invalidateQueries({ queryKey: ['features'] })
      queryClient.invalidateQueries({ queryKey: ['feature-groups'] })
    },
  })
}

export function useResetSettings() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => api.resetSettings(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      queryClient.invalidateQueries({ queryKey: ['processing', 'features'] })
      queryClient.invalidateQueries({ queryKey: ['features'] })
    },
  })
}

export function useExportSettings() {
  return useMutation({
    mutationFn: () => api.exportSettings(),
  })
}

export function useImportSettings() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (file: File) => api.importSettings(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      queryClient.invalidateQueries({ queryKey: ['processing', 'features'] })
      queryClient.invalidateQueries({ queryKey: ['features'] })
    },
  })
}