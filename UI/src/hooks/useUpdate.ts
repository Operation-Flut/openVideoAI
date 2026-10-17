import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../services/api'
import type { UpdateCheckResponse, PreReleaseCheckResponse, UpdateInstallResponse } from '../types/api'

export function useUpdateCheck() {
  return useQuery<UpdateCheckResponse>({
    queryKey: ['update', 'check'],
    queryFn: () => api.checkForUpdate(),
    staleTime: 1000 * 60 * 30, // 30 minutes
    retry: 1,
  })
}

export function usePreReleaseCheck() {
  return useQuery<PreReleaseCheckResponse>({
    queryKey: ['update', 'pre-release'],
    queryFn: () => api.checkPreRelease(),
    staleTime: 1000 * 60 * 30, // 30 minutes
    retry: 1,
  })
}

export function useInstallUpdate() {
  const queryClient = useQueryClient()

  return useMutation<UpdateInstallResponse, Error, string>({
    mutationFn: (releaseTag) => api.installUpdate(releaseTag),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['update', 'check'] })
      queryClient.invalidateQueries({ queryKey: ['update', 'pre-release'] })
    },
  })
}