import { create } from 'zustand'
import type {
  ExportOptions,
  OutputFormat,
  PipelineStep,
  ProcessingTask,
} from '../types/api'

/**
 * A single export waiting for (or using) the one render slot the backend
 * exposes. Files stay in memory: the queue is deliberately not persisted,
 * because a `File` handle cannot be restored after a reload.
 */
export type RenderJobStatus =
  | 'queued'
  | 'uploading'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface RenderJobInput {
  file: File
  fileName: string
  fileSize: number
  fileType: string
  steps: PipelineStep[]
  outputFormat: OutputFormat
  quality: string
  exportOptions?: ExportOptions
}

export interface RenderJob extends RenderJobInput {
  id: string
  status: RenderJobStatus
  progress: number
  message: string
  taskId: string | null
  /** Epoch ms when the user queued the job. */
  createdAt: number
  /** Epoch ms when the upload/render started (used for total duration). */
  startedAt: number | null
  /** Epoch ms when the backend task actually started (used for the ETA). */
  processingStartedAt: number | null
  completedAt: number | null
  outputPath: string | null
  error: string | null
  /** Smoothed estimate of the seconds still needed for this render. */
  etaSeconds: number | null
  /** Total wall-clock seconds once the job has finished. */
  durationSeconds: number | null
  /** Do not claim before this epoch ms (used to wait out another backend task). */
  retryAfter: number | null
  /** Set while a running export is being cancelled. */
  cancelRequested: boolean
}

export type QueueSortOrder = 'oldest' | 'newest'

interface RenderQueueState {
  jobs: RenderJob[]
  isOpen: boolean
  sortOrder: QueueSortOrder
  /** Show jobs from every source, or only the file currently on the canvas. */
  allSources: boolean
  openQueue: () => void
  closeQueue: () => void
  setSortOrder: (order: QueueSortOrder) => void
  setAllSources: (allSources: boolean) => void
  enqueue: (input: RenderJobInput) => RenderJob
  claimNext: () => RenderJob | null
  updateJob: (id: string, patch: Partial<RenderJob>) => void
  applyTask: (id: string, task: ProcessingTask) => void
  failJob: (id: string, error: string) => void
  cancelQueued: (id: string) => void
  requestCancel: (id: string) => void
  retryJob: (id: string) => void
  removeJob: (id: string) => void
  clearFinished: () => void
}

export function isFinishedStatus(status: RenderJobStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}

export function isActiveStatus(status: RenderJobStatus): boolean {
  return status === 'uploading' || status === 'processing'
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `job-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function mapTaskStatus(status: ProcessingTask['status']): RenderJobStatus {
  switch (status) {
    case 'completed':
      return 'completed'
    case 'failed':
      return 'failed'
    case 'aborted':
      return 'cancelled'
    case 'pending':
    case 'processing':
    default:
      return 'processing'
  }
}

/**
 * Estimate the remaining time from the progress made since the backend task
 * started. The estimate is smoothed so the number does not jump every second.
 */
function estimateRemainingSeconds(job: RenderJob, progress: number, now: number): number | null {
  if (!job.processingStartedAt || progress < 1) return job.etaSeconds

  const elapsed = (now - job.processingStartedAt) / 1000
  if (elapsed < 2) return job.etaSeconds

  const total = elapsed / Math.max(progress / 100, 0.01)
  const raw = Math.max(0, total - elapsed)
  const previous = job.etaSeconds

  return previous == null ? raw : previous * 0.65 + raw * 0.35
}

const MAX_FINISHED_JOBS = 50

function pruneJobs(jobs: RenderJob[]): RenderJob[] {
  const finished = jobs.filter((job) => isFinishedStatus(job.status))
  if (finished.length <= MAX_FINISHED_JOBS) return jobs

  const keep = new Set(
    finished
      .slice()
      .sort((a, b) => (b.completedAt ?? b.createdAt) - (a.completedAt ?? a.createdAt))
      .slice(0, MAX_FINISHED_JOBS)
      .map((job) => job.id)
  )

  return jobs.filter((job) => !isFinishedStatus(job.status) || keep.has(job.id))
}

export const useRenderQueueStore = create<RenderQueueState>()((set, get) => ({
  jobs: [],
  isOpen: false,
  sortOrder: 'oldest',
  allSources: true,

  openQueue: () => set({ isOpen: true }),
  closeQueue: () => set({ isOpen: false }),
  setSortOrder: (sortOrder) => set({ sortOrder }),
  setAllSources: (allSources) => set({ allSources }),

  enqueue: (input) => {
    const job: RenderJob = {
      ...input,
      // Snapshot the pipeline: later edits in the sidebar must not change what
      // an already-queued export runs.
      steps: input.steps.map((step) => ({
        ...step,
        parameters: { ...step.parameters },
      })),
      exportOptions: input.exportOptions ? { ...input.exportOptions } : undefined,
      id: makeId(),
      status: 'queued',
      progress: 0,
      message: 'Waiting to start',
      taskId: null,
      createdAt: Date.now(),
      startedAt: null,
      processingStartedAt: null,
      completedAt: null,
      outputPath: null,
      error: null,
      etaSeconds: null,
      durationSeconds: null,
      retryAfter: null,
      cancelRequested: false,
    }

    set((state) => ({ jobs: pruneJobs([...state.jobs, job]) }))
    return job
  },

  claimNext: () => {
    const now = Date.now()
    let claimed: RenderJob | null = null

    set((state) => {
      const candidate = state.jobs
        .filter(
          (job) =>
            job.status === 'queued' &&
            (job.retryAfter == null || job.retryAfter <= now)
        )
        .sort((a, b) => a.createdAt - b.createdAt)[0]

      if (!candidate) return state

      claimed = {
        ...candidate,
        status: 'uploading',
        progress: 0,
        message: 'Preparing export…',
        startedAt: Date.now(),
        processingStartedAt: null,
        completedAt: null,
        error: null,
        etaSeconds: null,
        durationSeconds: null,
        retryAfter: null,
        cancelRequested: false,
      }

      return {
        jobs: state.jobs.map((job) => (job.id === claimed!.id ? claimed! : job)),
      }
    })

    return claimed
  },

  updateJob: (id, patch) =>
    set((state) => ({
      jobs: state.jobs.map((job) => (job.id === id ? { ...job, ...patch } : job)),
    })),

  applyTask: (id, task) =>
    set((state) => ({
      jobs: state.jobs.map((job) => {
        if (job.id !== id) return job
        // A finished row is terminal. A late poll response must not revive it.
        if (isFinishedStatus(job.status)) return job

        const now = Date.now()
        const status = mapTaskStatus(task.status)
        const finished = isFinishedStatus(status)
        const progress = status === 'completed'
          ? 100
          : Math.max(job.progress, Math.min(100, Math.max(0, task.progress ?? 0)))
        const etaSeconds = finished
          ? status === 'completed' ? 0 : null
          : status === 'processing'
            ? estimateRemainingSeconds(job, progress, now)
            : null
        const message = job.cancelRequested && !finished
          ? 'Cancelling…'
          : task.message || job.message

        return {
          ...job,
          status,
          progress,
          message,
          error: task.error ?? job.error,
          outputPath: task.output_path ?? job.outputPath,
          etaSeconds,
          completedAt: finished ? (job.completedAt ?? now) : job.completedAt,
          durationSeconds: finished
            ? (now - (job.startedAt ?? now)) / 1000
            : job.durationSeconds,
        }
      }),
    })),

  failJob: (id, error) =>
    set((state) => ({
      jobs: state.jobs.map((job) => {
        if (job.id !== id || isFinishedStatus(job.status)) return job
        const now = Date.now()
        return {
          ...job,
          status: 'failed',
          message: 'Export failed',
          error,
          etaSeconds: null,
          completedAt: now,
          durationSeconds: (now - (job.startedAt ?? now)) / 1000,
        }
      }),
    })),

  cancelQueued: (id) =>
    set((state) => ({
      jobs: state.jobs.map((job) => {
        if (job.id !== id || job.status !== 'queued') return job
        const now = Date.now()
        return {
          ...job,
          status: 'cancelled',
          message: 'Cancelled',
          completedAt: now,
          durationSeconds: 0,
          etaSeconds: null,
        }
      }),
    })),

  requestCancel: (id) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.id === id && !isFinishedStatus(job.status)
          ? { ...job, cancelRequested: true, message: 'Cancelling…' }
          : job
      ),
    })),

  retryJob: (id) =>
    set((state) => ({
      jobs: state.jobs.map((job) => {
        if (job.id !== id || (job.status !== 'failed' && job.status !== 'cancelled')) {
          return job
        }
        return {
          ...job,
          status: 'queued',
          progress: 0,
          message: 'Queued again',
          taskId: null,
          startedAt: null,
          processingStartedAt: null,
          completedAt: null,
          outputPath: null,
          error: null,
          etaSeconds: null,
          durationSeconds: null,
          retryAfter: null,
          cancelRequested: false,
        }
      }),
    })),

  removeJob: (id) =>
    set((state) => ({ jobs: state.jobs.filter((job) => job.id !== id) })),

  clearFinished: () => {
    const jobs = get().jobs.filter((job) => !isFinishedStatus(job.status))
    set({ jobs })
  },
}))
