import { useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import { usePipelineStore, useProcessingStore } from '../store'
import {
  isActiveStatus,
  isFinishedStatus,
  useRenderQueueStore,
  type RenderJob,
} from '../store/renderQueue'
import type { ProcessingTask } from '../types/api'

/**
 * Runs queued exports one at a time. The backend owns a single render slot,
 * so the queue uploads the next source only after the previous task leaves the
 * `processing` state. Progress is polled once a second so the row can show a
 * percentage and a smoothed ETA; the same data is mirrored into the processing
 * store so the canvas keeps showing the enhanced result for the loaded file.
 */
export function RenderQueueRunner() {
  const jobs = useRenderQueueStore((state) => state.jobs)
  const claimNext = useRenderQueueStore((state) => state.claimNext)
  const updateJob = useRenderQueueStore((state) => state.updateJob)
  const failJob = useRenderQueueStore((state) => state.failJob)
  const applyTask = useRenderQueueStore((state) => state.applyTask)
  const startingRef = useRef(false)
  const [wakeTick, setWakeTick] = useState(0)

  const activeJob = jobs.find((job) => isActiveStatus(job.status)) ?? null
  const processingJob = jobs.find((job) => job.status === 'processing' && job.taskId) ?? null

  // -- Start the next job -------------------------------------------------
  useEffect(() => {
    if (activeJob || startingRef.current) return

    const now = Date.now()
    const next = jobs
      .filter((job) => job.status === 'queued' && (job.retryAfter == null || job.retryAfter <= now))
      .sort((a, b) => a.createdAt - b.createdAt)[0]

    if (!next) {
      // A job that hit the backend's "one task at a time" guard is parked with
      // a retry delay. Wake the effect when that delay expires.
      const soonest = jobs
        .filter((job) => job.status === 'queued' && job.retryAfter != null)
        .reduce<number | null>(
          (min, job) => (min == null || job.retryAfter! < min ? job.retryAfter! : min),
          null
        )
      if (soonest != null) {
        const timer = window.setTimeout(
          () => setWakeTick((tick) => tick + 1),
          Math.max(250, soonest - Date.now())
        )
        return () => window.clearTimeout(timer)
      }
      return
    }

    startingRef.current = true
    const claimed = claimNext()
    if (!claimed) {
      startingRef.current = false
      return
    }

    void (async () => {
      try {
        const task = await api.startProcessingWithFile(
          claimed.file,
          claimed.steps,
          claimed.outputFormat,
          claimed.quality,
          claimed.exportOptions,
          (percent) => {
            updateJob(claimed.id, {
              status: 'uploading',
              progress: percent,
              message: `Uploading ${Math.round(percent)}%`,
            })
          }
        )

        const latest = useRenderQueueStore.getState().jobs.find((job) => job.id === claimed.id)
        if (!latest) return

        if (latest.status === 'cancelled') {
          // The row was removed while the upload was in flight; stop the task
          // the backend just created so the queue can move on.
          try {
            await api.cancelTask(task.task_id)
          } catch {
            // The task may have finished already; nothing else to do.
          }
          return
        }

        const cancelling = latest.cancelRequested
        updateJob(claimed.id, {
          status: 'processing',
          taskId: task.task_id,
          processingStartedAt: Date.now(),
          progress: 0,
          message: cancelling ? 'Cancelling…' : 'Starting processing…',
          retryAfter: null,
        })

        if (cancelling) {
          try {
            await api.cancelTask(task.task_id)
          } catch {
            // Polling will surface the final state either way.
          }
          return
        }

        syncStartedTask(claimed, task.task_id)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        const latest = useRenderQueueStore.getState().jobs.find((job) => job.id === claimed.id)
        if (!latest) return

        if (latest.cancelRequested) {
          const finishedAt = Date.now()
          updateJob(claimed.id, {
            status: 'cancelled',
            message: 'Cancelled',
            completedAt: finishedAt,
            durationSeconds: (finishedAt - (latest.startedAt ?? finishedAt)) / 1000,
            etaSeconds: null,
          })
          return
        }

        if (/already processing/i.test(message)) {
          // Another task (a preview, or a late cancel) still owns the backend.
          // Park this job and retry shortly; its place in line is preserved.
          updateJob(claimed.id, {
            status: 'queued',
            progress: 0,
            message: 'Waiting for the current task to finish…',
            retryAfter: Date.now() + 3000,
          })
          return
        }

        failJob(claimed.id, message)
      } finally {
        startingRef.current = false
      }
    })()
  }, [activeJob, jobs, claimNext, updateJob, failJob, wakeTick])

  // -- Poll the active task ----------------------------------------------
  useEffect(() => {
    if (!processingJob?.taskId) return

    const jobId = processingJob.id
    const taskId = processingJob.taskId
    let stopped = false
    let failures = 0
    let timer: number | undefined

    const poll = async () => {
      try {
        const task = await api.getTaskStatus(taskId)
        if (stopped) return
        failures = 0

        applyTask(jobId, task)

        const job = useRenderQueueStore.getState().jobs.find((row) => row.id === jobId)
        if (!job) return

        if (isFinishedStatus(job.status)) {
          syncFinishedTask(job, task)
          return
        }

        syncJobProgress(job)
      } catch (error) {
        failures += 1
        if (failures >= 5) {
          failJob(jobId, error instanceof Error ? error.message : String(error))
          return
        }
      }

      if (!stopped) {
        timer = window.setTimeout(() => void poll(), 1000)
      }
    }

    void poll()

    return () => {
      stopped = true
      if (timer) window.clearTimeout(timer)
    }
  }, [processingJob?.id, processingJob?.taskId, applyTask, failJob])

  return null
}

function toProcessingTask(job: RenderJob, task: ProcessingTask): ProcessingTask {
  return {
    task_id: task.task_id,
    status: task.status,
    progress: task.progress ?? job.progress,
    message: job.message || task.message,
    current_step: task.current_step ?? null,
    created_at: new Date(job.createdAt).toISOString(),
    started_at: new Date(job.startedAt ?? job.createdAt).toISOString(),
    completed_at: job.completedAt ? new Date(job.completedAt).toISOString() : null,
    error: task.error ?? job.error,
    result: task.result ?? null,
    output_path: task.output_path ?? job.outputPath,
  }
}

function isCurrentFile(job: RenderJob): boolean {
  return usePipelineStore.getState().selectedFiles?.[0]?.name === job.fileName
}

/** Mirror an in-flight render into the canvas overlay for the loaded file. */
function syncStartedTask(job: RenderJob, taskId: string): void {
  if (!isCurrentFile(job)) return

  useProcessingStore.getState().setCurrentTask({
    task_id: taskId,
    status: 'processing',
    progress: 0,
    message: 'Starting processing…',
    current_step: null,
    created_at: new Date(job.createdAt).toISOString(),
    started_at: new Date().toISOString(),
    completed_at: null,
    error: null,
    result: null,
    output_path: null,
  })
}

function syncJobProgress(job: RenderJob): void {
  if (!isCurrentFile(job)) return

  const store = useProcessingStore.getState()
  const current = store.currentTask
  if (!current || current.task_id !== job.taskId) return

  store.applyTaskStatus({
    ...current,
    progress: job.progress,
    message: job.message,
  })
}

function syncFinishedTask(job: RenderJob, task: ProcessingTask): void {
  const processingTask = toProcessingTask(job, task)
  const store = useProcessingStore.getState()

  if (isCurrentFile(job)) {
    store.applyTaskStatus(processingTask)
  }
  store.addToHistory(processingTask)
}
