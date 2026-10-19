import { useMemo } from 'react'
import {
  Ban,
  Clock3,
  Download,
  Film,
  Image as ImageIcon,
  Layers,
  ListVideo,
  Loader2,
  RotateCw,
  Trash2,
} from 'lucide-react'
import { clsx } from 'clsx'
import { Dialog, Button, Select, Toggle } from './ui'
import { api } from '../services/api'
import { usePipelineStore } from '../store'
import {
  isActiveStatus,
  isFinishedStatus,
  useRenderQueueStore,
  type QueueSortOrder,
  type RenderJob,
  type RenderJobStatus,
} from '../store/renderQueue'
import {
  describeJobFilters,
  describeJobOutput,
  fileNameFromPath,
  formatBytes,
  formatDuration,
} from '../utils/renderQueue'

const SORT_OPTIONS = [
  { value: 'oldest', label: 'Oldest first' },
  { value: 'newest', label: 'Newest first' },
]

const ROW_GRID =
  'grid grid-cols-[minmax(200px,1.15fr)_minmax(220px,1.05fr)_minmax(260px,1fr)] gap-4'

const ICON_BUTTON =
  'grid place-items-center w-6 h-6 rounded-[var(--radius-sm)] text-[var(--color-text-muted)] transition-colors ' +
  'hover:text-[var(--color-text-primary)] hover:bg-white/[0.07] disabled:opacity-40 disabled:pointer-events-none'

interface StatusTone {
  label: string
  dot: string
  bar: string
  text: string
}

function statusTone(status: RenderJobStatus): StatusTone {
  switch (status) {
    case 'queued':
      return {
        label: 'Queued',
        dot: 'bg-[var(--color-text-muted)]',
        bar: 'bg-[var(--color-text-muted)]',
        text: 'text-[var(--color-text-secondary)]',
      }
    case 'uploading':
      return {
        label: 'Uploading',
        dot: 'bg-[var(--color-accent)] animate-pulse',
        bar: 'bg-[var(--color-accent)]',
        text: 'text-[var(--color-text-secondary)]',
      }
    case 'processing':
      return {
        label: 'Exporting',
        dot: 'bg-[var(--color-accent)] animate-pulse',
        bar: 'bg-[var(--color-accent)]',
        text: 'text-[var(--color-text-secondary)]',
      }
    case 'completed':
      return {
        label: 'Completed',
        dot: 'bg-emerald-500',
        bar: 'bg-emerald-500',
        text: 'text-emerald-400',
      }
    case 'failed':
      return {
        label: 'Failed',
        dot: 'bg-[var(--color-error)]',
        bar: 'bg-[var(--color-error)]',
        text: 'text-[var(--color-error)]',
      }
    case 'cancelled':
    default:
      return {
        label: 'Cancelled',
        dot: 'bg-[var(--color-text-muted)]',
        bar: 'bg-[var(--color-border-strong)]',
        text: 'text-[var(--color-text-muted)]',
      }
  }
}

function statusDetail(job: RenderJob, position: number | null): string {
  switch (job.status) {
    case 'queued':
      return position != null ? `#${position} in queue` : 'Waiting'
    case 'uploading':
      return job.progress > 0 ? `${Math.round(job.progress)}%` : 'Preparing…'
    case 'processing': {
      const percent = `${Math.round(job.progress)}%`
      return job.etaSeconds != null ? `${percent} · ~${formatDuration(job.etaSeconds)} left` : percent
    }
    case 'completed':
      return job.durationSeconds != null ? `in ${formatDuration(job.durationSeconds)}` : 'Done'
    case 'failed':
      return 'Needs attention'
    case 'cancelled':
      return 'Stopped'
    default:
      return ''
  }
}

function statusMessage(job: RenderJob): string {
  switch (job.status) {
    case 'queued':
      return job.message || 'Waiting to start'
    case 'uploading':
      return job.message || 'Preparing export…'
    case 'processing':
      return job.message || 'Enhancing…'
    case 'completed':
      return job.outputPath ? fileNameFromPath(job.outputPath) : 'Saved'
    case 'failed':
      return job.error || job.message
    case 'cancelled':
    default:
      return job.message || 'Cancelled'
  }
}

function StatusProgress({ job, tone }: { job: RenderJob; tone: StatusTone }) {
  if (job.status === 'uploading') {
    // Real upload progress when the browser reports it, indeterminate before
    // the first progress event arrives.
    return (
      <div className="progress">
        {job.progress > 0 ? (
          <div
            className={clsx('h-full rounded-full transition-all duration-300 ease-out', tone.bar)}
            style={{ width: `${Math.min(100, job.progress)}%` }}
          />
        ) : (
          <div className="queue-indeterminate-bar" />
        )}
      </div>
    )
  }

  const width = job.status === 'completed' ? 100 : Math.max(0, Math.min(100, job.progress))

  return (
    <div className="progress">
      <div
        className={clsx('h-full rounded-full transition-all duration-500 ease-out', tone.bar)}
        style={{ width: `${width}%` }}
      />
    </div>
  )
}

interface RowActionsProps {
  job: RenderJob
  onCancel: (job: RenderJob) => void
  onRetry: (job: RenderJob) => void
  onRemove: (job: RenderJob) => void
  onDownload: (job: RenderJob) => void
}

function RowActions({ job, onCancel, onRetry, onRemove, onDownload }: RowActionsProps) {
  if (job.status === 'queued' || job.status === 'uploading' || job.status === 'processing') {
    const cancelling = job.cancelRequested
    return (
      <div className="flex items-center gap-0.5 shrink-0">
        <button
          type="button"
          className={ICON_BUTTON}
          onClick={() => onCancel(job)}
          disabled={cancelling}
          title={job.status === 'queued' ? 'Remove from queue' : 'Cancel export'}
          aria-label={job.status === 'queued' ? 'Remove from queue' : 'Cancel export'}
        >
          {cancelling ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Ban className="w-3.5 h-3.5" />}
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-0.5 shrink-0">
      {job.status === 'completed' && (
        <button
          type="button"
          className={ICON_BUTTON}
          onClick={() => onDownload(job)}
          title="Open result"
          aria-label="Open result"
        >
          <Download className="w-3.5 h-3.5" />
        </button>
      )}
      {(job.status === 'failed' || job.status === 'cancelled') && (
        <button
          type="button"
          className={ICON_BUTTON}
          onClick={() => onRetry(job)}
          title="Export again"
          aria-label="Export again"
        >
          <RotateCw className="w-3.5 h-3.5" />
        </button>
      )}
      <button
        type="button"
        className={ICON_BUTTON}
        onClick={() => onRemove(job)}
        title="Remove from list"
        aria-label="Remove from list"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

export function ExportQueueDialog() {
  const isOpen = useRenderQueueStore((state) => state.isOpen)
  const closeQueue = useRenderQueueStore((state) => state.closeQueue)
  const jobs = useRenderQueueStore((state) => state.jobs)
  const sortOrder = useRenderQueueStore((state) => state.sortOrder)
  const setSortOrder = useRenderQueueStore((state) => state.setSortOrder)
  const allSources = useRenderQueueStore((state) => state.allSources)
  const setAllSources = useRenderQueueStore((state) => state.setAllSources)
  const cancelQueued = useRenderQueueStore((state) => state.cancelQueued)
  const requestCancel = useRenderQueueStore((state) => state.requestCancel)
  const retryJob = useRenderQueueStore((state) => state.retryJob)
  const removeJob = useRenderQueueStore((state) => state.removeJob)
  const clearFinished = useRenderQueueStore((state) => state.clearFinished)
  const updateJob = useRenderQueueStore((state) => state.updateJob)

  const selectedFileName = usePipelineStore((state) => state.selectedFiles?.[0]?.name ?? null)

  const activeJobs = jobs.filter((job) => isActiveStatus(job.status))
  const queuedJobs = jobs.filter((job) => job.status === 'queued')
  const finishedJobs = jobs.filter((job) => isFinishedStatus(job.status))
  const activeJob = activeJobs[0] ?? null

  const queuedOrder = useMemo(
    () => queuedJobs.slice().sort((a, b) => a.createdAt - b.createdAt).map((job) => job.id),
    [queuedJobs]
  )

  const visibleJobs = useMemo(() => {
    const filtered =
      allSources || !selectedFileName
        ? jobs
        : jobs.filter((job) => job.fileName === selectedFileName)
    return filtered
      .slice()
      .sort((a, b) =>
        sortOrder === 'oldest' ? a.createdAt - b.createdAt : b.createdAt - a.createdAt
      )
  }, [jobs, allSources, selectedFileName, sortOrder])

  const queuePosition = (job: RenderJob): number | null => {
    if (job.status !== 'queued') return null
    const index = queuedOrder.indexOf(job.id)
    return index < 0 ? null : activeJobs.length + index + 1
  }

  const handleCancel = async (job: RenderJob) => {
    if (job.status === 'queued') {
      cancelQueued(job.id)
      return
    }

    requestCancel(job.id)
    if (job.status === 'processing' && job.taskId) {
      try {
        await api.cancelTask(job.taskId)
      } catch (error) {
        updateJob(job.id, {
          message: error instanceof Error ? error.message : 'Could not cancel this export',
        })
      }
    }
  }

  const handleDownload = (job: RenderJob) => {
    if (!job.taskId) return
    window.open(api.taskResultUrl(job.taskId), '_blank')
  }

  return (
    <Dialog
      open={isOpen}
      onClose={closeQueue}
      title="Export queue"
      icon={<Layers className="w-4 h-4" />}
      size="max-w-[1180px]"
      className="h-[min(82vh,760px)]"
      bodyClassName="p-0 flex flex-1 flex-col min-h-0 overflow-hidden"
      headerActions={
        <>
          <div className="hidden sm:flex items-center gap-2 text-[11px] text-[var(--color-text-secondary)] select-none">
            All sources
            <Toggle
              size="sm"
              checked={allSources}
              onChange={setAllSources}
              aria-label="Show exports from all sources"
            />
          </div>
          <Select
            value={sortOrder}
            onChange={(event) => setSortOrder(event.target.value as QueueSortOrder)}
            options={SORT_OPTIONS}
            className="h-8 w-[142px] text-[12px]"
            aria-label="Sort exports"
          />
        </>
      }
      footer={
        <div className="w-full flex items-center justify-between gap-4">
          <div className="flex items-center gap-4 text-[11px] text-[var(--color-text-secondary)]">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-accent)]" />
              {activeJobs.length} exporting
            </span>
            <span>{queuedJobs.length} queued</span>
            <span>{finishedJobs.length} finished</span>
          </div>
          <div className="flex items-center gap-3">
            {activeJob?.etaSeconds != null && (
              <span className="flex items-center gap-1.5 text-[11px] text-[var(--color-text-secondary)]">
                <Clock3 className="w-3.5 h-3.5" />
                Current export · ~{formatDuration(activeJob.etaSeconds)} left
              </span>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={clearFinished}
              disabled={finishedJobs.length === 0}
            >
              Clear finished
            </Button>
          </div>
        </div>
      }
    >
      <div
        className={clsx(
          ROW_GRID,
          'h-9 shrink-0 items-center px-5 text-[11px] font-medium text-[var(--color-text-muted)]',
          'bg-[var(--color-bg-base)]/40 border-b border-[var(--color-border)]'
        )}
      >
        <span>Exports</span>
        <span>Properties</span>
        <span>Status</span>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {visibleJobs.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-2 text-center px-6 py-12">
            <ListVideo className="w-8 h-8 text-[var(--color-border-strong)]" />
            <p className="text-[13px] font-medium text-[var(--color-text-primary)]">
              No exports in the queue
            </p>
            <p className="text-[11px] text-[var(--color-text-muted)] max-w-[320px]">
              Set up your filters and choose Export. Renders are processed one at a time and
              show their progress and estimated time here.
            </p>
          </div>
        ) : (
          visibleJobs.map((job) => {
            const tone = statusTone(job.status)
            const isVideo = job.fileType.startsWith('video/')
            const filters = describeJobFilters(job)
            const output = describeJobOutput(job)

            return (
              <div
                key={job.id}
                className={clsx(
                  ROW_GRID,
                  'items-center px-5 py-3 border-b border-[var(--color-border-subtle)]',
                  'transition-colors hover:bg-white/[0.02]'
                )}
              >
                {/* Exports */}
                <div className="flex items-center gap-3 min-w-0">
                  <div className="grid place-items-center w-8 h-8 shrink-0 rounded-[var(--radius-md)] bg-[var(--color-bg-input)] border border-[var(--color-border)] text-[var(--color-text-muted)]">
                    {isVideo ? <Film className="w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
                  </div>
                  <div className="min-w-0">
                    <div
                      className="truncate text-[12.5px] font-medium text-[var(--color-text-primary)]"
                      title={job.fileName}
                    >
                      {job.fileName}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5 text-[10.5px] text-[var(--color-text-muted)]">
                      <span>{formatBytes(job.fileSize)}</span>
                      <span className="text-[var(--color-border)]">·</span>
                      <span>
                        {job.steps.length} filter{job.steps.length === 1 ? '' : 's'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Properties */}
                <div className="min-w-0">
                  <div
                    className="truncate text-[11.5px] text-[var(--color-text-secondary)]"
                    title={filters}
                  >
                    {filters}
                  </div>
                  <div
                    className="mt-0.5 truncate text-[10.5px] text-[var(--color-text-muted)]"
                    title={output}
                  >
                    {output}
                  </div>
                </div>

                {/* Status */}
                <div className="min-w-0">
                  <div className="flex items-center justify-between gap-3">
                    <span className={clsx('inline-flex items-center gap-1.5 text-[11px] font-medium', tone.text)}>
                      <span className={clsx('w-1.5 h-1.5 rounded-full', tone.dot)} />
                      {tone.label}
                    </span>
                    <span className="text-[10.5px] font-mono text-[var(--color-text-muted)] whitespace-nowrap">
                      {statusDetail(job, queuePosition(job))}
                    </span>
                  </div>

                  <div className="mt-1.5">
                    <StatusProgress job={job} tone={tone} />
                  </div>

                  <div className="mt-1.5 flex items-center justify-between gap-2">
                    <span
                      className="truncate text-[10.5px] text-[var(--color-text-muted)]"
                      title={statusMessage(job)}
                    >
                      {statusMessage(job)}
                    </span>
                    <RowActions
                      job={job}
                      onCancel={handleCancel}
                      onRetry={(target) => retryJob(target.id)}
                      onRemove={(target) => removeJob(target.id)}
                      onDownload={handleDownload}
                    />
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </Dialog>
  )
}
