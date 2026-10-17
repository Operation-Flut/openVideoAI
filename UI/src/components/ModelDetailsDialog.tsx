import { Download, Trash2, Boxes } from 'lucide-react'
import { clsx } from 'clsx'
import { Dialog, Button, Progress } from './ui'
import {
  FEATURE_TAG_STYLES, FALLBACK_TAG_STYLE, STATUS_BADGE, STATUS_LABEL, statusFor,
} from '../utils/modelMeta'
import type { ModelResponse } from '../types/api'

interface ModelDetailsDialogProps {
  model: ModelResponse | null
  onClose: () => void
  onDownload: () => void
  onDelete: () => void
  downloading: boolean
  downloadProgress?: number
  deleting: boolean
}

const yesNo = (value: boolean) => (value ? 'Yes' : 'No')

/**
 * Popup showing every detail known about a model: description, features,
 * specifications, compatibility, paths and download sources.
 */
export function ModelDetailsDialog({
  model, onClose, onDownload, onDelete, downloading, downloadProgress, deleting,
}: ModelDetailsDialogProps) {
  if (!model) return null

  const status = statusFor(model)
  const canDownload = !model.downloaded && model.has_download_url && model.compatible && !model.is_builtin
  const canDelete = model.downloaded && !model.is_builtin
  const pathEntries = Object.entries(model.additional_paths ?? {})

  return (
    <Dialog
      open
      onClose={onClose}
      icon={<Boxes className="w-4 h-4" />}
      title={model.name}
      description={model.is_ai ? 'AI model' : 'Classical model'}
      footer={
        <>
          <Button variant="secondary" size="sm" onClick={onClose}>Close</Button>
          {canDelete && (
            <Button variant="danger" size="sm" onClick={onDelete} disabled={deleting || downloading}>
              <Trash2 className="w-3.5 h-3.5" />
              {deleting ? 'Deleting' : 'Delete model'}
            </Button>
          )}
          {canDownload && (
            <Button
              variant="primary"
              size="sm"
              onClick={onDownload}
              disabled={downloading || deleting}
              loading={downloading}
            >
              {!downloading && <Download className="w-3.5 h-3.5" />}
              {downloading ? 'Downloading' : 'Download'}
            </Button>
          )}
        </>
      }
    >
      {/* Status + feature tags */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className={clsx('badge', STATUS_BADGE[status])}>{STATUS_LABEL[status]}</span>
        {model.features.map((feature) => (
          <span
            key={feature}
            className={clsx(
              'inline-flex items-center rounded-[3px] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide leading-none',
              FEATURE_TAG_STYLES[feature] || FALLBACK_TAG_STYLE
            )}
          >
            {feature}
          </span>
        ))}
      </div>

      {/* Description */}
      <p className="mt-4 text-[13px] leading-relaxed text-[var(--color-text-secondary)]">
        {model.full_description || model.description}
      </p>

      {/* Specifications */}
      <div className="detail-section mt-4">
        <div className="detail-section-title">Specifications</div>
        <div className="detail-grid">
          <DetailItem label="Size" value={model.size} />
          <DetailItem label="Tier" value={model.weight} />
          <DetailItem label="Type" value={model.is_ai ? 'AI' : 'Classical'} />
          <DetailItem label="Status" value={STATUS_LABEL[status]} />
          <DetailItem label="Downloaded" value={yesNo(model.downloaded)} />
          <DetailItem label="Built-in" value={yesNo(model.is_builtin)} />
          <DetailItem label="Implemented" value={yesNo(model.implemented)} />
          <DetailItem label="Compatible" value={yesNo(model.compatible)} />
          <DetailItem label="Runs on CPU" value={yesNo(model.can_run_on_cpu)} />
          <DetailItem label="Requires CUDA" value={yesNo(model.requires_cuda)} />
          <DetailItem label="Requires Metal" value={yesNo(model.requires_metal)} />
          {model.downloaded && <DetailItem label="Verified" value={yesNo(!!model.verified)} />}
          {model.filename && <DetailItem label="File name" value={model.filename} mono />}
          {model.module && <DetailItem label="Module" value={model.module} mono />}
        </div>
      </div>

      {/* Compatibility */}
      {(model.compatibility_reason || !model.implemented) && (
        <div className="detail-section">
          <div className="detail-section-title">Compatibility</div>
          {!model.implemented && (
            <p className="text-[12px] text-amber-400">
              No runtime implementation in this build — this catalog entry is declared for
              reference and is not offered as a pipeline step.
            </p>
          )}
          {model.compatibility_reason && (
            <p className={clsx('text-[12px]', model.compatible ? 'text-[var(--color-text-secondary)]' : 'text-amber-400')}>
              {model.compatibility_reason}
            </p>
          )}
        </div>
      )}

      {/* Last error, if any */}
      {model.error && (
        <div className="detail-section">
          <div className="detail-section-title">Error</div>
          <p className="text-[12px] text-red-400">{model.error}</p>
        </div>
      )}

      {/* Paths */}
      {(model.local_path || pathEntries.length > 0) && (
        <div className="detail-section">
          <div className="detail-section-title">Paths</div>
          <div className="space-y-1.5">
            {model.local_path && <PathRow label="Local" value={model.local_path} />}
            {pathEntries.map(([key, value]) => (
              <PathRow key={key} label={key} value={value} />
            ))}
          </div>
        </div>
      )}

      {/* Download sources */}
      {(model.source_url || model.download_url) && (
        <div className="detail-section">
          <div className="detail-section-title">Download</div>
          <div className="space-y-1.5">
            {model.source_url && <PathRow label="Source" value={model.source_url} />}
            {model.download_url && <PathRow label="URL" value={model.download_url} />}
          </div>
        </div>
      )}

      {/* Live download progress */}
      {downloading && downloadProgress !== undefined && (
        <div className="mt-4">
          <Progress value={downloadProgress} showLabel label="Downloading model..." />
        </div>
      )}
    </Dialog>
  )
}

function DetailItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="detail-label text-[11px]">{label}</div>
      <div className={clsx('detail-value text-[12px] truncate', mono && 'font-mono')} title={value}>
        {value}
      </div>
    </div>
  )
}

function PathRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2 min-w-0">
      <span className="shrink-0 w-16 pt-0.5 text-[11px] text-[var(--color-text-muted)] capitalize">{label}</span>
      <span className="min-w-0 flex-1 font-mono text-[11px] text-[var(--color-text-secondary)] break-all">
        {value}
      </span>
    </div>
  )
}
