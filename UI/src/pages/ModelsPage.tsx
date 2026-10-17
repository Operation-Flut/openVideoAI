import { useState, useMemo } from 'react'
import {
  Download, Trash2, RefreshCw, AlertCircle, CheckCircle2, Loader2,
  Search, Database, Cpu, HardDrive, Gauge, Info, ChevronRight, Sparkles,
} from 'lucide-react'
import { useModels, useDownloadModel, useDeleteModel, useFeatures, useSystemStatus } from '../hooks/useModels'
import { Button, Progress } from '../components/ui'
import { ModelDetailsDialog } from '../components/ModelDetailsDialog'
import {
  FEATURE_TAG_STYLES, FALLBACK_TAG_STYLE, STATUS_BADGE, STATUS_LABEL, statusFor,
} from '../utils/modelMeta'
import { displayFeature, useFeatureDisplayNames } from '../utils/displayNames'
import { clsx } from 'clsx'
import type { ModelResponse } from '../types/api'

type AiFilter = 'all' | 'ai' | 'non-ai'

const AI_FILTERS: { value: AiFilter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'ai', label: 'AI' },
  { value: 'non-ai', label: 'Non-AI' },
]

/** Parse a catalog size string such as "~464MB" or "~6.4GB" into megabytes. */
function parseSizeMB(size: string): number {
  const match = size.toLowerCase().match(/([\d.]+)\s*(tb|gb|mb|kb)/)
  if (!match) return 1 // "Built-in (< 1MB)" and similar tiny/bundled models
  const value = parseFloat(match[1])
  switch (match[2]) {
    case 'tb': return value * 1024 * 1024
    case 'gb': return value * 1024
    case 'kb': return value / 1024
    default: return value
  }
}

export function ModelsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedFeature, setSelectedFeature] = useState<string>('all')
  const [showDownloadedOnly, setShowDownloadedOnly] = useState(false)
  const [showCompatibleOnly, setShowCompatibleOnly] = useState(false)
  const [aiFilter, setAiFilter] = useState<AiFilter>('all')
  const [showRecommendedOnly, setShowRecommendedOnly] = useState(false)
  const [detailsModel, setDetailsModel] = useState<ModelResponse | null>(null)

  const { data: models, isLoading, error, refetch } = useModels()
  const { data: features } = useFeatures()
  const { data: systemStatus } = useSystemStatus()
  const downloadMutation = useDownloadModel()
  const deleteMutation = useDeleteModel()

  // Models that suit the detected hardware (RAM / VRAM / GPU backend).
  const recommendedModels = useMemo(() => {
    const recommended = new Set<string>()
    if (!models) return recommended

    const ramGB = systemStatus?.ram_gb ?? 0
    const vramGB = systemStatus?.vram_gb ?? 0
    const hasGpu = vramGB > 0
    const hasSystemInfo = ramGB > 0

    for (const model of models) {
      if (!model.compatible) continue

      const sizeMB = parseSizeMB(model.size)

      // GPU-only models need an accelerator with enough VRAM.
      if (model.requires_cuda) {
        if (!hasGpu) continue
        if (sizeMB * 1.6 > vramGB * 1024) continue
      }
      if (model.requires_metal && !hasGpu) continue

      // Rough system-RAM budget: weights + activations + overhead.
      if (hasSystemInfo && sizeMB * 2.5 > ramGB * 1024 * 0.75) continue

      // On modest GPUs, skip heavyweight models that will not fit comfortably.
      const weight = model.weight.toLowerCase()
      if (weight.includes('heavy') && hasGpu && vramGB < 12) continue

      recommended.add(model.name)
    }

    return recommended
  }, [models, systemStatus])

  const filteredModels = useMemo(() => {
    if (!models) return []
    const query = searchQuery.trim().toLowerCase()

    return models.filter((model) => {
      if (showDownloadedOnly && !model.downloaded) return false
      if (showCompatibleOnly && !model.compatible) return false
      if (aiFilter === 'ai' && !model.is_ai) return false
      if (aiFilter === 'non-ai' && model.is_ai) return false
      if (selectedFeature !== 'all' && !model.features.includes(selectedFeature)) return false
      if (showRecommendedOnly && !recommendedModels.has(model.name)) return false
      if (query) {
        const haystack = `${model.name} ${model.description}`.toLowerCase()
        if (!haystack.includes(query)) return false
      }
      return true
    })
  }, [models, searchQuery, selectedFeature, showDownloadedOnly, showCompatibleOnly, aiFilter, showRecommendedOnly, recommendedModels])

  // Keep the open popup in sync with refreshed list data (e.g. after a delete).
  const activeDetailsModel = useMemo(() => {
    if (!detailsModel) return null
    return models?.find(model => model.name === detailsModel.name) ?? detailsModel
  }, [models, detailsModel])

  const featureLabels = useFeatureDisplayNames()

  const featureOptions = useMemo(() => {
    const names = new Set<string>()
    features?.forEach(f => names.add(f.name))
    models?.forEach(m => m.features.forEach(f => names.add(f)))
    return [
      { value: 'all', label: 'All Features' },
      ...[...names].sort().map(name => ({ value: name, label: displayFeature(name) })),
    ]
  }, [features, models, featureLabels])

  const handleDownload = async (modelName: string) => {
    try {
      await downloadMutation.mutateAsync({ name: modelName, force: false })
      refetch()
    } catch (err) {
      console.error('Download failed:', err)
    }
  }

  const handleDelete = async (modelName: string) => {
    if (!confirm(`Delete ${modelName}? This cannot be undone.`)) return
    try {
      await deleteMutation.mutateAsync(modelName)
      refetch()
    } catch (err) {
      console.error('Delete failed:', err)
    }
  }

  const activeFilterCount =
    (showDownloadedOnly ? 1 : 0) +
    (showCompatibleOnly ? 1 : 0) +
    (aiFilter !== 'all' ? 1 : 0) +
    (selectedFeature !== 'all' ? 1 : 0) +
    (showRecommendedOnly ? 1 : 0) +
    (searchQuery.trim() ? 1 : 0)

  const clearFilters = () => {
    setSearchQuery('')
    setSelectedFeature('all')
    setShowDownloadedOnly(false)
    setShowCompatibleOnly(false)
    setAiFilter('all')
    setShowRecommendedOnly(false)
  }

  return (
    <div className="page">
      <div className="page-inner">
        {/* Page header */}
        <div className="page-header">
          <div>
            <h1 className="page-title">Models</h1>
            <p className="page-subtitle">Manage and download AI enhancement models</p>
          </div>
          <Button variant="secondary" onClick={() => refetch()} disabled={isLoading}>
            <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
            Refresh
          </Button>
        </div>

        {/* Compact desktop metric bar */}
        <div className="metric-bar">
          <MetricBlock value={systemStatus?.total_models ?? 0} label="Total" icon={<Database className="w-3.5 h-3.5" />} />
          <MetricBlock value={systemStatus?.downloaded ?? 0} label="Downloaded" tone="text-emerald-400" icon={<CheckCircle2 className="w-3.5 h-3.5" />} />
          <MetricBlock value={systemStatus?.available ?? 0} label="Available" tone="text-blue-400" icon={<Download className="w-3.5 h-3.5" />} />
          <MetricBlock value={systemStatus?.cpu_threads ?? 0} label="CPU Threads" icon={<Cpu className="w-3.5 h-3.5" />} />
          <MetricBlock
            value={`${(systemStatus?.ram_gb ?? 0).toFixed(1)}`}
            suffix="GB"
            label="System RAM"
            icon={<HardDrive className="w-3.5 h-3.5" />}
          />
          <MetricBlock
            value={`${(systemStatus?.vram_gb ?? 0).toFixed(1)}`}
            suffix="GB"
            label="GPU VRAM"
            icon={<Gauge className="w-3.5 h-3.5" />}
          />
        </div>

        {/* Filter toolbar */}
        <div className="filter-bar">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-text-muted)] pointer-events-none" />
            <input
              className="input pl-8"
              placeholder="Search models..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <select
            className="select w-full sm:w-52"
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
          >
            {featureOptions.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>

          {/* AI / Non-AI segmented control */}
          <div className="inline-flex items-center shrink-0 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-input)] p-0.5">
            {AI_FILTERS.map(option => (
              <button
                key={option.value}
                type="button"
                onClick={() => setAiFilter(option.value)}
                className={clsx(
                  'px-3 py-1 text-[12px] font-medium rounded-[var(--radius-sm)] transition-colors',
                  aiFilter === option.value
                    ? 'bg-[var(--color-accent)] text-white'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                )}
              >
                {option.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-4 shrink-0">
            <FilterCheckbox
              label="Downloaded"
              checked={showDownloadedOnly}
              onChange={setShowDownloadedOnly}
            />
            <FilterCheckbox
              label="Compatible"
              checked={showCompatibleOnly}
              onChange={setShowCompatibleOnly}
            />
            <Button
              variant={showRecommendedOnly ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setShowRecommendedOnly(!showRecommendedOnly)}
              className="flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Recommended
            </Button>
            {activeFilterCount > 0 && (
              <button
                type="button"
                onClick={clearFilters}
                className="text-[12px] text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] underline"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Result count */}
        {models && (
          <p className="text-[12px] text-[var(--color-text-muted)] -mt-1">
            Showing {filteredModels.length} of {models.length} models
            {showRecommendedOnly && systemStatus && (
              <> · Recommended for {systemStatus.ram_gb.toFixed(0)}GB RAM
                {systemStatus.vram_gb ? ` / ${systemStatus.vram_gb.toFixed(0)}GB VRAM` : ' · CPU only'}</>
            )}
          </p>
        )}

        {/* Grid */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-7 h-7 animate-spin text-[var(--color-accent)]" />
          </div>
        ) : error ? (
          <div className="flex items-center justify-center gap-2 py-20 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <span className="text-sm">Failed to load models: {error.message}</span>
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 py-20 text-[var(--color-text-muted)]">
            <Info className="w-5 h-5" />
            <p className="text-sm">No models found matching your criteria</p>
            {activeFilterCount > 0 && (
              <button type="button" onClick={clearFilters} className="text-[12px] underline hover:text-[var(--color-text-primary)]">
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <div className="model-grid">
            {filteredModels.map((model) => (
              <ModelCard
                key={model.name}
                model={model}
                recommended={recommendedModels.has(model.name)}
                onShowDetails={() => setDetailsModel(model)}
                onDownload={() => handleDownload(model.name)}
                onDelete={() => handleDelete(model.name)}
                downloading={downloadMutation.variables?.name === model.name}
                downloadProgress={downloadMutation.variables?.name === model.name ? downloadMutation.data?.progress_percent : undefined}
                deleting={deleteMutation.variables === model.name}
              />
            ))}
          </div>
        )}
      </div>

      {/* Model details popup */}
      <ModelDetailsDialog
        model={activeDetailsModel}
        onClose={() => setDetailsModel(null)}
        onDownload={() => activeDetailsModel && handleDownload(activeDetailsModel.name)}
        onDelete={() => activeDetailsModel && handleDelete(activeDetailsModel.name)}
        downloading={!!activeDetailsModel && downloadMutation.variables?.name === activeDetailsModel.name}
        downloadProgress={
          activeDetailsModel && downloadMutation.variables?.name === activeDetailsModel.name
            ? downloadMutation.data?.progress_percent
            : undefined
        }
        deleting={!!activeDetailsModel && deleteMutation.variables === activeDetailsModel.name}
      />
    </div>
  )
}

function MetricBlock({
  value, label, suffix, tone = 'text-[var(--color-text-primary)]', icon,
}: {
  value: string | number
  label: string
  suffix?: string
  tone?: string
  icon: React.ReactNode
}) {
  return (
    <div className="metric-block">
      <div className="metric-block-label flex items-center gap-1.5">
        <span className="text-[var(--color-text-muted)]">{icon}</span>
        {label}
      </div>
      <div className={clsx('metric-block-value', tone)}>
        {value}
        {suffix && <span className="ml-1 text-[11px] font-medium text-[var(--color-text-muted)]">{suffix}</span>}
      </div>
    </div>
  )
}

function FilterCheckbox({
  label, checked, onChange,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer select-none">
      <span
        className={clsx(
          'grid place-items-center w-4 h-4 rounded-[4px] border transition-colors duration-[var(--transition-fast)]',
          checked
            ? 'bg-[var(--color-accent)] border-[var(--color-accent)]'
            : 'bg-[var(--color-bg-input)] border-[var(--color-border-strong)]'
        )}
      >
        {checked && <CheckCircle2 className="w-3 h-3 text-white" />}
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
      </span>
      <span className="text-[12px] text-[var(--color-text-secondary)]">{label}</span>
    </label>
  )
}

interface ModelCardProps {
  model: ModelResponse
  recommended: boolean
  onShowDetails: () => void
  onDownload: () => void
  onDelete: () => void
  downloading: boolean
  downloadProgress?: number
  deleting: boolean
}

function ModelCard({
  model, recommended, onShowDetails, onDownload, onDelete, downloading, downloadProgress, deleting,
}: ModelCardProps) {
  const status = statusFor(model)
  const canDownload = !model.downloaded && model.has_download_url && model.compatible && !model.is_builtin
  const canDelete = model.downloaded && !model.is_builtin

  return (
    <article className={clsx('model-card group', !model.compatible && 'opacity-75')}>
      {/* Header */}
      <div className="model-card-head">
        <div className="min-w-0 flex-1">
          <h3 className="model-card-title" title={model.name}>{model.name}</h3>
          <div className="model-card-kind flex items-center gap-2">
            <span>{model.is_ai ? 'AI model' : 'Classical model'}</span>
            {recommended && (
              <span className="inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-accent)]">
                <Sparkles className="w-3 h-3" />
                Recommended
              </span>
            )}
          </div>
        </div>
        <span className={clsx('badge shrink-0', STATUS_BADGE[status])}>
          {STATUS_LABEL[status]}
        </span>
      </div>

      {/* Body */}
      <div className="model-card-body">
        <div className="model-card-meta">
          <span className="chip">
            <HardDrive className="w-3 h-3" />
            {model.size}
          </span>
          <span className="chip">
            <Gauge className="w-3 h-3" />
            {model.weight}
          </span>
        </div>

        <p className="model-card-desc">{model.description}</p>

        {model.features.length > 0 && (
          <div className="model-card-tags">
            {model.features.map((feature) => (
              <span
                key={feature}
                className={clsx(
                  'inline-flex items-center rounded-[3px] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide leading-none',
                  FEATURE_TAG_STYLES[feature] || FALLBACK_TAG_STYLE
                )}
              >
                {displayFeature(feature)}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="model-card-actions">
        {canDownload && (
          <Button
            variant="primary"
            size="sm"
            className="flex-1"
            onClick={onDownload}
            disabled={downloading || deleting}
            loading={downloading}
          >
            {!downloading && <Download className="w-3.5 h-3.5" />}
            {downloading ? 'Downloading' : 'Download'}
          </Button>
        )}
        {canDelete && (
          <Button
            variant="danger"
            size="sm"
            className="flex-1"
            onClick={onDelete}
            disabled={deleting || downloading}
          >
            <Trash2 className="w-3.5 h-3.5" />
            {deleting ? 'Deleting' : 'Delete'}
          </Button>
        )}
        {model.is_builtin && (
          <span className="flex-1 inline-flex items-center justify-center gap-1.5 text-[12px] font-medium text-[var(--color-text-muted)]">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Ready to use
          </span>
        )}
        {!canDownload && !canDelete && !model.is_builtin && !model.compatible && (
          <span className="flex-1 inline-flex items-center justify-center gap-1.5 text-[12px] font-medium text-amber-400/90">
            <AlertCircle className="w-3.5 h-3.5" />
            Not supported
          </span>
        )}

        <button
          type="button"
          onClick={onShowDetails}
          className="btn btn-secondary btn-sm shrink-0"
        >
          Details
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Download progress */}
      {downloading && downloadProgress !== undefined && (
        <div className="px-3.5 py-3 border-t border-[var(--color-border-subtle)]">
          <Progress value={downloadProgress} showLabel label="Downloading model..." />
        </div>
      )}
    </article>
  )
}
