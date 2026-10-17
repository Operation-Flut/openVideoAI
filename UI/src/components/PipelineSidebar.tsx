import { useMemo, useState, type ComponentType } from 'react'
import {
  ChevronDown, Sparkles, Layers, Download, Play, X,
  Maximize2, Film, Palette, Wand2,
} from 'lucide-react'
import { usePipeline } from '../hooks/usePipeline'
import { useModelStore, useUIStore } from '../store'
import { Badge, Button, Toggle, Slider, Select, Input } from '../components/ui'
import { clsx } from 'clsx'
import type { ModelResponse } from '../types/api'
import { displayFeature, useFeatureDisplayNames } from '../utils/displayNames'
import { useFeatures, useFeatureGroups, useModelRecommendations } from '../hooks/useModels'
import type { FeatureGroupDefinition } from '../types/api'

type IconType = ComponentType<{ className?: string }>

/**
 * Icons referenced by name from the catalog's Simple-Mode groups. The grouping,
 * labels and descriptions all come from models.json; only the icon component
 * lookup lives here (icons are a presentation concern).
 */
const GROUP_ICONS: Record<string, IconType> = {
  Maximize2, Film, Palette, Layers, Wand2, Sparkles,
}

function iconForFeature(name: string): IconType {
  const n = name.toLowerCase()
  if (n.includes('upscale') || n.includes('resolution') || n.includes('restoration') || n.includes('cugan')) return Maximize2
  if (n.includes('interpol') || n.includes('motion') || n.includes('smo') || n.includes('stabil')) return Film
  if (n.includes('color') || n.includes('light') || n.includes('hdr')) return Palette
  if (n.includes('encod')) return Layers
  return Sparkles
}

interface Section {
  key: string
  label: string
  description: string
  icon: IconType
  features: string[]
}

function buildSections(
  availableFeatures: Record<string, { description?: string }>,
  simpleMode: boolean,
  groups: FeatureGroupDefinition[],
): Section[] {
  const implemented = (name: string) => Boolean(availableFeatures[name])
  const allNames = Object.keys(availableFeatures)

  if (simpleMode && groups.length > 0) {
    const grouped: Section[] = groups
      .map(g => ({
        key: g.key,
        label: g.label,
        description: g.description,
        icon: GROUP_ICONS[g.icon] || iconForFeature(g.label),
        features: g.features.filter(implemented),
      }))
      .filter(s => s.features.length > 0)

    const covered = new Set(groups.flatMap(g => g.features))
    const extras = allNames.filter(f => !covered.has(f))
    if (extras.length > 0) {
      grouped.push({
        key: 'other',
        label: 'Other',
        description: 'Additional filters',
        icon: Layers,
        features: extras,
      })
    }
    return grouped
  }

  // Advanced Mode: one card per backend feature.
  return allNames.map(name => ({
    key: name,
    label: displayFeature(name),
    description: availableFeatures[name]?.description || '',
    icon: iconForFeature(name),
    features: [name],
  }))
}

function prettyLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export function PipelineSidebar() {
  const models = useModelStore(state => state.models)
  const simpleMode = useUIStore(state => state.simpleMode)
  const {
    steps,
    availableFeatures,
    outputFormat,
    quality,
    selectedFiles,
    addStep,
    removeStep,
    updateStep,
    getAvailableModels,
    getDefaultModel,
    getParametersSchema,
    stepCount,
    startProcessing,
    previewCurrentFile,
    isExporting,
    setOutputFormat,
    setQuality,
  } = usePipeline()

  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [activeFeature, setActiveFeature] = useState<Record<string, string>>({})

  // Feature labels and Simple-Mode groups both come from the models.json catalog.
  useFeatures()
  const featureLabels = useFeatureDisplayNames()
  const { data: featureGroups } = useFeatureGroups()

  // Smart Select: the backend ranks the models for the file being processed
  // (media, memory budget, what is installed). The query is disabled while the
  // setting is off, so nothing here changes unless it is enabled.
  const media: 'image' | 'video' | null = selectedFiles?.[0]
    ? (selectedFiles[0].type.startsWith('video/') ? 'video' : 'image')
    : null
  const { smartSelect, recommendations } = useModelRecommendations(media)

  const sections = useMemo(
    () => buildSections(availableFeatures, simpleMode, featureGroups ?? []),
    [availableFeatures, simpleMode, featureGroups, featureLabels],
  )

  const catalogModelFor = (key: string): ModelResponse | undefined =>
    models.find(m => m.name === key)

  const modelsFor = (feature: string): ModelResponse[] => {
    return getAvailableModels(feature)
      .map(catalogModelFor)
      .filter((m): m is ModelResponse => Boolean(m))
  }

  const recommendationFor = (feature: string) => recommendations?.[feature]

  const pickModel = (feature: string): string | null => {
    // Work with pipeline keys (the value the backend expects) but rank them by
    // their catalog (download/compatibility) status.
    const choices = getAvailableModels(feature).map(key => ({ key, catalog: catalogModelFor(key) }))
    const downloaded = choices.find(c => c.catalog?.downloaded && c.catalog?.compatible)
    if (downloaded) return downloaded.key
    const compatible = choices.find(c => c.catalog?.compatible)
    if (compatible) return compatible.key
    // Pseudo/uncatalogued models (e.g. Reencoding: ffmpeg/handbrake) fall back
    // to the first pipeline key so the section can still be enabled.
    return choices[0]?.key ?? null
  }

  const initialModel = (feature: string): string | null => {
    // Smart Select wins when it has an answer the picker can offer.
    const recommended = recommendationFor(feature)?.model
    if (recommended && getAvailableModels(feature).includes(recommended)) return recommended
    // Smart Select is on but has not answered yet: prefer the installed and
    // compatible model over a catalog default that may not be downloaded.
    if (smartSelect && !recommendations) return pickModel(feature)
    const preferred = getDefaultModel(feature)
    if (preferred && getAvailableModels(feature).includes(preferred)) return preferred
    return pickModel(feature)
  }

  const featureFor = (section: Section): string =>
    activeFeature[section.key] || section.features[0]

  const stepsFor = (section: Section) =>
    steps.filter(s => section.features.includes(s.feature))

  const toggleSection = (section: Section) => {
    const existing = stepsFor(section)
    if (existing.length > 0) {
      existing.forEach(s => removeStep(s.feature))
      setExpanded(prev => {
        const next = new Set(prev)
        next.delete(section.key)
        return next
      })
      return
    }
    const feature = featureFor(section)
    const model = initialModel(feature)
    if (!feature || !model) return
    addStep({ feature, model, parameters: {} })
    setExpanded(prev => new Set(prev).add(section.key))
  }

  const changeSectionFeature = (section: Section, feature: string) => {
    setActiveFeature(prev => ({ ...prev, [section.key]: feature }))
    // If the section is currently active, swap the underlying step.
    const existing = stepsFor(section)
    if (existing.length > 0) {
      existing.forEach(s => removeStep(s.feature))
      const model = initialModel(feature)
      if (model) addStep({ feature, model, parameters: {} })
    }
  }

  const handleModelChange = (feature: string, modelName: string) => {
    // Ignore the placeholder "Select model" option so an active step never
    // ends up with an empty model name.
    if (!modelName) return
    if (steps.some(s => s.feature === feature)) {
      updateStep(feature, { model: modelName })
    } else {
      addStep({ feature, model: modelName, parameters: {} })
    }
  }

  const handleParamChange = (feature: string, paramKey: string, value: unknown) => {
    if (steps.some(s => s.feature === feature)) {
      updateStep(feature, { parameters: { [paramKey]: value } })
    }
  }

  const handleExport = async () => {
    await startProcessing()
  }

  const [busy, setBusy] = useState(false)

  const handlePreview = async () => {
    setBusy(true)
    await previewCurrentFile(2)
    setBusy(false)
  }

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--color-bg-panel)]">
      {/* Fixed panel toolbar */}
      <div className="panel-toolbar justify-between border-b border-[var(--color-border)]">
        <h2 className="flex items-center gap-2 text-[13px] font-semibold text-[var(--color-text-primary)]">
          <Layers className="w-4 h-4 text-[var(--color-text-muted)]" />
          Enhancement
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
            {simpleMode ? 'Simple' : 'Advanced'}
          </span>
          <Badge variant={stepCount > 0 ? 'success' : 'outline'} size="sm">
            {stepCount} Filter{stepCount !== 1 ? 's' : ''} Active
          </Badge>
        </div>
      </div>

      {/* Independently scrolling pipeline body */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
        {sections.length === 0 && (
          <div className="module-card p-8 text-center">
            <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-[var(--color-bg-surface)] flex items-center justify-center">
              <Sparkles className="w-7 h-7 text-[var(--color-border-strong)]" />
            </div>
            <h3 className="text-[14px] font-medium text-[var(--color-text-secondary)] mb-1.5">No Filters Available</h3>
            <p className="text-[12px] text-[var(--color-text-muted)]">
              No processing pipelines are registered. Check that the backend started correctly.
            </p>
          </div>
        )}

        {sections.map((section) => {
          const Icon = section.icon
          const feature = featureFor(section)
          const availableModels = getAvailableModels(feature)
          const selectableModels = modelsFor(feature)
          const downloadedCount = selectableModels.filter(m => m.downloaded).length
          const step = steps.find(s => s.feature === feature)
          const isEnabled = !!step
          const isExpanded = expanded.has(section.key)
          const recommendation = recommendationFor(feature)
          const smartPicked = Boolean(recommendation && step?.model === recommendation.model)
          const schema = getParametersSchema(feature)
          const properties: Record<string, any> = schema?.properties || {}
          const paramEntries = Object.entries(properties)

          const modelOptions = getAvailableModels(feature).map((name) => {
            const catalogModel = models.find(m => m.name === name)
            const suffix = catalogModel ? (catalogModel.downloaded ? ' ✓' : ' ⬇') : ''
            return { value: name, label: `${name}${suffix}` }
          })

          const featureOptions = section.features.map(name => ({ value: name, label: displayFeature(name) }))

          return (
            <div key={section.key} className="filter-module-card">
              <div className="module-card">
                {/* Module header */}
                <div className="module-header">
                  <div className="flex items-start gap-3 min-w-0 flex-1">
                    <Icon className={clsx('w-5 h-5 shrink-0 mt-0.5', isEnabled ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)]')} />
                    <div className="min-w-0 flex-1">
                      <span className={clsx('font-medium text-[13px] truncate block', isEnabled ? 'text-[var(--color-text-primary)]' : 'text-[var(--color-text-secondary)]')}>
                        {section.label}
                      </span>
                      <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5">{section.description}</p>
                      {isEnabled && step && (
                        <span className="text-[11px] text-[var(--color-accent)] font-medium mt-1 inline-block">
                          {section.features.length > 1
                            ? `${displayFeature(step.feature)} · ${step.model}`
                            : step.model}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {isEnabled && (
                      <Badge variant="success" size="sm" className="mr-1">ENABLED</Badge>
                    )}
                    {isEnabled && smartPicked && recommendation && (
                      <span title={`Smart Select: ${recommendation.reason}`} className="mr-1">
                        <Badge variant="info" size="sm">SMART</Badge>
                      </span>
                    )}
                    {!isEnabled && downloadedCount > 0 && (
                      <Badge variant="outline" size="sm" className="mr-1">AVAILABLE</Badge>
                    )}
                    {!isEnabled && downloadedCount === 0 && availableModels.length > 0 && (
                      <Badge variant="error" size="sm" className="mr-1">NO MODEL</Badge>
                    )}
                    {!isEnabled && availableModels.length === 0 && (
                      <Badge variant="outline" size="sm" className="mr-1">NO MODELS</Badge>
                    )}
                    <div className="flex items-center gap-2">
                      {isEnabled && isExpanded && (
                        <button
                          type="button"
                          onClick={() => setExpanded(prev => { const n = new Set(prev); n.delete(section.key); return n })}
                          className="p-1 rounded-[var(--radius-sm)] hover:bg-[var(--color-bg-surface-hover)] text-[var(--color-text-muted)] transition-colors"
                          aria-label="Collapse"
                        >
                          <ChevronDown className="w-4 h-4 rotate-180" />
                        </button>
                      )}
                      <Toggle
                        checked={isEnabled}
                        onChange={() => toggleSection(section)}
                        size="sm"
                      />
                      {isEnabled && (
                        <button
                          type="button"
                          onClick={() => toggleSection(section)}
                          className="p-1 rounded-[var(--radius-sm)] hover:bg-[var(--color-bg-surface-hover)] text-[var(--color-text-muted)] transition-colors"
                          aria-label={`Remove ${section.label}`}
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Expanded parameter panel */}
                {isEnabled && isExpanded && (
                  <div className="module-content space-y-4 border-t border-[var(--color-border)] pt-3 animate-slide-down bg-[var(--color-bg-surface)] rounded-b-[var(--radius-lg)] px-3 pb-3">
                    {/* Feature picker (Simple Mode groups several features) */}
                    {section.features.length > 1 && (
                      <div className="space-y-2">
                        <label className="block text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)]">Filter</label>
                        <Select
                          value={feature}
                          onChange={(e) => changeSectionFeature(section, e.target.value)}
                          options={featureOptions}
                        />
                      </div>
                    )}

                    {/* Model selection */}
                    <div className="space-y-2">
                      <label className="flex items-center justify-between text-[11px]">
                        <span className="font-medium uppercase tracking-wide text-[var(--color-text-muted)]">Model</span>
                      </label>
                      <Select
                        value={step?.model || ''}
                        onChange={(e) => handleModelChange(feature, e.target.value)}
                        options={modelOptions}
                        placeholder="Select model"
                        className="model-selector-compact"
                      />
                      {smartPicked && recommendation && (
                        <p className="text-[11px] text-[var(--color-text-muted)]">
                          Smart Select: {recommendation.reason}
                        </p>
                      )}
                      {recommendation?.hint && (
                        <p className="text-[11px] text-[var(--color-warning)] flex items-start gap-1">
                          <span>⚠</span>
                          <span>
                            {recommendation.hint}{' '}
                            <a href="/models" className="underline hover:text-[var(--color-accent)]">Models page</a>
                          </span>
                        </p>
                      )}
                      {downloadedCount === 0 && availableModels.length > 0 && (
                        <p className="text-[11px] text-[var(--color-warning)] flex items-center gap-1">
                          <span>⚠</span> No models downloaded.{' '}
                          <a href="/models" className="underline hover:text-[var(--color-accent)]">Download from Models page</a>
                        </p>
                      )}
                    </div>

                    {/* Parameter controls (from backend JSON schema) */}
                    {paramEntries.length > 0 && (
                      <div className="space-y-3">
                        <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)]">Parameters</span>
                        {paramEntries.map(([paramKey, param]) => (
                          <ParamControl
                            key={paramKey}
                            paramKey={paramKey}
                            param={param}
                            value={step?.parameters?.[paramKey] ?? param?.default}
                            onChange={(value) => handleParamChange(feature, paramKey, value)}
                          />
                        ))}
                      </div>
                    )}

                  </div>
                )}

                {!isEnabled && downloadedCount > 0 && (
                  <div className="module-cta p-3 text-center border-t border-[var(--color-border)]">
                    <Button variant="primary" size="sm" className="w-full" onClick={() => toggleSection(section)}>
                      Add {section.label}
                    </Button>
                  </div>
                )}

                {!isEnabled && downloadedCount === 0 && availableModels.length > 0 && (
                  <div className="module-cta p-3 text-center text-[11px] text-[var(--color-warning)] border-t border-[var(--color-border)]">
                    No models downloaded.{' '}
                    <a href="/models" className="underline hover:text-[var(--color-accent)]">Download from Models page</a>
                  </div>
                )}
                {!isEnabled && availableModels.length === 0 && (
                  <div className="module-cta p-3 text-center text-[11px] text-[var(--color-text-muted)] border-t border-[var(--color-border)]">
                    No models available for this filter.
                  </div>
                )}
              </div>
            </div>
          )
        })}

        {/* Output settings - always visible when steps exist */}
        {stepCount > 0 && (
          <div className="filter-module">
            <div className="module-header">
              <div className="flex items-center gap-3">
                <Layers className="w-4 h-4 text-[var(--color-text-muted)]" />
                <span className="font-medium text-[13px] text-[var(--color-text-primary)]">Output</span>
              </div>
            </div>
            <div className="module-content space-y-3 pt-3 border-t border-[var(--color-border)] animate-slide-down">
              <div>
                <label className="block text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] mb-1.5">Output Format</label>
                <Select
                  value={outputFormat}
                  onChange={(e) => setOutputFormat(e.target.value as typeof outputFormat)}
                  options={[
                    { value: 'source', label: 'Same as source' },
                    { value: 'jpg', label: 'JPG' },
                    { value: 'png', label: 'PNG (Lossless)' },
                    { value: 'webp', label: 'WebP' },
                    { value: 'tiff', label: 'TIFF' },
                    { value: 'mp4', label: 'MP4 (Video)' },
                  ]}
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] mb-1.5">Quality</label>
                <Select
                  value={quality}
                  onChange={(e) => setQuality(e.target.value)}
                  options={[
                    { value: 'high', label: 'High Quality' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'low', label: 'Low (Smaller)' },
                  ]}
                />
              </div>
            </div>
          </div>
        )}

        {/* Empty state */}
        {stepCount === 0 && sections.length > 0 && (
          <div className="filter-module-card">
            <div className="module-card p-8 text-center">
              <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-[var(--color-bg-surface)] flex items-center justify-center">
                <Sparkles className="w-7 h-7 text-[var(--color-border-strong)]" />
              </div>
              <h3 className="text-[14px] font-medium text-[var(--color-text-secondary)] mb-1.5">No Filters Active</h3>
              <p className="text-[12px] text-[var(--color-text-muted)] mb-4">
                Toggle a filter above to build your enhancement pipeline
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Sticky action footer */}
      {stepCount > 0 && (
        <div className="shrink-0 p-3 space-y-2 border-t border-[var(--color-border)] bg-[var(--color-bg-panel)]">
          <div className="flex items-center justify-between text-[11px] text-[var(--color-text-muted)]">
            <span>Ready to process</span>
            <span className="font-mono text-[var(--color-text-primary)]">{stepCount} Filter{stepCount !== 1 ? 's' : ''}</span>
          </div>
          <Button variant="secondary" size="sm" className="w-full" onClick={handlePreview} disabled={isExporting || busy} loading={busy}>
            <Play className="w-3.5 h-3.5" />
            Generate Preview (2s)
          </Button>
          <Button
            variant="primary"
            size="sm"
            className="w-full"
            onClick={handleExport}
            disabled={isExporting}
            loading={isExporting}
          >
            <Download className="w-3.5 h-3.5" />
            Export
          </Button>
        </div>
      )}
    </div>
  )
}

function ParamControl({
  paramKey, param, value, onChange,
}: {
  paramKey: string
  param: any
  value: any
  onChange: (value: unknown) => void
}) {
  const label = prettyLabel(paramKey)

  if (Array.isArray(param?.enum) && param.enum.length > 0) {
    return (
      <div>
        <label className="block text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] mb-1.5">
          {label}
        </label>
        <Select
          value={String(value ?? param.default ?? param.enum[0])}
          onChange={(e) => {
            const raw = e.target.value
            const asNumber = Number(raw)
            onChange(param.type === 'integer' || param.type === 'number' ? asNumber : raw)
          }}
          options={param.enum.map((v: unknown) => ({ value: String(v), label: String(v) }))}
        />
      </div>
    )
  }

  if (param?.type === 'boolean') {
    return (
      <Toggle
        label={label}
        checked={Boolean(value ?? param.default ?? false)}
        onChange={(checked) => onChange(checked)}
        size="sm"
      />
    )
  }

  if (param?.type === 'number' || param?.type === 'integer') {
    const isInt = param.type === 'integer'
    const min = typeof param.minimum === 'number' ? param.minimum : 0
    const max = typeof param.maximum === 'number' ? param.maximum : min + 100
    const stepSize = typeof param.multipleOf === 'number' ? param.multipleOf : (isInt ? 1 : 0.05)
    const num = Number(value ?? param.default ?? min)

    return (
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-[var(--color-text-secondary)]">{label}</span>
          <input
            type="number"
            className="w-20 px-2 py-1 text-[11px] font-mono text-[var(--color-text-primary)] bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-[var(--radius-sm)] text-right focus:outline-none focus:border-[var(--color-accent)]"
            value={num}
            min={min}
            max={max}
            step={stepSize}
            onChange={(e) => {
              const parsed = isInt ? parseInt(e.target.value, 10) : parseFloat(e.target.value)
              onChange(Number.isNaN(parsed) ? 0 : parsed)
            }}
          />
        </div>
        <Slider
          min={min}
          max={max}
          step={stepSize}
          value={num}
          showValue={false}
          onChange={(v) => onChange(v)}
        />
      </div>
    )
  }

  // Fallback: text input for string params.
  return (
    <div>
      <label className="block text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] mb-1.5">
        {label}
      </label>
      <Input
        value={String(value ?? param?.default ?? '')}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  )
}
