import { useState, useId } from 'react'
import {
  Download, Upload, RefreshCw, AlertCircle, CheckCircle2, Loader2,
  Monitor, Sun, Moon, Settings as SettingsIcon, Zap, Folder,
  DownloadCloud, Palette, Sliders, Cpu, Info, FlaskConical,
} from 'lucide-react'
import { useSettings, useUpdateSettings, useResetSettings, useExportSettings, useImportSettings } from '../hooks/useSettings'
import { useExperimental } from '../hooks/useModels'
import { useUIStore } from '../store'
import { Button, Select, Slider, Toggle, Input, Badge, Dialog } from '../components/ui'
import { clsx } from 'clsx'
import type { SettingsModel, ComputeBackend, OutputFormat } from '../types/api'

const THEME_OPTIONS = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'system', label: 'System', icon: Monitor },
] as const

const OUTPUT_FORMAT_OPTIONS: { value: OutputFormat; label: string }[] = [
  { value: 'mp4', label: 'MP4 (H.264)' },
  { value: 'jpg', label: 'JPG' },
  { value: 'png', label: 'PNG' },
  { value: 'webp', label: 'WebP' },
]

const TABS = [
  { id: 'general', label: 'General', icon: SettingsIcon },
  { id: 'system', label: 'System', icon: Cpu },
  { id: 'gpu', label: 'GPU', icon: Zap },
  { id: 'export', label: 'Export', icon: DownloadCloud },
  { id: 'models', label: 'Models', icon: Folder },
  { id: 'appearance', label: 'Appearance', icon: Palette },
  { id: 'advanced', label: 'Advanced', icon: Sliders },
] as const

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<string>('general')
  const { data: settingsResponse, isLoading, error } = useSettings()
  const updateMutation = useUpdateSettings()
  const resetMutation = useResetSettings()
  const exportMutation = useExportSettings()
  const importMutation = useImportSettings()
  const simpleMode = useUIStore((s) => s.simpleMode)
  const setSimpleMode = useUIStore((s) => s.setSimpleMode)

  if (isLoading) {
    return (
      <div className="flex-1 grid place-items-center">
        <Loader2 className="w-7 h-7 animate-spin text-[var(--color-accent)]" />
      </div>
    )
  }

  if (error || !settingsResponse) {
    return (
      <div className="flex-1 grid place-items-center text-red-400">
        <div className="flex items-center gap-2">
          <AlertCircle className="w-5 h-5" />
          <span className="text-sm">Failed to load settings</span>
        </div>
      </div>
    )
  }

  const settings = settingsResponse.settings

  // The backend lists which compute backends this machine can actually use
  // (models.json-style single source of truth on the server). Only those are
  // selectable; a saved value that is not available here stays visible but
  // disabled, so importing settings from another machine never silently
  // rewrites the user's choice.
  const currentBackend = settings.gpu.compute_backend
  const backends = settingsResponse.available_backends ?? []
  const backendOptions = [
    // Defensive: never leave the select without its current value.
    ...(backends.some((backend) => backend.value === currentBackend)
      ? []
      : [{ value: currentBackend, label: currentBackend, disabled: false }]),
    ...backends
      .filter((backend) => backend.available || backend.value === currentBackend)
      .map((backend) => ({
        value: backend.value,
        label: backend.recommended ? `${backend.label} — detected` : backend.label,
        disabled: !backend.available,
      })),
  ]
  const unavailableBackend = backends.find(
    (backend) => backend.value === currentBackend && !backend.available,
  )

  const handleUpdate = async (updates: Partial<SettingsModel>) => {
    await updateMutation.mutateAsync(updates)
  }

  return (
    <div className="settings-layout">
      {/* Tab rail */}
      <aside className="settings-rail">
        <div className="px-3 pt-2 pb-3">
          <h1 className="text-[13px] font-semibold tracking-tight text-[var(--color-text-primary)]">Settings</h1>
        </div>
        {TABS.map((tab) => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx('rail-tab', activeTab === tab.id && 'rail-tab-active')}
              aria-current={activeTab === tab.id}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {tab.label}
            </button>
          )
        })}
      </aside>

      {/* Content */}
      <div className="settings-content">
        <div className="settings-inner">
          {activeTab === 'general' && (
            <Section title="General" description="Default application behavior">
              <Field label="Default Output Folder">
                <Input
                  value={settings.export.export_dir}
                  onChange={(e) => handleUpdate({ export: { ...settings.export, export_dir: e.target.value } })}
                />
              </Field>
              <Divider />
              <Toggle
                label="Auto-Download Models"
                description="Automatically download missing models"
                checked={settings.models.auto_download_models}
                onChange={(checked) => handleUpdate({ models: { ...settings.models, auto_download_models: checked } })}
              />
              <Divider />
              <Toggle
                label="Simple Mode"
                description="Group related enhancement filters together (Topaz-style). Turn off to control each feature individually."
                checked={simpleMode}
                onChange={setSimpleMode}
              />
            </Section>
          )}

          {activeTab === 'system' && (
            <Section title="System Resources" description="CPU and memory allocation">
              <SliderField
                label="CPU Threads"
                value={settings.system.cpu_threads}
                min={1}
                max={Number(settingsResponse.max_values?.cpuThreads) || 128}
                onChange={(v) => handleUpdate({ system: { ...settings.system, cpu_threads: v } })}
              />
              <SliderField
                label="Tile Size"
                value={settings.system.tile_size}
                min={64}
                max={Number(settingsResponse.max_values?.tileSize) || 1024}
                step={64}
                onChange={(v) => handleUpdate({ system: { ...settings.system, tile_size: v } })}
              />
            </Section>
          )}

          {activeTab === 'gpu' && (
            <Section
              title="GPU Acceleration"
              description="Only compute backends detected on this machine can be selected"
            >
              <Field label="Compute Backend">
                <Select
                  value={settings.gpu.compute_backend}
                  onChange={(e) => handleUpdate({ gpu: { ...settings.gpu, compute_backend: e.target.value as ComputeBackend } })}
                  options={backendOptions}
                />
                {unavailableBackend && (
                  <p className="mt-1.5 text-[12px] text-amber-400/90">
                    {unavailableBackend.label} is not available on this machine
                    {unavailableBackend.reason ? ` (${unavailableBackend.reason})` : ''}.
                    {' '}Auto uses the best detected backend.
                  </p>
                )}
              </Field>
              <Divider />
              <Toggle
                label="FP16 Precision"
                description="Use half-precision for faster GPU processing"
                checked={settings.gpu.fp16_enabled}
                onChange={(checked) => handleUpdate({ gpu: { ...settings.gpu, fp16_enabled: checked } })}
              />
            </Section>
          )}

          {activeTab === 'export' && (
            <Section title="Export" description="Configure output format and quality">
              <Field label="Output Format">
                <Select
                  value={settings.export.output_format}
                  onChange={(e) => handleUpdate({ export: { ...settings.export, output_format: e.target.value as OutputFormat } })}
                  options={OUTPUT_FORMAT_OPTIONS}
                />
              </Field>
              <SliderField
                label="JPEG Quality"
                value={settings.export.jpg_quality}
                min={1}
                max={100}
                onChange={(v) => handleUpdate({ export: { ...settings.export, jpg_quality: v } })}
              />
              <Divider />
              <Toggle
                label="Overwrite Existing Files"
                description="Automatically overwrite files with the same name"
                checked={settings.export.overwrite_existing}
                onChange={(checked) => handleUpdate({ export: { ...settings.export, overwrite_existing: checked } })}
              />
            </Section>
          )}

          {activeTab === 'models' && (
            <Section title="Model Storage" description="Configure the model cache and download location">
              <Field label="Model Directory">
                <Input
                  value={settings.models.model_dir}
                  onChange={(e) => handleUpdate({ models: { ...settings.models, model_dir: e.target.value } })}
                />
              </Field>
            </Section>
          )}

          {activeTab === 'appearance' && (
            <Section title="Appearance" description="Customize the theme, accent color, and interface sizing">
              <div className="grid grid-cols-3 gap-2">
                {THEME_OPTIONS.map((option) => {
                  const Icon = option.icon
                  const active = settings.appearance.theme === option.value
                  return (
                    <button
                      key={option.value}
                      onClick={() => handleUpdate({ appearance: { ...settings.appearance, theme: option.value } })}
                      className={clsx(
                        'flex flex-col items-center gap-2 p-4 rounded-[var(--radius-md)] border transition-colors duration-[var(--transition-fast)]',
                        active
                          ? 'border-[var(--color-accent)] bg-[var(--color-accent-soft)] text-[var(--color-text-primary)]'
                          : 'border-[var(--color-border)] bg-[var(--color-bg-input)] text-[var(--color-text-secondary)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-primary)]'
                      )}
                    >
                      <Icon className="w-5 h-5" />
                      <span className="text-[12px] font-medium">{option.label}</span>
                    </button>
                  )
                })}
              </div>
              <Divider />
              <Field label="Accent Color">
                <div className="flex items-center gap-3">
                  <Input
                    type="color"
                    value={settings.appearance.accent_color}
                    aria-label="Accent color"
                    className="h-9 w-14 cursor-pointer p-1"
                    onChange={(event) => handleUpdate({
                      appearance: { ...settings.appearance, accent_color: event.target.value },
                    })}
                  />
                  <span className="font-mono text-[12px] text-[var(--color-text-secondary)]">
                    {settings.appearance.accent_color}
                  </span>
                </div>
              </Field>
              <Divider />
              <Field label="UI Font Size">
                <Select
                  value={settings.appearance.font_size}
                  onChange={(event) => handleUpdate({
                    appearance: { ...settings.appearance, font_size: event.target.value as SettingsModel['appearance']['font_size'] },
                  })}
                  options={[
                    { value: 'small', label: 'Small' },
                    { value: 'medium', label: 'Medium' },
                    { value: 'large', label: 'Large' },
                  ]}
                />
              </Field>
              <Divider />
              <Field label="UI Density">
                <Select
                  value={settings.appearance.density}
                  onChange={(event) => handleUpdate({
                    appearance: { ...settings.appearance, density: event.target.value as SettingsModel['appearance']['density'] },
                  })}
                  options={[
                    { value: 'compact', label: 'Compact' },
                    { value: 'comfortable', label: 'Comfortable' },
                    { value: 'spacious', label: 'Spacious' },
                  ]}
                />
              </Field>
            </Section>
          )}

          {activeTab === 'advanced' && (
            <Section title="Advanced" description="Developer and debugging options">
              <Toggle
                label="Debug Logging"
                description="Enable detailed debug logs"
                checked={settings.advanced.debug_logging}
                onChange={(checked) => handleUpdate({ advanced: { ...settings.advanced, debug_logging: checked } })}
              />
              <Toggle
                label="Show Metrics"
                description="Show processing metrics and performance info"
                checked={settings.advanced.show_metrics}
                onChange={(checked) => handleUpdate({ advanced: { ...settings.advanced, show_metrics: checked } })}
              />
              <Toggle
                label="Experimental Features"
                description="Unlock features and models the catalog marks as experimental"
                checked={settings.advanced.experimental_features}
                labelExtra={<ExperimentalInfoButton />}
                onChange={(checked) => handleUpdate({ advanced: { ...settings.advanced, experimental_features: checked } })}
              />

              <Divider />

              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" size="sm" onClick={() => exportMutation.mutate()} disabled={exportMutation.isPending}>
                  <Download className="w-3.5 h-3.5" />
                  Export Settings
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={importMutation.isPending}
                  onClick={() => {
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.accept = '.json'
                    input.onchange = (e) => {
                      const file = (e.target as HTMLInputElement).files?.[0]
                      if (file) importMutation.mutate(file)
                    }
                    input.click()
                  }}
                >
                  <Upload className="w-3.5 h-3.5" />
                  Import Settings
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-red-400 hover:text-red-300"
                  disabled={resetMutation.isPending}
                  onClick={() => {
                    if (confirm('Reset all settings to defaults? This cannot be undone.')) {
                      resetMutation.mutate()
                    }
                  }}
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Reset to Defaults
                </Button>
              </div>

              {updateMutation.isSuccess && (
                <div className="flex items-center gap-2 text-emerald-400 text-[12px]">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Settings saved
                </div>
              )}
              {updateMutation.isError && (
                <div className="flex items-center gap-2 text-red-400 text-[12px]">
                  <AlertCircle className="w-3.5 h-3.5" />
                  Failed to save settings
                </div>
              )}
            </Section>
          )}
        </div>
      </div>
    </div>
  )
}

function Section({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="panel-section mb-4">
      <div className="panel-section-head">
        <h2 className="panel-section-title">{title}</h2>
        {description && <p className="panel-section-desc">{description}</p>}
      </div>
      <div className="panel-section-body">{children}</div>
    </section>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] mb-1.5">
        {label}
      </label>
      {children}
    </div>
  )
}

function SliderField({
  label, value, min, max, step = 1, onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  const id = useId()
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label htmlFor={id} className="text-[12px] font-medium text-[var(--color-text-secondary)]">{label}</label>
        <span className="text-[12px] text-[var(--color-text-primary)] font-mono">{value}</span>
      </div>
      <Slider
        id={id}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(val) => onChange(Number(val))}
      />
    </div>
  )
}

function Divider() {
  return <div className="divider" />
}

/**
 * Little "i" next to the Experimental Features toggle. Opening it lists
 * whatever the model catalog marks with `"experimental": true` — features and
 * models alike — so the toggle's effect is discoverable. The list is fetched
 * from the backend (single source of truth is models.json), never hardcoded.
 */
function ExperimentalInfoButton() {
  const [open, setOpen] = useState(false)
  const { data, isLoading } = useExperimental(open)

  const items = data?.items ?? []
  const enabled = data?.enabled ?? false

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Which features are experimental?"
        title="Which features are experimental?"
        className="grid place-items-center w-4 h-4 shrink-0 rounded-full text-[var(--color-text-muted)] transition-colors duration-[var(--transition-fast)] hover:text-[var(--color-text-primary)] hover:bg-white/[0.06]"
      >
        <Info className="w-3.5 h-3.5" />
      </button>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title="Experimental Features"
        description="Everything the model catalog marks as experimental."
        icon={<FlaskConical className="w-4 h-4" />}
        size="max-w-[520px]"
        footer={
          <Button variant="secondary" size="sm" onClick={() => setOpen(false)}>
            Close
          </Button>
        }
      >
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-[var(--color-accent)]" />
          </div>
        ) : items.length === 0 ? (
          <div className="flex flex-col items-center gap-2 py-8 text-center text-[var(--color-text-muted)]">
            <FlaskConical className="w-5 h-5" />
            <p className="text-[13px]">Nothing is marked experimental in this build.</p>
            <p className="text-[11px] max-w-[340px] leading-relaxed">
              Add{' '}
              <code className="font-mono text-[var(--color-text-secondary)]">"experimental": true</code>{' '}
              to a feature or model in{' '}
              <code className="font-mono text-[var(--color-text-secondary)]">models.json</code>{' '}
              and it will be listed here.
            </p>
          </div>
        ) : (
          <>
            <p className="text-[12px] text-[var(--color-text-muted)] mb-3">
              {enabled
                ? 'Experimental items are enabled, so they appear in the Enhance page and can be processed.'
                : 'Experimental items are hidden from the Enhance page and skipped during processing until you turn the toggle on.'}
            </p>
            <ul className="flex flex-col gap-2">
              {items.map((item) => (
                <li
                  key={`${item.kind}-${item.name}`}
                  className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-input)] px-3 py-2.5"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[13px] font-medium text-[var(--color-text-primary)]">
                      {item.displayName}
                    </span>
                    <Badge variant="outline" size="sm">
                      {item.kind === 'feature' ? 'Feature' : 'Model'}
                    </Badge>
                  </div>
                  {item.kind === 'model' && item.features.length > 0 && (
                    <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5">
                      Used by {item.features.join(', ')}
                    </p>
                  )}
                  {item.description && (
                    <p className="text-[12px] text-[var(--color-text-secondary)] mt-1">{item.description}</p>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
      </Dialog>
    </>
  )
}
