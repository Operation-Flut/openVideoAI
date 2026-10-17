import { useState, useEffect, useCallback, useMemo } from 'react'
import { X, FolderOpen, FileText, ChevronDown, ChevronUp, Save, AlertTriangle, HelpCircle } from 'lucide-react'
import { Dialog, Button, Select, Input, Slider } from '../components/ui'
import { api } from '../services/api'
import { useSettings, useUpdateSettings } from '../hooks/useSettings'
import { clsx } from 'clsx'
import type { ExportOptions, ExportOptionsResponse, ExportField } from '../types/api'

interface ExportDialogProps {
  open: boolean
  onClose: () => void
  onStartExport: (options: ExportOptions) => void
  selectedFile: File | null
  steps: Array<{ feature: string; model: string; parameters: Record<string, any> }>
}

function SectionTitle({ children, help }: { children: React.ReactNode; help?: string }) {
  return (
    <div className="flex items-start gap-1.5 mb-2">
      <label className="text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] flex-1">
        {children}
      </label>
      {help && (
        <span className="w-3.5 h-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5 cursor-help" title={help}>
          <HelpCircle className="w-3.5 h-3.5" />
        </span>
      )}
    </div>
  )
}

function DisabledBanner({ reason }: { reason: string }) {
  return (
    <div className="flex items-center gap-1.5 p-2 bg-[var(--color-warning-soft)] border border-[var(--color-warning)]/30 rounded-[var(--radius-sm)] text-[11px] text-[var(--color-warning)]">
      <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
      <span>{reason}</span>
    </div>
  )
}

function SegmentedControl({
  value,
  onChange,
  options,
  disabled,
}: {
  value: string
  onChange: (value: string) => void
  options: Array<{ value: string; label: string; disabled?: boolean }>
  disabled?: boolean
}) {
  return (
    <div className="flex bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-[var(--radius-sm)] p-0.5" role="group">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => !opt.disabled && !disabled && onChange(opt.value)}
          disabled={disabled || opt.disabled}
          className={clsx(
            'px-3 py-1.5 text-[11px] font-medium rounded-[var(--radius-xs)] transition-colors',
            value === opt.value
              ? 'bg-[var(--color-accent)] text-white shadow-sm'
              : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-surface)]',
            (disabled || opt.disabled) && 'opacity-40 cursor-not-allowed'
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

function SelectField({
  label,
  value,
  onChange,
  options,
  disabled,
  help,
  placeholder,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: Array<{ value: string; label: string; disabled?: boolean; reason?: string; help?: string; codecs?: string[]; containers?: string[] }>
  disabled?: boolean
  help?: string
  placeholder?: string
}) {
  const visibleOptions = options.filter((o) => !o.disabled)
  return (
    <div className="space-y-1">
      <SectionTitle>{label}{help && <span className="w-3.5 h-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5 cursor-help" title={help}><HelpCircle className="w-3.5 h-3.5" /></span>}</SectionTitle>
      {disabled && (
        <DisabledBanner reason={options.find((o) => o.value === value)?.reason || "Not available on this machine."} />
      )}
      <Select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        options={visibleOptions.length > 0 ? visibleOptions.map((o) => ({ value: o.value, label: o.label })) : [{ value: '', label: 'None available' }]}
        placeholder={placeholder}
      />
    </div>
  )
}

function TextField({
  label,
  value,
  onChange,
  disabled,
  help,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  disabled?: boolean
  help?: string
}) {
  return (
    <div className="space-y-1">
      <SectionTitle>{label}{help && <span className="w-3.5 h-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5 cursor-help" title={help}><HelpCircle className="w-3.5 h-3.5" /></span>}</SectionTitle>
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={help}
      />
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  unit,
  disabled,
  help,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  unit?: string
  disabled?: boolean
  help?: string
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <SectionTitle>{label}{help && <span className="w-3.5 h-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5 cursor-help" title={help}><HelpCircle className="w-3.5 h-3.5" /></span>}</SectionTitle>
      </div>
      <div className="flex items-center gap-2">
        <input
          type="number"
          className="w-20 px-2 py-1 text-[11px] font-mono text-[var(--color-text-primary)] bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-[var(--radius-sm)] text-right focus:outline-none focus:border-[var(--color-accent)]"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            const parsed = parseFloat(e.target.value)
            onChange(Number.isNaN(parsed) ? min : parsed)
          }}
          disabled={disabled}
        />
        {unit && <span className="text-[11px] text-[var(--color-text-muted)]">{unit}</span>}
        <Slider
          min={min}
          max={max}
          step={step}
          value={value}
          showValue={false}
          onChange={(v) => onChange(v)}
          className="flex-1"
        />
      </div>
    </div>
  )
}

function BrowseField({
  label,
  value,
  onChange,
  browse,
  filters,
  choices,
  choices_dir,
  disabled,
  help,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  browse: 'file' | 'folder'
  filters?: string[]
  choices?: Array<{ value: string; label: string }>
  choices_dir?: string
  disabled?: boolean
  help?: string
}) {
  const [showDropdown, setShowDropdown] = useState(false)

  const handleBrowse = async () => {
    try {
      const res = await api.systemBrowse(browse, filters, choices_dir)
      if (res.ok && res.path) {
        onChange(res.path)
      } else if (!res.ok) {
        alert(res.reason)
      }
    } catch (err) {
      console.error('Browse failed:', err)
    }
  }

  return (
    <div className="space-y-1">
      <SectionTitle>{label}{help && <span className="w-3.5 h-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5 cursor-help" title={help}><HelpCircle className="w-3.5 h-3.5" /></span>}</SectionTitle>
      {disabled && <DisabledBanner reason="Not available on this machine." />}
      <div className="flex items-center gap-2">
        <Input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="Click Browse or type a path"
          className="flex-1"
        />
        <Button
          variant="secondary"
          size="sm"
          onClick={handleBrowse}
          disabled={disabled}
        >
          <FolderOpen className="w-3.5 h-3.5" />
        </Button>
      </div>
      {(choices?.length || 0) > 0 && (
        <div className="relative">
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start"
            onClick={() => setShowDropdown(!showDropdown)}
            disabled={disabled}
          >
            <FileText className="w-3.5 h-3.5" />
            <span className="text-[11px]">Choose from {choices_dir}</span>
            {showDropdown ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </Button>
          {showDropdown && (
            <div className="absolute bottom-full left-0 right-0 mb-1 bg-[var(--color-bg-panel)] border border-[var(--color-border)] rounded-[var(--radius-md)] shadow-lg overflow-hidden z-10 max-h-60 overflow-y-auto">
              {choices?.map((choice) => (
                <button
                  key={choice.value}
                  type="button"
                  onClick={() => { onChange(choice.value); setShowDropdown(false); }}
                  className={clsx(
                    'w-full px-3 py-2 text-left text-[11px] hover:bg-[var(--color-bg-surface-hover)]',
                    value === choice.value && 'bg-[var(--color-accent-soft)] text-[var(--color-accent)]'
                  )}
                >
                  {choice.label}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FieldRenderer({
  field,
  value,
  onChange,
  schema,
  disabled,
}: {
  field: ExportField
  value: any
  onChange: (value: any) => void
  schema: ExportOptionsResponse
  disabled?: boolean
}) {
  const fieldDisabled = disabled || field.disabled

  // For profile, narrow the option set based on the selected codec
  if (field.key === 'profile' && field.option_sets) {
    const codec = schema.defaults?.codec || value?.codec
    const options = field.option_sets[codec] || [{ value: '', label: 'Automatic' }]
    return (
      <SelectField
        label={field.label}
        value={value || ''}
        onChange={onChange}
        options={options}
        disabled={fieldDisabled}
        help={field.help}
      />
    )
  }

  // For codec, apply depends_on (only show for video output)
  const dependsOn = field.depends_on
  const shouldShow = !dependsOn || Object.entries(dependsOn).every(([key, allowed]) =>
    schema.defaults[key] ? allowed.includes(schema.defaults[key]) : true
  )

  if (!shouldShow) return null

  switch (field.control) {
    case 'text':
      return <TextField label={field.label} value={value || ''} onChange={onChange} disabled={fieldDisabled} help={field.help} />
    case 'select':
      return (
        <SelectField
          label={field.label}
          value={value || ''}
          onChange={onChange}
          options={field.options || []}
          disabled={fieldDisabled}
          help={field.help}
        />
      )
    case 'segmented':
      return (
        <div className="space-y-1">
          <SectionTitle>{field.label}{field.help && <span className="w-3.5 h-3.5 text-[var(--color-text-muted)] flex-shrink-0 mt-0.5 cursor-help" title={field.help}><HelpCircle className="w-3.5 h-3.5" /></span>}</SectionTitle>
          {fieldDisabled && <DisabledBanner reason={field.reason || "Not available on this machine."} />}
          <SegmentedControl
            value={value || ''}
            onChange={onChange}
            options={field.options || []}
            disabled={fieldDisabled}
          />
        </div>
      )
    case 'number':
      return (
        <NumberField
          label={field.label}
          value={typeof value === 'number' ? value : field.default || 0}
          onChange={onChange}
          min={field.min ?? 0}
          max={field.max ?? 100}
          step={field.step ?? 1}
          unit={field.unit}
          disabled={fieldDisabled}
          help={field.help}
        />
      )
    case 'browse':
      return (
        <BrowseField
          label={field.label}
          value={value || ''}
          onChange={onChange}
          browse={field.browse || 'file'}
          filters={field.filters}
          choices={field.choices}
          choices_dir={field.choices_dir}
          disabled={fieldDisabled}
          help={field.help}
        />
      )
    default:
      return <TextField label={field.label} value={String(value || '')} onChange={onChange} disabled={fieldDisabled} help={field.help} />
  }
}

export function ExportDialog({
  open,
  onClose,
  onStartExport,
  selectedFile,
  steps,
}: ExportDialogProps) {
  const [state, setState] = useState<{
    schema: ExportOptionsResponse | null
    loading: boolean
    error: string | null
    values: ExportOptions
    warnings: string[]
  }>({
    schema: null,
    loading: true,
    error: null,
    values: {},
    warnings: [],
  })
  const { data: settingsResponse } = useSettings()
  const updateSettings = useUpdateSettings()

  const settings = settingsResponse?.settings
  const scale = useMemo(() => {
    for (const step of steps) {
      if (step.feature === 'Upscale' && typeof step.parameters?.scale === 'number') return step.parameters.scale
    }
    return settings?.export.default_upscale || 1
  }, [steps, settings?.export.default_upscale])

  // Fetch schema when dialog opens
  useEffect(() => {
    if (!open) return
    let cancelled = false
    setState((s) => ({ ...s, loading: true, error: null, values: {}, warnings: [] }))

    api.getExportOptions(selectedFile?.name, scale)
      .then((schema) => {
        if (!cancelled) {
          const defaults = schema.defaults || {}
          setState((s) => ({
            ...s,
            schema,
            loading: false,
            values: defaults,
          }))
        }
      })
      .catch((err) => {
        if (!cancelled) setState((s) => ({ ...s, loading: false, error: String(err) }))
      })

    return () => { cancelled = true }
  }, [open, selectedFile?.name, scale])

  const handleChange = useCallback((key: string, value: any) => {
    setState((s) => ({
      ...s,
      values: { ...s.values, [key]: value },
    }))
  }, [])

  const handleStartExport = async () => {
    if (!state.schema || !settings) return

    const options = state.values
    const exportSettings = {
      // Persist last used values as the new defaults
      save_to: (options.save_to || 'exports') as 'original' | 'exports' | 'custom',
      export_dir: options.export_dir || '',
      output_type: (options.output_type || 'video') as 'video' | 'image_sequence',
      container: options.container || 'mp4',
      video_codec: options.codec || 'libx264',
      video_profile: options.profile || '',
      bitrate_mode: (options.bitrate_mode || 'dynamic') as 'dynamic' | 'constant',
      video_bitrate_mbps: options.bitrate_mbps || 12,
      video_quality: (options.quality || 'high') as 'low' | 'medium' | 'high',
      audio_mode: options.audio_mode || 'copy',
      image_format: options.image_format || 'png',
      lut_path: options.lut_path || '',
    }

    // Update settings (persists to disk)
    updateSettings.mutate({ export: { ...settings.export, ...exportSettings } })

    onStartExport(options)
    onClose()
  }

  if (!open) return null

  const { schema, loading, error, values, warnings } = state

  const leftFields = schema?.fields.filter((f) => f.key !== 'lut_path') || []
  const rightFields = schema?.fields.filter((f) => f.key === 'lut_path') || []

  // Partition: left = filename, save_to, export_dir, output_type
  // Right = everything else
  const leftKeys = ['filename', 'save_to', 'export_dir', 'output_type']
  const left = leftFields.filter((f) => leftKeys.includes(f.key))
  const right = leftFields.filter((f) => !leftKeys.includes(f.key)).concat(rightFields)

  return (
    <Dialog open={open} onClose={onClose} className="max-w-5xl w-full">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[var(--color-border)]">
          <h2 className="text-[15px] font-semibold text-[var(--color-text-primary)]">Export Settings</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-4">
          {error && (
            <div className="mb-4 p-3 bg-[var(--color-error-soft)] border border-[var(--color-error)]/30 rounded-[var(--radius-sm)] text-[12px] text-[var(--color-error)]">
              {error}
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
            </div>
          ) : schema ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left column */}
              <div className="space-y-4">
                {left.map((field) => (
                  <FieldRenderer
                    key={field.key}
                    field={field}
                    value={values[field.key]}
                    onChange={(v) => handleChange(field.key, v)}
                    schema={schema}
                  />
                ))}
              </div>

              {/* Right column */}
              <div className="space-y-4">
                {right.map((field) => (
                  <FieldRenderer
                    key={field.key}
                    field={field}
                    value={values[field.key]}
                    onChange={(v) => handleChange(field.key, v)}
                    schema={schema}
                  />
                ))}

                {/* Summary */}
                {warnings.length > 0 && (
                  <div className="p-3 bg-[var(--color-warning-soft)] border border-[var(--color-warning)]/30 rounded-[var(--radius-sm)]">
                    <div className="flex items-center gap-2 text-[11px] text-[var(--color-warning)] mb-1">
                      <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                      <span className="font-medium">Adjustments made:</span>
                    </div>
                    <ul className="text-[11px] text-[var(--color-text-secondary)] space-y-0.5 pl-4 list-disc">
                      {warnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  </div>
                )}

                {/* Output summary when we have all the key values */}
                {values.container && values.codec && (
                  <div className="p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-[var(--radius-sm)]">
                    <div className="text-[11px] font-medium uppercase tracking-wide text-[var(--color-text-muted)] mb-1">Output summary</div>
                    <div className="text-[12px] text-[var(--color-text-secondary)] font-mono">
                      {values.output_type === 'image_sequence'
                        ? `${values.image_format?.toUpperCase()} sequence in {${values.save_to || 'exports'}}`
                        : `${values.container?.toUpperCase()} · ${values.codec} · ${values.bitrate_mode === 'constant' ? values.bitrate_mbps + ' Mbps' : values.quality ? values.quality[0].toUpperCase() + values.quality.slice(1) + ' quality' : ''} · ${values.audio_mode || 'no audio'}${values.lut_path ? ` · LUT: ${values.lut_path.split(/[\\/]/).pop()}` : ''}`}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 p-4 border-t border-[var(--color-border)] bg-[var(--color-bg-panel)]">
          <Button variant="secondary" onClick={onClose} disabled={loading}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleStartExport}
            disabled={loading || !schema}
            className="min-w-[120px]"
          >
            <Save className="w-3.5 h-3.5 mr-1.5" />
            Start Export
          </Button>
        </div>
      </div>
    </Dialog>
  )
}