import { displayFeature } from './displayNames'
import type { PipelineStep } from '../types/api'
import type { RenderJob } from '../store/renderQueue'

/** Format a byte count in the compact style used across the app. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 KB'
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / 1024).toFixed(1)} KB`
}

/** Human-readable duration. Short values stay in seconds so ETAs feel precise. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  const total = Math.round(seconds)
  if (total < 60) return `${total}s`

  const minutes = Math.floor(total / 60)
  const rest = total % 60
  if (minutes < 60) return rest ? `${minutes}m ${rest}s` : `${minutes}m`

  const hours = Math.floor(minutes / 60)
  const rem = minutes % 60
  return rem ? `${hours}h ${rem}m` : `${hours}h`
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

const CODEC_LABELS: Record<string, string> = {
  libx264: 'H.264',
  h264: 'H.264',
  h264_nvenc: 'H.264 NVENC',
  libx265: 'H.265',
  hevc: 'H.265',
  hevc_nvenc: 'H.265 NVENC',
  libvpx_vp9: 'VP9',
  'libvpx-vp9': 'VP9',
  vp9: 'VP9',
  av1_nvenc: 'AV1 NVENC',
  av1: 'AV1',
  libaom_av1: 'AV1',
  libsvtav1: 'AV1',
  prores: 'ProRes',
  prores_ks: 'ProRes',
  mjpeg: 'MJPEG',
}

export function codecLabel(codec: string | undefined): string {
  if (!codec) return ''
  return CODEC_LABELS[codec.toLowerCase()] || codec
}

/** One pipeline step, with its most meaningful parameter appended. */
export function describeStep(step: PipelineStep): string {
  const label = displayFeature(step.feature)
  const params = step.parameters || {}

  switch (step.feature) {
    case 'Upscale':
      return params.scale ? `${label} ${params.scale}×` : label
    case 'Frame Interpolation':
      return params.multiplier ? `${label} ${params.multiplier}×` : label
    case 'Face Recovery':
      return params.fidelity_weight != null
        ? `${label} ${Number(params.fidelity_weight).toFixed(2)}`
        : label
    case 'Sharpen / Deblur':
    case 'Denoise': {
      const strength = params.strength ?? params.unsharp_amount
      return strength != null ? `${label} ${Number(strength).toFixed(2)}` : label
    }
    default:
      return label
  }
}

export function describeJobFilters(job: RenderJob): string {
  if (job.steps.length === 0) return 'No filters'
  return job.steps.map(describeStep).join(' + ')
}

/** What the encoder will produce, e.g. `MP4 · H.265 · High quality · Audio copy`. */
export function describeJobOutput(job: RenderJob): string {
  const options = job.exportOptions || {}

  if (options.output_type === 'image_sequence') {
    return `${String(options.image_format || 'png').toUpperCase()} sequence`
  }

  const parts: string[] = []
  const container = options.container || (job.outputFormat !== 'source' ? job.outputFormat : '')
  if (container) parts.push(String(container).toUpperCase())
  if (options.codec) parts.push(codecLabel(options.codec))

  if (options.bitrate_mode === 'constant' && options.bitrate_mbps) {
    parts.push(`${options.bitrate_mbps} Mbps`)
  } else if (options.quality) {
    parts.push(`${capitalize(String(options.quality))} quality`)
  }

  if (options.audio_mode && options.audio_mode !== 'none') {
    parts.push(options.audio_mode === 'copy' ? 'Audio copy' : `Audio ${String(options.audio_mode).toUpperCase()}`)
  }

  if (options.lut_path) parts.push('LUT')

  return parts.filter(Boolean).join(' · ') || 'Source format'
}

/** Last path component, for showing a produced file without the full folder. */
export function fileNameFromPath(path: string | null | undefined): string {
  if (!path) return ''
  const parts = path.split(/[\\/]/)
  return parts[parts.length - 1] || path
}
