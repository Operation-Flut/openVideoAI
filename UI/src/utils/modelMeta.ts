import type { ModelResponse } from '../types/api'

// Feature pill colours — distinct hues on the monochrome surface.
export const FEATURE_TAG_STYLES: Record<string, string> = {
  'Upscale': 'bg-sky-500/10 text-sky-300',
  'Face Recovery': 'bg-pink-500/10 text-pink-300',
  'Sharpen / Deblur': 'bg-amber-500/10 text-amber-300',
  'Denoise': 'bg-emerald-500/10 text-emerald-300',
  'Frame Interpolation': 'bg-violet-500/10 text-violet-300',
  'Match Color': 'bg-orange-500/10 text-orange-300',
  'Low-Light Enhancement': 'bg-indigo-500/10 text-indigo-300',
  'SDR to HDR': 'bg-rose-500/10 text-rose-300',
  'Reencoding': 'bg-zinc-500/10 text-zinc-300',
}

export const FALLBACK_TAG_STYLE = 'bg-white/5 text-[var(--color-text-secondary)]'

// Pill colours for the "Best used for" line in the model details popup, keyed by
// the catalog's `media` value (the label shown alongside comes from the backend).
export const MEDIA_TAG_STYLES: Record<string, string> = {
  image: 'bg-sky-500/10 text-sky-300',
  video: 'bg-violet-500/10 text-violet-300',
  both: 'bg-teal-500/10 text-teal-300',
}

export type ModelStatusKey = 'builtin' | 'downloaded' | 'available' | 'incompatible'

export const STATUS_BADGE: Record<ModelStatusKey, string> = {
  builtin: 'bg-violet-500/15 text-violet-300',
  downloaded: 'bg-emerald-500/15 text-emerald-400',
  available: 'bg-blue-500/15 text-blue-300',
  incompatible: 'bg-amber-500/15 text-amber-400',
}

export const STATUS_LABEL: Record<ModelStatusKey, string> = {
  builtin: 'Built-in',
  downloaded: 'Downloaded',
  available: 'Available',
  incompatible: 'Incompatible',
}

export function statusFor(model: ModelResponse): ModelStatusKey {
  if (model.is_builtin) return 'builtin'
  if (model.downloaded) return 'downloaded'
  if (model.compatible) return 'available'
  return 'incompatible'
}
