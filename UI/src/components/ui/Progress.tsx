import { clsx } from 'clsx'

interface ProgressProps {
  value: number
  max?: number
  showLabel?: boolean
  label?: string
  variant?: 'default' | 'success' | 'warning' | 'error'
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const VARIANTS = {
  default: 'bg-[var(--color-accent)]',
  success: 'bg-emerald-500',
  warning: 'bg-amber-500',
  error: 'bg-red-500',
} as const

const SIZES = {
  sm: 'h-1',
  md: 'h-1.5',
  lg: 'h-2.5',
} as const

export function Progress({ value, max = 100, showLabel, label, variant = 'default', size = 'md', className }: ProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={clsx('w-full', className)}>
      {(label || showLabel) && (
        <div className="flex items-center justify-between mb-1.5">
          {label && <span className="text-[12px] font-medium text-[var(--color-text-secondary)]">{label}</span>}
          {showLabel && <span className="text-[12px] text-[var(--color-text-muted)] font-mono">{percentage.toFixed(0)}%</span>}
        </div>
      )}
      <div className={clsx('progress', SIZES[size])}>
        <div
          className={clsx('h-full rounded-full transition-all duration-300 ease-out', VARIANTS[variant])}
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={value}
          aria-valuemin={0}
          aria-valuemax={max}
        />
      </div>
    </div>
  )
}
