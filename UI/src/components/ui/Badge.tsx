import type { ReactNode } from 'react'
import { clsx } from 'clsx'

interface BadgeProps {
  children: ReactNode
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info' | 'outline'
  size?: 'sm' | 'md'
  className?: string
}

const VARIANTS = {
  default: 'bg-white/[0.06] text-[var(--color-text-secondary)]',
  success: 'bg-emerald-500/15 text-emerald-400',
  warning: 'bg-amber-500/15 text-amber-400',
  error: 'bg-red-500/15 text-red-400',
  info: 'bg-blue-500/15 text-blue-300',
  outline: 'bg-transparent text-[var(--color-text-muted)] ring-1 ring-inset ring-[var(--color-border)]',
} as const

const SIZES = {
  sm: 'px-1.5 py-0.5 text-[10px]',
  md: 'px-2 py-0.5 text-[10px]',
} as const

export function Badge({ children, variant = 'default', size = 'md', className }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-[3px] font-semibold uppercase tracking-wider leading-none whitespace-nowrap',
        SIZES[size],
        VARIANTS[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
