import { forwardRef, useId, type InputHTMLAttributes, type ReactNode } from 'react'
import { clsx } from 'clsx'

interface ToggleProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size' | 'onChange'> {
  label?: string
  description?: string
  size?: 'sm' | 'md' | 'lg'
  /** Optional inline element rendered directly after the label (e.g. an info button). */
  labelExtra?: ReactNode
  onChange?: (checked: boolean) => void
}

// Track + thumb travel are computed so the thumb lands flush at the far edge.
const SIZES = {
  sm: { track: 'w-8 h-[18px]', thumb: 'w-3.5 h-3.5 top-0.5 left-0.5', on: 'peer-checked:translate-x-[14px]' },
  md: { track: 'w-11 h-6', thumb: 'w-5 h-5 top-0.5 left-0.5', on: 'peer-checked:translate-x-5' },
  lg: { track: 'w-14 h-7', thumb: 'w-6 h-6 top-0.5 left-0.5', on: 'peer-checked:translate-x-7' },
} as const

export const Toggle = forwardRef<HTMLInputElement, ToggleProps>(
  ({ className, label, description, size = 'md', id, labelExtra, onChange, ...props }, ref) => {
    const reactId = useId()
    const toggleId = id || `toggle-${reactId.replace(/:/g, '')}`
    const { track, thumb, on } = SIZES[size]

    // Handle onChange to pass the boolean checked value instead of the event
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const checked = e.target.checked
      if (onChange) {
        onChange(checked)
      }
    }

    return (
      <div className={clsx('flex items-center gap-3', className)}>
        <label htmlFor={toggleId} className="cursor-pointer flex items-center shrink-0">
          {/* input, track and thumb are siblings so peer-checked variants apply */}
          <span className="relative inline-flex">
            <input
              ref={ref}
              type="checkbox"
              id={toggleId}
              className="sr-only peer"
              onChange={handleChange}
              {...props}
            />
            <span
              className={clsx(
                'block rounded-full transition-colors duration-[var(--transition-fast)]',
                track,
                'bg-[var(--color-bg-inset)] ring-1 ring-inset ring-[var(--color-border)]',
                'peer-checked:bg-[var(--color-accent)] peer-checked:ring-[var(--color-accent)]',
                'peer-focus-visible:ring-2 peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-[var(--color-bg-base)] peer-focus-visible:ring-[var(--color-accent)]',
                'shadow-[inset_0_1px_2px_rgba(0,0,0,0.5)]'
              )}
            />
            <span
              className={clsx(
                'absolute block rounded-full bg-white transition-transform duration-[var(--transition-fast)]',
                'shadow-[0_1px_3px_rgba(0,0,0,0.5)]',
                thumb,
                on
              )}
            />
          </span>
        </label>
        {(label || description) && (
          <div className="text-left min-w-0">
            {label && (
              <span className="flex items-center gap-1.5">
                <span className="text-[13px] font-medium text-[var(--color-text-primary)]">{label}</span>
                {labelExtra}
              </span>
            )}
            {description && <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5">{description}</p>}
          </div>
        )}
      </div>
    )
  }
)

Toggle.displayName = 'Toggle'
