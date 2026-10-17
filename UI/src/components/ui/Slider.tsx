import { useId } from 'react'
import { clsx } from 'clsx'

interface SliderProps {
  label?: string
  valueLabel?: string
  min?: number
  max?: number
  step?: number
  showValue?: boolean
  value?: number
  defaultValue?: number
  className?: string
  id?: string
  onChange?: (value: number) => void
}

export const Slider = ({
  className,
  label,
  valueLabel,
  min = 0,
  max = 100,
  step = 1,
  showValue = true,
  id,
  value,
  defaultValue,
  onChange,
  ...props
}: SliderProps) => {
  const reactId = useId()
  const sliderId = id || `slider-${reactId.replace(/:/g, '')}`
  const resolvedValue = value ?? defaultValue ?? 0
  const clampedValue = Math.max(0, Math.min(max, Math.max(min, resolvedValue)))
  const percentage = max === min ? 0 : ((clampedValue - min) / (max - min)) * 100

  return (
    <div className={clsx('w-full', className)}>
      {label && (
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor={sliderId} className="text-[12px] font-medium text-[var(--color-text-secondary)]">
            {label}
          </label>
          {showValue && valueLabel && (
            <span className="text-[12px] text-[var(--color-text-primary)] font-mono">{valueLabel}</span>
          )}
        </div>
      )}
      <input
        type="range"
        id={sliderId}
        min={min}
        max={max}
        step={step}
        value={clampedValue}
        className={clsx('slider', className)}
        style={{
          background: `linear-gradient(to right, var(--color-accent) ${percentage}%, var(--color-border) ${percentage}%)`,
        } as React.CSSProperties}
        onChange={(e) => {
          const val = Number(e.target.value)
          onChange?.(val)
        }}
        {...props}
      />
    </div>
  )
}

export { clsx } from 'clsx'