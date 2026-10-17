import { forwardRef, type SelectHTMLAttributes } from 'react'
import { clsx } from 'clsx'
import type { ModelResponse } from '../../types/api'

interface ModelSelectorProps extends SelectHTMLAttributes<HTMLSelectElement> {
  models: ModelResponse[]
  placeholder?: string
  showStatus?: boolean
  filter?: (model: ModelResponse) => boolean
}

export const ModelSelector = forwardRef<HTMLSelectElement, ModelSelectorProps>(
  ({ className, models, placeholder = 'Select a model', showStatus = true, filter, ...props }, ref) => {
    const filteredModels = models.filter((m) => !filter || filter(m))

    return (
      <select
        ref={ref}
        className={clsx('select', className)}
        {...props}
      >
        <option value="" disabled>
          {placeholder}
        </option>
        {filteredModels.length === 0 && (
          <option value="" disabled>
            No models available
          </option>
        )}
        {filteredModels.map((model) => (
          <option key={model.name} value={model.name}>
            {model.name}
            {showStatus && model.downloaded && ' ✓'}
            {showStatus && !model.downloaded && ' ⬇'}
            {!model.compatible && ' (incompatible)'}
          </option>
        ))}
      </select>
    )
  }
)

ModelSelector.displayName = 'ModelSelector'
