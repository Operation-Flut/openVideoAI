import { useEffect, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { clsx } from 'clsx'

interface DialogProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  description?: ReactNode
  /** Optional icon rendered next to the title. */
  icon?: ReactNode
  /** Controls shown on the right side of the header, before the close button. */
  headerActions?: ReactNode
  children: ReactNode
  footer?: ReactNode
  /** Tailwind max-width utility; controls the popup width. */
  size?: string
  className?: string
  /** Extra classes for the scrollable body (e.g. `p-0` for full-bleed content). */
  bodyClassName?: string
}

/**
 * Lightweight modal dialog rendered into a portal. Matches the native desktop
 * chrome: hairline borders, tight radii and a subtle blurred backdrop.
 *
 * Closes on Escape and on backdrop click; locks page scroll while open.
 */
export function Dialog({
  open,
  onClose,
  title,
  description,
  icon,
  headerActions,
  children,
  footer,
  size = 'max-w-[760px]',
  className,
  bodyClassName,
}: DialogProps) {
  useEffect(() => {
    if (!open) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onClose()
      }
    }

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.body.style.overflow = previousOverflow
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open, onClose])

  if (!open) return null

  return createPortal(
    <div className="modal-overlay" role="presentation" onMouseDown={onClose}>
      <div
        className={clsx('modal', size, className)}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === 'string' ? title : undefined}
        onMouseDown={(event) => event.stopPropagation()}
      >
        {(title || description) && (
          <header className={clsx('modal-header', headerActions && 'items-center')}>
            <div className="flex items-start gap-3 min-w-0">
              {icon && <span className="mt-0.5 shrink-0 text-[var(--color-accent)]">{icon}</span>}
              <div className="min-w-0">
                {title && <h2 className="modal-title">{title}</h2>}
                {description && <div className="modal-description">{description}</div>}
              </div>
            </div>
            <div className="flex items-center gap-2.5 shrink-0">
              {headerActions}
              <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
                <X className="w-4 h-4" />
              </button>
            </div>
          </header>
        )}

        <div className={clsx('modal-body', bodyClassName)}>{children}</div>

        {footer && <footer className="modal-footer">{footer}</footer>}
      </div>
    </div>,
    document.body
  )
}
