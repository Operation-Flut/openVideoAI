import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from 'react'
import { clsx } from 'clsx'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  error?: boolean
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, error, ...props }, ref) => (
    <input
      ref={ref}
      className={clsx('input', error && 'border-red-500/70 focus:border-red-500', className)}
      {...props}
    />
  )
)

Input.displayName = 'Input'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: boolean
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, error, ...props }, ref) => (
    <textarea
      ref={ref}
      className={clsx('input resize-y min-h-[80px]', error && 'border-red-500/70 focus:border-red-500', className)}
      {...props}
    />
  )
)

Textarea.displayName = 'Textarea'
