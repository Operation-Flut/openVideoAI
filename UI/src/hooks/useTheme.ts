import { useCallback, useEffect, useState } from 'react'
import { useUIStore } from '../store'

function darkenHexColor(color: string, amount: number): string {
  const hex = /^#([\da-f]{3}|[\da-f]{6})$/i.test(color)
    ? color.slice(1)
    : '2563eb'
  const expanded = hex.length === 3 ? [...hex].map((part) => part.repeat(2)).join('') : hex
  const channels = [0, 2, 4].map((index) =>
    Math.max(0, Math.floor(parseInt(expanded.slice(index, index + 2), 16) * (1 - amount))),
  )
  return `#${channels.map((channel) => channel.toString(16).padStart(2, '0')).join('')}`
}

export function useTheme() {
  const { theme, setTheme } = useUIStore()
  const accentColor = useUIStore((state) => state.accentColor)
  const fontSize = useUIStore((state) => state.fontSize)
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('dark')

  const initializeTheme = useCallback(() => {
    let resolved: 'light' | 'dark'

    if (theme === 'system') {
      resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    } else {
      resolved = theme
    }

    setResolvedTheme(resolved)
    document.documentElement.classList.toggle('light', resolved === 'light')
    document.documentElement.classList.toggle('dark', resolved === 'dark')
  }, [theme])

  const toggleTheme = () => {
    const themes: ('light' | 'dark' | 'system')[] = ['light', 'dark', 'system']
    const currentIndex = themes.indexOf(theme)
    const nextIndex = (currentIndex + 1) % themes.length
    setTheme(themes[nextIndex])
  }

  useEffect(() => {
    initializeTheme()
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      if (theme === 'system') initializeTheme()
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [initializeTheme, theme])

  useEffect(() => {
    const root = document.documentElement
    const validColor = /^#(?:[\da-f]{3}|[\da-f]{6})$/i.test(accentColor)
      ? accentColor
      : '#2563eb'
    const [red, green, blue] = validColor.length === 4
      ? [...validColor.slice(1)].map((channel) => parseInt(channel + channel, 16))
      : [1, 3, 5].map((index) => parseInt(validColor.slice(index, index + 2), 16))
    const size = { small: '14px', medium: '16px', large: '18px' }[fontSize]

    root.style.setProperty('--color-accent', validColor)
    root.style.setProperty('--color-accent-hover', darkenHexColor(validColor, 0.15))
    root.style.setProperty('--color-accent-muted', darkenHexColor(validColor, 0.45))
    root.style.setProperty('--color-accent-soft', `rgba(${red}, ${green}, ${blue}, 0.14)`)
    root.style.fontSize = size
  }, [accentColor, fontSize])

  return {
    theme,
    resolvedTheme,
    setTheme,
    toggleTheme,
    initializeTheme,
  }
}