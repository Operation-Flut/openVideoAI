import { useEffect, useState } from 'react'
import { useUIStore } from '../store'

export function useTheme() {
  const { theme, setTheme } = useUIStore()
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('dark')

  const initializeTheme = () => {
    let resolved: 'light' | 'dark'

    if (theme === 'system') {
      resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    } else {
      resolved = theme
    }

    setResolvedTheme(resolved)
    document.documentElement.classList.toggle('light', resolved === 'light')
    document.documentElement.classList.toggle('dark', resolved === 'dark')
  }

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
      if (theme === 'system') {
        initializeTheme()
      }
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [theme])

  useEffect(() => {
    initializeTheme()
  }, [theme])

  return {
    theme,
    resolvedTheme,
    setTheme,
    toggleTheme,
    initializeTheme,
  }
}