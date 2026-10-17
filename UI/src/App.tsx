import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { EnhancePage } from './pages/EnhancePage'
import { ModelsPage } from './pages/ModelsPage'
import { SettingsPage } from './pages/SettingsPage'
import { Layout } from './components/Layout'
import { useTheme } from './hooks/useTheme'
import { useSettings } from './hooks/useSettings'
import { useUIStore } from './store'

function App() {
  const { initializeTheme } = useTheme()
  const { data: settingsResponse } = useSettings()
  const setTheme = useUIStore((state) => state.setTheme)
  const setAccentColor = useUIStore((state) => state.setAccentColor)
  const setFontSize = useUIStore((state) => state.setFontSize)
  const setDensity = useUIStore((state) => state.setDensity)

  // Initialize theme on mount
  React.useEffect(() => {
    initializeTheme()
  }, [initializeTheme])

  React.useEffect(() => {
    const appearance = settingsResponse?.settings.appearance
    if (!appearance) return
    setTheme(appearance.theme)
    setAccentColor(appearance.accent_color)
    setFontSize(appearance.font_size)
    setDensity(appearance.density)
  }, [settingsResponse, setTheme, setAccentColor, setFontSize, setDensity])

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/enhance" replace />} />
        <Route path="/enhance" element={<EnhancePage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/enhance" replace />} />
      </Routes>
    </Layout>
  )
}

export default App