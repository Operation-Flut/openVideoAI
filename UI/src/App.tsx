import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { EnhancePage } from './pages/EnhancePage'
import { ModelsPage } from './pages/ModelsPage'
import { SettingsPage } from './pages/SettingsPage'
import { Layout } from './components/Layout'
import { useTheme } from './hooks/useTheme'

function App() {
  const { initializeTheme } = useTheme()

  // Initialize theme on mount
  React.useEffect(() => {
    initializeTheme()
  }, [initializeTheme])

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