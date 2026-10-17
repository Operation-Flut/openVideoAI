import { Link, useLocation } from 'react-router-dom'
import { ChevronRight, Settings, Download, Info, Check, Loader2 } from 'lucide-react'
import { useUIStore } from '../store'
import { useModels, useImplementedFeatures } from '../hooks/useModels'
import { clsx } from 'clsx'

export function Sidebar() {
  const { sidebarOpen, setSidebarOpen, density } = useUIStore()
  const { data: models, isLoading: modelsLoading } = useModels()
  const { data: implementedFeatures } = useImplementedFeatures()
  const location = useLocation()

  const densityClasses = {
    compact: 'py-1.5 px-3 text-sm',
    comfortable: 'py-2 px-3 text-sm',
    spacious: 'py-3 px-4 text-base',
  }

  // Merge downloaded models with feature info
  const featureSections = [
    { key: 'upscale', label: 'Upscale', icon: Settings, required: true },
    { key: 'face', label: 'Face Recovery', icon: Settings },
    { key: 'sharpen', label: 'Sharpen / Deblur', icon: Settings },
    { key: 'denoise', label: 'Denoise', icon: Settings },
    { key: 'frame', label: 'Frame Interpolation', icon: Settings },
    { key: 'color', label: 'Match Color', icon: Settings },
    { key: 'lowlight', label: 'Low-Light', icon: Settings },
  ]

  if (!sidebarOpen) {
    return (
      <aside className="fixed inset-y-0 left-0 z-50 w-20 bg-dark-900 border-r border-dark-700 transform transition-transform duration-300 lg:translate-x-0">
        <div className="flex flex-col h-full">
          <div className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
            {featureSections.map((section) => {
              const implemented = implementedFeatures?.[section.label]
              const downloaded = models?.find(m => m.downloaded && implemented?.availableModels?.includes(m.name)) || null
              const isActive = location.pathname === '/enhance'

              return (
                <Link
                  key={section.key}
                  to="/enhance"
                  className={clsx(
                    'relative flex flex-col items-center gap-1 rounded-lg transition-all duration-200',
                    densityClasses[density],
                    isActive
                      ? 'bg-primary-600/20 text-primary-400'
                      : 'text-dark-400 hover:text-dark-100 hover:bg-dark-800'
                  )}
                  title={section.label}
                >
                  <section.icon className="w-5 h-5" />
                  <span className="hidden lg:block whitespace-nowrap">{section.label}</span>
                  {downloaded && (
                    <Check className="absolute top-1 right-1 w-4 h-4 text-green-400" />
                  )}
                  {implemented && !downloaded && modelsLoading && (
                    <Loader2 className="absolute top-1 right-1 w-4 h-4 text-yellow-400 animate-spin" />
                  )}
                </Link>
              )
            })}
          </div>

          <button
            onClick={() => setSidebarOpen(true)}
            className="p-3 mx-2 rounded-lg hover:bg-dark-800 transition-colors"
            aria-label="Expand sidebar"
          >
            <ChevronRight className="w-5 h-5 text-dark-400 mx-auto" />
          </button>
        </div>
      </aside>
    )
  }

  return (
    <aside className="fixed inset-y-0 left-0 z-50 w-64 bg-dark-900 border-r border-dark-700 transform transition-transform duration-300 lg:translate-x-0">
      <div className="flex flex-col h-full">
        {/* Logo section */}
        <div className="p-4 border-b border-dark-700">
          <Link to="/enhance" className="flex items-center gap-2 font-bold text-lg text-dark-50">
            <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
              <Settings className="w-5 h-5 text-white" />
            </div>
            <span>openVideoAI</span>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-2 py-4 space-y-1">
          <div className="px-3 py-2 text-xs font-semibold text-dark-500 uppercase tracking-wider">
            Enhancement Pipeline
          </div>

          {featureSections.map((section) => {
            const implemented = implementedFeatures?.[section.label]
            const availableModels = implemented?.availableModels || []
            const defaultModel = implemented?.defaultModel
            const downloadedModelsForFeature = models?.filter(m => m.downloaded && availableModels.includes(m.name)) || []
            const downloadedCount = downloadedModelsForFeature.length

            const isActive = location.pathname === '/enhance'

            return (
              <div key={section.key} className="group">
                <Link
                  to="/enhance"
                  className={clsx(
                    'flex items-center gap-3 rounded-lg transition-all duration-200',
                    densityClasses[density],
                    isActive
                      ? 'bg-primary-600/20 text-primary-400'
                      : 'text-dark-400 hover:text-dark-100 hover:bg-dark-800'
                  )}
                >
                  <section.icon className="w-5 h-5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{section.label}</p>
                    <p className="text-xs text-dark-500 truncate">
                      {downloadedCount} / {availableModels.length} models
                    </p>
                  </div>
                  {downloadedCount > 0 && (
                    <Check className="w-4 h-4 text-green-400 flex-shrink-0" />
                  )}
                </Link>

                {/* Model dropdown on hover */}
                {availableModels.length > 0 && (
                  <div className="ml-7 mt-1 space-y-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    {availableModels.map((modelName: string) => {
                      const model = downloadedModelsForFeature.find(m => m.name === modelName)
                      const isDefault = modelName === defaultModel

                      return (
                        <Link
                          key={modelName}
                          to="/models"
                          className={clsx(
                            'flex items-center gap-2 rounded transition-colors',
                            densityClasses[density],
                            model?.downloaded
                              ? 'text-dark-300 hover:text-dark-100 hover:bg-dark-800'
                              : 'text-dark-500 hover:text-dark-400'
                          )}
                        >
                          {model?.downloaded && <Check className="w-3 h-3 text-green-400 flex-shrink-0" />}
                          <span className="truncate text-sm flex-1">{modelName}</span>
                          {isDefault && (
                            <span className="text-xs px-1.5 py-0.5 bg-primary-600/20 text-primary-400 rounded">
                              Default
                            </span>
                          )}
                        </Link>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}

          <div className="pt-4 border-t border-dark-700">
            <div className="px-3 py-2 text-xs font-semibold text-dark-500 uppercase tracking-wider">
              Navigation
            </div>

            {[
              { path: '/models', label: 'Model Manager', icon: Download, badge: models?.filter(m => !m.downloaded && m.has_download_url).length || 0 },
              { path: '/settings', label: 'Settings', icon: Info, badge: null },
            ].map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={clsx(
                  'flex items-center gap-3 rounded-lg transition-all duration-200',
                  densityClasses[density],
                  location.pathname === item.path
                    ? 'bg-primary-600/20 text-primary-400'
                    : 'text-dark-400 hover:text-dark-100 hover:bg-dark-800'
                )}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                <span className="flex-1">{item.label}</span>
                {item.badge && item.badge > 0 && (
                  <span className="text-xs px-2 py-0.5 bg-primary-600/20 text-primary-400 rounded-full">
                    {item.badge}
                  </span>
                )}
              </Link>
            ))}
          </div>
        </nav>

        {/* Collapse button */}
        <div className="p-3 border-t border-dark-700">
          <button
            onClick={() => setSidebarOpen(false)}
            className="w-full p-2 rounded-lg hover:bg-dark-800 transition-colors"
            aria-label="Collapse sidebar"
          >
            <ChevronRight className="w-5 h-5 text-dark-400 mx-auto rotate-180" />
          </button>
        </div>
      </div>
    </aside>
  )
}