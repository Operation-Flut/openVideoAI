import { ReactNode } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Header } from './Header'
import { PipelineSidebar } from './PipelineSidebar'
import { TimelinePanel } from './TimelinePanel'
import { useModels } from '../hooks/useModels'
import { useUIStore } from '../store'
import { clsx } from 'clsx'

export function Layout({ children }: { children?: ReactNode }) {
  const { sidebarOpen } = useUIStore()
  const location = useLocation()
  const isEnhanceRoute = location.pathname === '/enhance'

  // Keep the catalog fresh app-wide: the Enhance sidebar needs current
  // download/compatibility status, and the store is persisted across sessions.
  useModels()

  return (
    <div className="workspace">
      <Header />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <main className="canvas-area">
          {children || <Outlet />}
          {isEnhanceRoute && <TimelinePanel />}
        </main>
        {isEnhanceRoute && (
          <aside
            className={clsx(
              'panel-right transition-[width] duration-200 ease-out',
              sidebarOpen ? 'w-[340px]' : 'w-0 overflow-hidden collapsed'
            )}
            aria-hidden={!sidebarOpen}
          >
            <PipelineSidebar />
          </aside>
        )}
      </div>
    </div>
  )
}
