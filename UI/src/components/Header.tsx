import { Link, useLocation } from 'react-router-dom'
import { Sun, Moon, Monitor, Menu, ListVideo } from 'lucide-react'
import { useUIStore } from '../store'
import { isActiveStatus, useRenderQueueStore } from '../store/renderQueue'
import { useTheme } from '../hooks/useTheme'
import { clsx } from 'clsx'

const NAV_ITEMS = [
  { path: '/enhance', label: 'Enhance' },
  { path: '/models', label: 'Models' },
  { path: '/settings', label: 'Settings' },
]

const THEME_ITEMS = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'System', Icon: Monitor },
] as const

export function Header() {
  const { sidebarOpen, toggleSidebar } = useUIStore()
  const { theme, resolvedTheme, setTheme, toggleTheme } = useTheme()
  const location = useLocation()

  // Export queue badge: show the in-flight percent while a render runs, or the
  // number of waiting jobs when the queue is idle but not empty.
  const jobs = useRenderQueueStore((state) => state.jobs)
  const openQueue = useRenderQueueStore((state) => state.openQueue)
  const activeJobs = jobs.filter((job) => isActiveStatus(job.status))
  const queuedCount = jobs.filter((job) => job.status === 'queued').length
  const pendingCount = activeJobs.length + queuedCount
  const activePercent = activeJobs[0]?.progress ?? 0

  return (
    <header className="app-header">
      {/* Branding */}
      <div className="app-header-left">
        <button
          onClick={toggleSidebar}
          className="lg:hidden btn-icon"
          aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          <Menu className="w-4 h-4" />
        </button>

        <Link to="/enhance" className="app-title">
          <span className="app-mark">
            <img src="/icon/black_glass.png" alt="" />
          </span>
          <span className="hidden sm:flex flex-col leading-none">
            <span className="tracking-tight">openVideoAI</span>
            <span className="text-[10px] font-normal tracking-wide text-[var(--color-text-muted)] mt-0.5">
              ENHANCEMENT SUITE
            </span>
          </span>
        </Link>
      </div>

      {/* Integrated navigation */}
      <nav className="nav-tabs" role="tablist" aria-label="Primary">
        {NAV_ITEMS.map((item) => {
          const active = location.pathname === item.path
          return (
            <Link
              key={item.path}
              to={item.path}
              role="tab"
              aria-selected={active}
              className={clsx('nav-tab', active && 'nav-tab-active')}
            >
              {item.label}
            </Link>
          )
        })}
      </nav>

      {/* Window actions */}
      <div className="header-actions">
        <button
          onClick={openQueue}
          className="btn-icon relative"
          aria-label="Open export queue"
          title="Export queue"
        >
          <ListVideo className="w-4 h-4" />
          {pendingCount > 0 && (
            <span
              className={clsx(
                'absolute -top-1 -right-1 grid place-items-center h-[15px] px-1 rounded-full',
                'text-[9px] font-semibold leading-none tabular-nums text-white',
                'bg-[var(--color-accent)]',
                activeJobs.length > 0 && 'animate-pulse'
              )}
            >
              {activeJobs.length > 0 ? `${Math.round(activePercent)}%` : pendingCount}
            </span>
          )}
        </button>

        <div className="segmented hidden sm:flex" role="group" aria-label="Theme">
          {THEME_ITEMS.map(({ value, label, Icon }) => (
            <button
              key={value}
              onClick={() => setTheme(value)}
              className={clsx('segmented-item', theme === value && 'segmented-item-active')}
              aria-label={`Set theme to ${label}`}
              aria-pressed={theme === value}
              title={label}
            >
              <Icon className="w-3.5 h-3.5" />
            </button>
          ))}
        </div>

        <button
          onClick={toggleTheme}
          className="sm:hidden btn-icon"
          aria-label="Toggle theme"
        >
          {resolvedTheme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  )
}
