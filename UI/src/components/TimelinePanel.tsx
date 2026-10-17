import { Play, Pause, SkipBack, SkipForward, RotateCcw, Volume2, VolumeX, Maximize2 } from 'lucide-react'
import { clsx } from 'clsx'
import { useState } from 'react'

export function TimelinePanel() {
  const [isPlaying, _setIsPlaying] = useState(false)
  const [currentTime, _setCurrentTime] = useState(0)
  const [duration, _setDuration] = useState(30)
  const [volume, _setVolume] = useState(1)
  const [isMuted, _setIsMuted] = useState(false)
  const [showVolume, _setShowVolume] = useState(false)

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 100)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
  }

  return (
    <div className="timeline">
      {/* Transport bar */}
      <div className="panel-toolbar h-12">
        <div className="flex items-center gap-1">
          <button className="btn-icon btn-ghost" aria-label="Restart">
            <RotateCcw className="w-4 h-4" />
          </button>
          <button className="btn-icon btn-ghost" aria-label="Step back">
            <SkipBack className="w-4 h-4" />
          </button>
          <button
            className="grid place-items-center w-9 h-9 mx-1 rounded-full text-white bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] transition-colors shadow-[inset_0_1px_0_rgba(255,255,255,0.2),0_1px_3px_rgba(0,0,0,0.5)]"
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>
          <button className="btn-icon btn-ghost" aria-label="Step forward">
            <SkipForward className="w-4 h-4" />
          </button>
        </div>

        {/* Scrubber */}
        <div className="flex-1 max-w-3xl mx-4">
          <div className="relative h-1.5 rounded-full bg-[var(--color-bg-inset)] shadow-[inset_0_1px_2px_rgba(0,0,0,0.5)]">
            <div
              className="absolute top-0 left-0 h-full rounded-full bg-[var(--color-accent)] transition-all duration-75"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            />
            <div
              className="absolute top-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-[0_1px_3px_rgba(0,0,0,0.6)] pointer-events-none"
              style={{ left: `${(currentTime / duration) * 100}%` }}
            />
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] font-mono text-[var(--color-text-secondary)]">
          <span className="text-[var(--color-text-primary)]">{formatTime(currentTime)}</span>
          <span className="text-[var(--color-text-muted)]">/</span>
          <span>{formatTime(duration)}</span>
        </div>

        <div className="flex items-center gap-1 ml-3">
          <div className="relative">
            <button
              className="btn-icon btn-ghost"
              onClick={() => _setShowVolume(!showVolume)}
              aria-label={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted || volume === 0 ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
            {showVolume && (
              <div className="absolute bottom-full right-0 mb-2 w-32 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg-card)] p-2.5 shadow-[var(--shadow-md)]">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={isMuted ? 0 : volume}
                  onChange={(e) => _setVolume(Number(e.target.value))}
                  className="slider w-full"
                />
              </div>
            )}
          </div>
          <button className="btn-icon btn-ghost" aria-label="Fullscreen">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Time ruler */}
      <div className="h-14 bg-[var(--color-bg-base)] px-3 flex items-center">
        <div className="w-full h-9 bg-[var(--color-bg-panel)] border border-[var(--color-border)] rounded-[var(--radius-md)] overflow-hidden relative">
          <div className="absolute inset-0 flex items-center gap-1 px-2 overflow-x-auto">
            {[...Array(60)].map((_, i) => (
              <div
                key={i}
                className={clsx(
                  'w-6 h-full flex-shrink-0 relative',
                  i % 15 === 0 && 'border-l border-[var(--color-border)]'
                )}
              >
                {i % 15 === 0 && (
                  <span className="absolute top-0.5 left-1 text-[9px] font-mono text-[var(--color-text-muted)] whitespace-nowrap">
                    {formatTime((i / 60) * duration)}
                  </span>
                )}
              </div>
            ))}
            <div className="absolute bottom-0 left-0 h-0.5 bg-[var(--color-accent)]" style={{ width: `${(currentTime / duration) * 100}%` }} />
          </div>
        </div>
      </div>
    </div>
  )
}
