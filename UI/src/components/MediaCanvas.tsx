import { Sparkles, Minimize2, Maximize2, RotateCcw, RotateCw } from 'lucide-react'
import { usePipeline } from '../hooks/usePipeline'
import { useProcessingStore } from '../store'
import { Button, Badge, Select } from '../components/ui'
import { api } from '../services/api'
import { useState, useRef, useEffect } from 'react'
import { clsx } from 'clsx'

interface MediaCanvasProps {
  selectedFile: File | null
  onFileSelect: (file: File) => void
}

type ViewMode = 'single' | 'split' | 'side-by-side' | 'compare'
type ZoomLevel = 'fit' | '100' | '200' | '400'

interface MediaMeta {
  width: number
  height: number
  duration: number
}

// Sub-component for preview panes
function PreviewPane({ 
  label, 
  isVideo, 
  src, 
  videoRef, 
  isProcessing = false,
  showControls = false,
  onLoadedMetadata,
}: { 
  label: string
  isVideo: boolean
  src: string
  videoRef: React.RefObject<HTMLVideoElement>
  isProcessing: boolean | (() => boolean)
  showControls?: boolean
  onLoadedMetadata?: (el: HTMLVideoElement | HTMLImageElement) => void
}) {
  return (
    <div className="relative flex flex-col overflow-hidden h-full">
      <div className="absolute top-2 left-2 z-10 bg-black/60 backdrop-blur px-2 py-1 rounded text-[10px] font-medium text-white">
        {label}
      </div>
      <div className="flex-1 relative overflow-hidden flex items-center justify-center">
        {isVideo ? (
          <video
            ref={videoRef}
            src={src}
            className="w-full h-full object-contain"
            controls={showControls}
            playsInline
            muted={!showControls}
            autoPlay={!showControls}
            loop={!showControls}
            onLoadedMetadata={(e) => onLoadedMetadata?.(e.currentTarget)}
          />
        ) : (
          <img
            src={src}
            alt={label}
            className="w-full h-full object-contain"
            onLoad={(e) => onLoadedMetadata?.(e.currentTarget)}
          />
        )}
        {typeof isProcessing === 'function' ? isProcessing() : isProcessing && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>
    </div>
  )
}

export function MediaCanvas({ selectedFile, onFileSelect }: MediaCanvasProps) {
  const { stepCount, steps } = usePipeline()
  const { currentTask } = useProcessingStore()
  const [viewMode, setViewMode] = useState<ViewMode>('single')
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>('fit')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [splitPosition, setSplitPosition] = useState(50)
  const [trimIn, setTrimIn] = useState(0)
  const [trimOut, setTrimOut] = useState(100)
  const [showTrim, setShowTrim] = useState(false)
  const [inputMeta, setInputMeta] = useState<MediaMeta | null>(null)
  const [outputMeta, setOutputMeta] = useState<MediaMeta | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const enhancedVideoRef = useRef<HTMLVideoElement>(null)

  const isVideo = selectedFile?.type.startsWith('video/') ?? false
  const currentTaskStatus = currentTask?.status
  const isProcessing = (): boolean => currentTaskStatus === 'processing'

  const objectUrl = selectedFile ? URL.createObjectURL(selectedFile) : ''

  // A finished task's output lives on the backend (exports folder or temp for
  // previews); the browser can only load it through the result endpoint.
  const hasResult = currentTaskStatus === 'completed' && !!currentTask?.output_path
  const resultUrl = hasResult ? api.taskResultUrl(currentTask!.task_id) : ''
  const enhancedSrc = resultUrl || objectUrl

  // Label the enhanced pane with the filters actually queued.
  const activeFiltersLabel = steps.length
    ? steps.map((s) => s.feature).join(' + ')
    : 'Enhanced'

  const readMeta = (el: HTMLVideoElement | HTMLImageElement): MediaMeta | null => {
    const width = el instanceof HTMLVideoElement ? el.videoWidth : el.naturalWidth
    const height = el instanceof HTMLVideoElement ? el.videoHeight : el.naturalHeight
    if (!width || !height) return null
    const duration = el instanceof HTMLVideoElement && Number.isFinite(el.duration) ? el.duration : 0
    return { width, height, duration }
  }

  const inputMetaInfo = inputMeta ? {
    resolution: `${inputMeta.width} × ${inputMeta.height}`,
    duration: isVideo && inputMeta.duration ? formatTime(inputMeta.duration) : '',
  } : null

  const outputMetaInfo = outputMeta ? {
    resolution: `${outputMeta.width} × ${outputMeta.height}`,
    duration: isVideo && outputMeta.duration ? formatTime(outputMeta.duration) : '',
  } : null

  useEffect(() => {
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [objectUrl])

  if (!selectedFile) {
    return (
      <div className="flex-1 flex flex-col">
        <div className="panel-toolbar justify-between">
          <h2 className="text-[13px] font-semibold text-[var(--color-text-primary)]">Preview</h2>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              No media loaded
            </Badge>
          </div>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center p-8">
          <div className="drop-zone w-full max-w-2xl">
            <input
              id="canvas-drop-input"
              type="file"
              accept="image/*,video/*"
              onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Sparkles className="w-12 h-12 text-[var(--color-border-strong)] mx-auto mb-4" />
            <h3 className="text-[15px] font-semibold text-[var(--color-text-primary)] mb-1.5">Drag & Drop Media Here</h3>
            <p className="text-[12px] text-[var(--color-text-muted)]">Supports JPG, PNG, MP4, MOV, AVI, MKV, WebM</p>
            <Button variant="secondary" className="mt-4" onClick={() => document.getElementById('canvas-drop-input')?.click()}>
              Browse Files
            </Button>
          </div>
        </div>
      </div>
    )
  }

  const handleSplitDrag = (e: MouseEvent) => {
    const container = (e.currentTarget as HTMLElement)?.parentElement
    if (!container) return
    const rect = container.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = Math.max(5, Math.min(95, (x / rect.width) * 100))
    setSplitPosition(percent)
  }

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || !isVideo) return
    const rect = e.currentTarget.getBoundingClientRect()
    const percent = (e.clientX - rect.left) / rect.width
    videoRef.current.currentTime = percent * videoRef.current.duration
    if (enhancedVideoRef.current) {
      enhancedVideoRef.current.currentTime = percent * enhancedVideoRef.current.duration
    }
  }

  function formatTime(seconds: number) {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 1000)
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`
  }

  const stepFrame = (direction: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime += direction * (1 / 30)
    }
    if (enhancedVideoRef.current) {
      enhancedVideoRef.current.currentTime += direction * (1 / 30)
    }
  }

  return (
    <div className="flex-1 flex flex-col relative h-full">
      {/* Video Metadata Header - Topaz Style */}
      <div className="panel-toolbar bg-[var(--color-bg-panel)] border-b border-[var(--color-border)] px-3">
        <div className="flex flex-wrap items-center gap-3 min-w-0">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <h2 className="text-[13px] font-medium text-[var(--color-text-primary)] truncate max-w-[300px]">
              {selectedFile.name}
            </h2>
            <Badge variant={isVideo ? 'info' : 'outline'} className="text-[10px]">
              {isVideo ? 'Video' : 'Image'}
            </Badge>
            {stepCount > 0 && (
              <Badge variant="success" className="text-[10px]">
                {stepCount} Filter{stepCount !== 1 ? 's' : ''}
              </Badge>
            )}
          </div>

          {inputMetaInfo && (
            <div className="flex items-center gap-2 text-[11px] font-mono text-[var(--color-text-secondary)] bg-[var(--color-bg-input)] px-2 py-1 rounded-[var(--radius-sm)] border border-[var(--color-border)]">
              <span className="text-[var(--color-text-muted)]">Input:</span>
              <span>{inputMetaInfo.resolution}</span>
              {inputMetaInfo.duration && (
                <>
                  <span className="text-[var(--color-border)]">|</span>
                  <span>{inputMetaInfo.duration}</span>
                </>
              )}
            </div>
          )}

          {outputMetaInfo && (
            <div className="flex items-center gap-2 text-[11px] font-mono text-emerald-400 bg-[var(--color-bg-input)] px-2 py-1 rounded-[var(--radius-sm)] border border-emerald-500/30">
              <span className="text-emerald-400">Output:</span>
              <span>{outputMetaInfo.resolution}</span>
              {outputMetaInfo.duration && (
                <>
                  <span className="text-[var(--color-border)]">|</span>
                  <span>{outputMetaInfo.duration}</span>
                </>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <div className="flex bg-[var(--color-bg-input)] border border-[var(--color-border)] rounded-[var(--radius-sm)] p-0.5" role="group" aria-label="View mode">
            {(['single', 'split', 'side-by-side', 'compare'] as ViewMode[]).map((mode) => (
              <Button
                key={mode}
                variant="ghost"
                size="sm"
                className={clsx(
                  'px-2 py-1.5 rounded-[var(--radius-xs)] transition-colors',
                  viewMode === mode && 'bg-[var(--color-accent)] text-white shadow-sm'
                )}
                onClick={() => setViewMode(mode)}
                aria-label={mode}
                aria-pressed={viewMode === mode}
              >
                {mode === 'single' && <span className="w-5 h-5">▢</span>}
                {mode === 'split' && <span className="w-5 h-5">⧖</span>}
                {mode === 'side-by-side' && <span className="w-5 h-5">◫</span>}
                {mode === 'compare' && <span className="w-5 h-5">⬚</span>}
              </Button>
            ))}
          </div>

          <Select
            value={zoomLevel}
            onChange={(e) => setZoomLevel(e.target.value as ZoomLevel)}
            options={[
              { value: 'fit', label: 'Fit' },
              { value: '100', label: '100%' },
              { value: '200', label: '200%' },
              { value: '400', label: '400%' },
            ]}
            className="w-[90px] text-[11px]"
          />

          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsFullscreen(!isFullscreen)}
            aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            className={isFullscreen ? 'bg-[var(--color-accent-muted)] text-[var(--color-accent)]' : ''}
          >
            {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
          </Button>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden bg-black" style={{ transform: zoomLevel !== 'fit' ? `scale(${parseInt(zoomLevel) / 100})` : 'none', transformOrigin: 'center center' }}>
        {viewMode === 'compare' ? (
          <div className="w-full h-full grid grid-rows-2 gap-1 p-1">
            <PreviewPane
              label="Original"
              isVideo={isVideo}
              src={objectUrl}
              videoRef={videoRef}
              isProcessing={false}
              onLoadedMetadata={(el) => setInputMeta(readMeta(el))}
            />
            <PreviewPane
              label={activeFiltersLabel}
              isVideo={isVideo}
              src={enhancedSrc}
              videoRef={enhancedVideoRef}
              isProcessing={isProcessing()}
              onLoadedMetadata={(el) => setOutputMeta(readMeta(el))}
            />
          </div>
        ) : (
          viewMode === 'split' ? (
            <div className="w-full h-full relative">
              <div className="w-full h-full flex relative">
                <div className="flex-1 relative overflow-hidden" style={{ width: `${splitPosition}%` }}>
                  <PreviewPane
                    label="Original"
                    isVideo={isVideo}
                    src={objectUrl}
                    videoRef={videoRef}
                    isProcessing={false}
                    onLoadedMetadata={(el) => setInputMeta(readMeta(el))}
                  />
                </div>
                <div
                  className="split-divider w-1 cursor-col-resize flex items-center justify-center"
                  onMouseDown={(e: React.MouseEvent) => {
                    e.preventDefault()
                    const handler = (evt: MouseEvent) => handleSplitDrag(evt)
                    document.addEventListener('mousemove', handler)
                    document.addEventListener('mouseup', () => document.removeEventListener('mousemove', handler), { once: true })
                  }}
                  style={{ background: 'linear-gradient(90deg, transparent, var(--color-accent), transparent)' }}
                >
                  <div className="w-px h-8 bg-white/50" />
                </div>
                <div className="flex-1 relative overflow-hidden" style={{ width: `${100 - splitPosition}%` }}>
                  <PreviewPane
                    label="Enhanced"
                    isVideo={isVideo}
                    src={enhancedSrc}
                    videoRef={enhancedVideoRef}
                    isProcessing={isProcessing()}
                    onLoadedMetadata={(el) => setOutputMeta(readMeta(el))}
                  />
                </div>
              </div>
            </div>
          ) : (
            viewMode === 'side-by-side' ? (
              <div className="w-full h-full flex">
                <div className="flex-1 relative overflow-hidden">
                  <PreviewPane
                    label="Original"
                    isVideo={isVideo}
                    src={objectUrl}
                    videoRef={videoRef}
                    isProcessing={false}
                    onLoadedMetadata={(el) => setInputMeta(readMeta(el))}
                  />
                </div>
                <div className="w-px bg-[var(--color-border)]" />
                <div className="flex-1 relative overflow-hidden">
                  <PreviewPane
                    label="Enhanced"
                    isVideo={isVideo}
                    src={enhancedSrc}
                    videoRef={enhancedVideoRef}
                    isProcessing={isProcessing()}
                    onLoadedMetadata={(el) => setOutputMeta(readMeta(el))}
                  />
                </div>
              </div>
            ) : (
              <div className="w-full h-full relative">
                <PreviewPane
                  label={hasResult ? 'Enhanced' : 'Original'}
                  isVideo={isVideo}
                  src={enhancedSrc}
                  videoRef={hasResult ? enhancedVideoRef : videoRef}
                  isProcessing={isProcessing()}
                  showControls={true}
                  onLoadedMetadata={(el) => (hasResult ? setOutputMeta(readMeta(el)) : setInputMeta(readMeta(el)))}
                />
              </div>
            )
          )
        )}

        {typeof isProcessing === 'function' ? isProcessing() : isProcessing && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
            <div className="text-center">
              <div className="w-12 h-12 border-3 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="text-white font-medium">Processing...</p>
              <p className="text-[var(--color-text-secondary)] text-sm mt-1">
                {currentTask?.message || 'Applying enhancements...'}
              </p>
            </div>
          </div>
        )}

        {typeof isProcessing === 'function' ? isProcessing() : isProcessing && (
          <div className="absolute bottom-4 left-4 right-4 flex justify-center z-10">
            <div className="bg-black/80 backdrop-blur px-6 py-3 rounded-[var(--radius-lg)] flex items-center gap-3 text-white max-w-md mx-auto">
              <div className="w-6 h-6 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
              <div>
                <p className="font-medium">Processing...</p>
                <p className="text-xs text-[var(--color-text-secondary)]">
                  {currentTask?.message || 'Applying enhancements...'}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {isVideo && (
        <div className="shrink-0 bg-[var(--color-bg-panel)] border-t border-[var(--color-border)]">
          <div className="relative h-16 px-3">
            <div 
              className="relative w-full h-full cursor-pointer"
              onClick={handleTimelineClick}
              onMouseDown={() => { videoRef.current?.pause(); enhancedVideoRef.current?.pause() }}
            >
              <div className="absolute inset-0 bg-[var(--color-bg-input)] rounded-[var(--radius-sm)] border border-[var(--color-border)]" />
              <div 
                className="absolute top-0 bottom-0 w-1 bg-emerald-500 pointer-events-none"
                style={{ left: `${trimIn}%` }}
              />
              <div 
                className="absolute top-0 bottom-0 w-1 bg-red-500 pointer-events-none"
                style={{ right: `${100 - trimOut}%` }}
              />
              {videoRef.current && (
                <div 
                  className="absolute top-0 bottom-0 w-px bg-white pointer-events-none"
                  style={{ left: `${(videoRef.current.currentTime / videoRef.current.duration) * 100}%` }}
                />
              )}
            </div>

            <div className="flex items-center justify-between mt-2 text-[11px] text-[var(--color-text-muted)]">
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={() => stepFrame(-1)} aria-label="Previous frame">
                  <RotateCcw className="w-3.5 h-3.5" />
                </Button>
                <Button variant="ghost" size="sm" onClick={() => stepFrame(1)} aria-label="Next frame">
                  <RotateCw className="w-3.5 h-3.5" />
                </Button>
                <span className="font-mono">
                  {videoRef.current ? formatTime(videoRef.current.currentTime) : '00:00:00.000'}
                </span>
                <span className="text-[var(--color-border)]">/</span>
                <span className="font-mono">
                  {videoRef.current ? formatTime(videoRef.current.duration) : '00:00:00.000'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-1 cursor-pointer">
                  <input type="checkbox" checked={showTrim} onChange={(e) => setShowTrim(e.target.checked)} className="w-3 h-3 accent-[var(--color-accent)]" />
                  Trim
                </label>
                {showTrim && (
                  <>
                    <span>In: {trimIn}%</span>
                    <span>Out: {trimOut}%</span>
                    <Button variant="ghost" size="sm" onClick={() => { setTrimIn(0); setTrimOut(100) }}>
                      <RotateCcw className="w-3.5 h-3.5" />
                    </Button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="px-4 py-2 border-t border-[var(--color-border)] bg-[var(--color-bg-panel)] shrink-0 flex items-center justify-between text-xs text-[var(--color-text-muted)]">
        <div className="flex items-center gap-4">
          <span className="font-mono">{selectedFile.size > 1024 * 1024 ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB` : `${(selectedFile.size / 1024).toFixed(1)} KB`}</span>
          <span className="text-[var(--color-border)]">|</span>
          <span>{isVideo ? 'Video' : 'Image'}</span>
        </div>
        <div className="flex items-center gap-2">
          {stepCount > 0 && (
            <Badge variant="success" className="text-xs">
              {stepCount} Filter{stepCount !== 1 ? 's' : ''} Ready
            </Badge>
          )}
        </div>
      </div>
    </div>
  )
}