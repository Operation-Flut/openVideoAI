// SSE (Server-Sent Events) service for real-time updates

type EventHandler = (event: MessageEvent) => void

interface SSEOptions {
  onMessage?: (data: any) => void
  onError?: (error: Event) => void
  onOpen?: () => void
  reconnect?: boolean
  reconnectInterval?: number
}

export class SSEClient {
  private eventSource: EventSource | null = null
  private url: string
  private options: SSEOptions
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5

  constructor(url: string, options: SSEOptions = {}) {
    this.url = url
    this.options = {
      reconnect: true,
      reconnectInterval: 3000,
      ...options,
    }
  }

  connect(): void {
    if (this.eventSource) {
      this.disconnect()
    }

    try {
      this.eventSource = new EventSource(this.url)

      this.eventSource.onopen = () => {
        this.reconnectAttempts = 0
        this.options.onOpen?.()
      }

      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.options.onMessage?.(data)
        } catch (e) {
          console.warn('Failed to parse SSE message:', event.data)
        }
      }

      this.eventSource.onerror = (error) => {
        this.options.onError?.(error)
        this.handleError()
      }
    } catch (e) {
      console.error('Failed to create EventSource:', e)
      this.handleError()
    }
  }

  private handleError(): void {
    if (this.options.reconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      setTimeout(() => this.connect(), this.options.reconnectInterval)
    } else {
      this.disconnect()
    }
  }

  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
  }

  // Add event listener for specific event types
  addEventListener(type: string, handler: EventHandler): void {
    this.eventSource?.addEventListener(type, handler)
  }

  removeEventListener(type: string, handler: EventHandler): void {
    this.eventSource?.removeEventListener(type, handler)
  }
}

// React hook for SSE
import { useEffect, useRef } from 'react'

export function useSSE(
  url: string | null,
  onMessage: (data: any) => void,
  options: Omit<SSEOptions, 'onMessage'> = {}
) {
  const clientRef = useRef<SSEClient | null>(null)
  const onMessageRef = useRef(onMessage)

  onMessageRef.current = onMessage

  useEffect(() => {
    if (!url) return

    const client = new SSEClient(url, {
      ...options,
      onMessage: (data) => onMessageRef.current(data),
    })
    clientRef.current = client
    client.connect()

    return () => {
      client.disconnect()
      clientRef.current = null
    }
  }, [url, options.reconnect, options.reconnectInterval])

  return clientRef.current
}

// Processing progress types
export interface ProcessingProgressEvent {
  type: 'processing.started' | 'processing.progress' | 'processing.completed' | 'processing.failed' | 'processing.aborted'
  task_id: string
  progress: number
  message: string
  current_frame?: number
  total_frames?: number
}

export interface ModelDownloadProgressEvent {
  type: 'model.download.started' | 'model.download.progress' | 'model.download.completed' | 'model.download.failed'
  model_name: string
  progress: number
  downloaded_bytes: number
  total_bytes: number
  speed_bps: number
  error?: string
}