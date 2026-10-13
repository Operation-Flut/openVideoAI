import { useState, useEffect } from 'react';

export class SSEService {
  private eventSource: EventSource | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  connect(url: string, onMessage: (data: any) => void, onError?: (error: Event) => void): void {
    if (this.eventSource) {
      this.eventSource.close();
    }
    
    this.eventSource = new EventSource(url);
    
    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (err) {
        console.error('Failed to parse SSE message:', err);
      }
    };
    
    this.eventSource.onerror = (error) => {
      console.error('SSE error:', error);
      this.reconnectAttempts++;
      
      if (this.reconnectAttempts <= this.maxReconnectAttempts) {
        console.log(`Reconnecting in ${this.reconnectDelay}ms... (attempt ${this.reconnectAttempts})`);
        setTimeout(() => {
          if (this.eventSource) {
            this.eventSource.close();
          }
          this.connect(url, onMessage, onError);
        }, this.reconnectDelay);
        this.reconnectDelay *= 2;
      } else {
        console.error('Max reconnect attempts reached');
        onError?.(error);
      }
    };
  }
  
  disconnect(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
    }
  }
  
  isConnected(): boolean {
    return this.eventSource !== null && this.eventSource.readyState === EventSource.OPEN;
  }
}

export const sseService = new SSEService();

export const useSSE = (url: string | null, onMessage: (data: any) => void) => {
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    if (!url) return;
    
    const handleOpen = () => setIsConnected(true);
    const handleError = () => setIsConnected(false);
    
    const eventSource = new EventSource(url);
    
    eventSource.onopen = handleOpen;
    eventSource.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch (err) {
        console.error('Failed to parse SSE message:', err);
      }
    };
    eventSource.onerror = handleError;
    
    return () => eventSource.close();
  }, [url, onMessage]);
  
  return isConnected;
};