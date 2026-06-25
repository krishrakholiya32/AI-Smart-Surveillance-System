import { useEffect, useRef, useCallback, useState } from 'react'
import { useStore } from '../store'

interface StreamPayload {
  camera_id: number
  frame: string        // base64 JPEG
  metrics: {
    people: number; gun: number; knife: number
    running: number; loiter: number; fall: number; fps: number
  }
  alerts: { type: string; message: string; person_id?: number }[]
  ts: number
}

export function useVideoStream(cameraId: number | null) {
  const [frame, setFrame] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const token = useStore(s => s.token)
  const updateMetrics = useStore(s => s.updateMetrics)
  const bumpAlertCount = useStore(s => s.bumpAlertCount)

  const connect = useCallback(() => {
    if (!cameraId || !token) return
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host  = window.location.host
    const url   = `${proto}://${host}/ws/${cameraId}?token=${token}`
    const ws    = new WebSocket(url)
    wsRef.current = ws

    ws.onopen  = () => setConnected(true)
    ws.onclose = () => { setConnected(false); setFrame(null) }
    ws.onerror = () => ws.close()

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data) as StreamPayload | { type: 'ping' }
      if ('type' in data && data.type === 'ping') return
      const payload = data as StreamPayload
      setFrame(`data:image/jpeg;base64,${payload.frame}`)
      updateMetrics(payload.camera_id, payload.metrics)
      if (payload.alerts?.length) bumpAlertCount()
    }
  }, [cameraId, token, updateMetrics, bumpAlertCount])

  useEffect(() => {
    connect()
    return () => wsRef.current?.close()
  }, [connect])

  const clearFrame = useCallback(() => setFrame(null), [])

  return { frame, connected, clearFrame }
}
