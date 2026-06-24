import { useVideoStream } from '../hooks/useVideoStream'
import { useStore } from '../store'

interface Props {
  cameraId: number
  cameraName?: string
  compact?: boolean
}

export default function VideoFeed({ cameraId, cameraName, compact }: Props) {
  const { frame, connected } = useVideoStream(cameraId)
  const metrics = useStore(s => s.metrics[cameraId])

  return (
    <div className="relative bg-cyber-surface rounded border border-cyber-border overflow-hidden">
      {/* Status badge */}
      <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
        <span className={`font-mono text-xs px-2 py-0.5 rounded border ${
          connected
            ? 'bg-cyber-cyan/10 border-cyber-cyan/40 text-cyber-cyan'
            : 'bg-white/5 border-cyber-border text-cyber-muted'
        }`}>
          {connected ? '● LIVE' : '◌ OFFLINE'}
        </span>
        {cameraName && (
          <span className="font-mono text-xs text-cyber-muted bg-cyber-bg/60 px-2 py-0.5 rounded">
            {cameraName}
          </span>
        )}
      </div>

      {/* Video */}
      {frame ? (
        <img src={frame} alt="live feed" className="w-full h-auto block" />
      ) : (
        <div className={`flex items-center justify-center text-cyber-muted font-mono text-sm ${compact ? 'h-48' : 'h-96'}`}>
          {connected ? 'Waiting for frames…' : 'Camera not streaming'}
        </div>
      )}

      {/* Metrics overlay (compact mode shows fewer) */}
      {metrics && !compact && (
        <div className="absolute bottom-0 left-0 right-0 bg-cyber-bg/70 backdrop-blur px-3 py-2 flex gap-4 text-xs font-mono">
          <span className="text-cyber-cyan">P: {metrics.people}</span>
          <span className={metrics.gun > 0 ? 'text-cyber-red' : 'text-cyber-muted'}>GUN: {metrics.gun}</span>
          <span className={metrics.knife > 0 ? 'text-cyber-red' : 'text-cyber-muted'}>KNIFE: {metrics.knife}</span>
          <span className={metrics.running > 0 ? 'text-cyber-orange' : 'text-cyber-muted'}>RUN: {metrics.running}</span>
          <span className={metrics.loiter > 0 ? 'text-cyber-orange' : 'text-cyber-muted'}>LOIT: {metrics.loiter}</span>
          <span className={metrics.fall > 0 ? 'text-cyber-red' : 'text-cyber-muted'}>FALL: {metrics.fall}</span>
          <span className="ml-auto text-cyber-muted">{metrics.fps} FPS</span>
        </div>
      )}
    </div>
  )
}
