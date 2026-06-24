import { useRef, useState, useEffect, useCallback } from 'react'
import type { Zone } from '../api/client'

interface Props {
  width: number
  height: number
  existingZones: Zone[]
  onSave: (points: number[][], name: string, color: string) => void
  onDelete: (zoneId: number) => void
}

const COLORS = ['#0050dc', '#00ff99', '#ffaa00', '#ff4b4b', '#cc44ff']

export default function ZoneCanvas({ width, height, existingZones, onSave, onDelete }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [points, setPoints] = useState<number[][]>([])
  const [drawing, setDrawing] = useState(false)
  const [zoneName, setZoneName] = useState('Zone 1')
  const [color, setColor] = useState(COLORS[0])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, width, height)

    // Draw existing zones
    existingZones.forEach(zone => {
      if (zone.points.length < 3) return
      ctx.beginPath()
      ctx.moveTo(zone.points[0][0], zone.points[0][1])
      zone.points.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.closePath()
      const hex = zone.color.replace('#', '')
      const r = parseInt(hex.slice(0, 2), 16)
      const g = parseInt(hex.slice(2, 4), 16)
      const b = parseInt(hex.slice(4, 6), 16)
      ctx.fillStyle = `rgba(${r},${g},${b},0.15)`
      ctx.fill()
      ctx.strokeStyle = zone.color
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.fillStyle = zone.color
      ctx.font = '12px Share Tech Mono'
      const cx = zone.points.reduce((s, p) => s + p[0], 0) / zone.points.length
      const cy = zone.points.reduce((s, p) => s + p[1], 0) / zone.points.length
      ctx.fillText(zone.name, cx - 20, cy)
    })

    // Draw current polygon in progress
    if (points.length > 0) {
      ctx.beginPath()
      ctx.moveTo(points[0][0], points[0][1])
      points.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([5, 4])
      ctx.stroke()
      ctx.setLineDash([])
      points.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, Math.PI * 2)
        ctx.fillStyle = color
        ctx.fill()
      })
    }
  }, [points, existingZones, color, width, height])

  useEffect(() => { draw() }, [draw])

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!drawing) return
    const rect = canvasRef.current!.getBoundingClientRect()
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY
    setPoints(p => [...p, [Math.round(x), Math.round(y)]])
  }

  const handleDblClick = () => {
    if (points.length >= 3) {
      onSave(points, zoneName, color)
      setPoints([])
      setDrawing(false)
    }
  }

  const reset = () => { setPoints([]); setDrawing(false) }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 flex-wrap">
        <input
          value={zoneName}
          onChange={e => setZoneName(e.target.value)}
          className="bg-cyber-bg border border-cyber-border rounded px-3 py-1.5 text-sm font-mono text-cyber-cyan w-36"
          placeholder="Zone name"
        />
        <div className="flex gap-1">
          {COLORS.map(c => (
            <button key={c} onClick={() => setColor(c)}
              className={`w-6 h-6 rounded border-2 ${color === c ? 'border-white' : 'border-transparent'}`}
              style={{ background: c }} />
          ))}
        </div>
        {!drawing ? (
          <button onClick={() => setDrawing(true)}
            className="px-4 py-1.5 border border-cyber-cyan text-cyber-cyan text-sm font-mono rounded hover:bg-cyber-cyan/10 transition-colors">
            + DRAW ZONE
          </button>
        ) : (
          <>
            <span className="text-cyber-muted text-xs font-mono">Click to add points • Double-click to finish</span>
            <button onClick={reset} className="px-3 py-1.5 border border-cyber-red text-cyber-red text-sm font-mono rounded hover:bg-cyber-red/10">
              CANCEL
            </button>
          </>
        )}
      </div>

      <div className="relative w-full">
        <div className="bg-cyber-bg rounded border border-cyber-border text-cyber-muted font-mono text-sm flex items-center justify-center"
          style={{ height: `${height * 0.6}px` }}>
          <span className="text-xs opacity-50">Video frame reference</span>
        </div>
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onClick={handleClick}
          onDoubleClick={handleDblClick}
          className="absolute inset-0 w-full h-full"
          style={{ cursor: drawing ? 'crosshair' : 'default' }}
        />
      </div>

      {existingZones.length > 0 && (
        <div className="space-y-1">
          <p className="text-cyber-muted text-xs font-mono">ACTIVE ZONES</p>
          {existingZones.map(z => (
            <div key={z.id} className="flex items-center gap-2 text-sm">
              <span className="w-3 h-3 rounded" style={{ background: z.color }} />
              <span className="text-cyber-dim font-mono">{z.name}</span>
              <span className="text-cyber-muted text-xs">({z.points.length} pts)</span>
              <button onClick={() => onDelete(z.id)}
                className="ml-auto text-cyber-red/60 hover:text-cyber-red text-xs font-mono">
                DELETE
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
