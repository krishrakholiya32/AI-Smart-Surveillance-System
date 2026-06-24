import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { camerasApi, zonesApi, type Camera, type Zone } from '../api/client'
import VideoFeed from '../components/VideoFeed'
import ZoneCanvas from '../components/ZoneCanvas'
import MetricsCard from '../components/MetricsCard'
import { useStore } from '../store'

type Tab = 'live' | 'zones' | 'metrics'

export default function CameraView() {
  const { id } = useParams<{ id: string }>()
  const cameraId = Number(id)

  const [camera, setCamera]   = useState<Camera | null>(null)
  const [zones, setZones]     = useState<Zone[]>([])
  const [tab, setTab]         = useState<Tab>('live')
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(false)

  const metrics = useStore(s => s.metrics[cameraId])

  useEffect(() => {
    camerasApi.get(cameraId).then(setCamera).catch(console.error)
    zonesApi.list(cameraId).then(setZones).catch(console.error)
    camerasApi.status(cameraId).then(s => setRunning(s.running)).catch(console.error)
  }, [cameraId])

  const toggleStream = async () => {
    setLoading(true)
    try {
      if (running) {
        await camerasApi.stop(cameraId)
        setRunning(false)
      } else {
        await camerasApi.start(cameraId)
        setRunning(true)
      }
    } finally {
      setLoading(false)
    }
  }

  const saveZone = async (points: number[][], name: string, color: string) => {
    const z = await zonesApi.create({ camera_id: cameraId, name, points, color })
    setZones(prev => [...prev, z])
  }

  const deleteZone = async (zoneId: number) => {
    await zonesApi.delete(zoneId)
    setZones(prev => prev.filter(z => z.id !== zoneId))
  }

  const TABS: { key: Tab; label: string }[] = [
    { key: 'live',    label: 'LIVE FEED' },
    { key: 'zones',   label: 'ZONE SETUP' },
    { key: 'metrics', label: 'METRICS' },
  ]

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-mono text-cyber-cyan tracking-widest">
            {camera?.name ?? `CAMERA ${cameraId}`}
          </h2>
          <p className="text-cyber-muted text-xs font-mono mt-0.5">{camera?.source}</p>
        </div>
        <button
          onClick={toggleStream}
          disabled={loading}
          className={`font-mono text-sm px-6 py-2 rounded border transition-colors ${
            running
              ? 'border-cyber-red text-cyber-red hover:bg-cyber-red/10'
              : 'border-cyber-green text-cyber-green hover:bg-cyber-green/10'
          } disabled:opacity-50`}
        >
          {loading ? '…' : running ? '■ STOP' : '▶ START'}
        </button>
      </div>

      {/* Tab nav */}
      <div className="flex border-b border-cyber-border">
        {TABS.map(({ key, label }) => (
          <button key={key} onClick={() => setTab(key)}
            className={`px-4 py-2 text-xs font-mono tracking-wider transition-colors ${
              tab === key
                ? 'text-cyber-cyan border-b-2 border-cyber-cyan -mb-px'
                : 'text-cyber-muted hover:text-cyber-dim'
            }`}>
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'live' && (
        <div className="space-y-4">
          <VideoFeed cameraId={cameraId} cameraName={camera?.name} />
          {!running && (
            <p className="text-center text-cyber-muted text-sm font-mono">
              Press ▶ START to begin detection
            </p>
          )}
        </div>
      )}

      {tab === 'zones' && (
        <div className="bg-cyber-surface border border-cyber-border rounded-lg p-4">
          <p className="text-cyber-muted text-xs font-mono mb-4">
            Draw restricted zones — alerts fire when people enter these areas.
          </p>
          <ZoneCanvas
            width={camera?.width ?? 640}
            height={camera?.height ?? 480}
            existingZones={zones}
            onSave={saveZone}
            onDelete={deleteZone}
          />
        </div>
      )}

      {tab === 'metrics' && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricsCard label="PEOPLE" value={metrics?.people ?? 0} />
          <MetricsCard label="GUNS" value={metrics?.gun ?? 0} danger={(metrics?.gun ?? 0) > 0} />
          <MetricsCard label="KNIVES" value={metrics?.knife ?? 0} danger={(metrics?.knife ?? 0) > 0} />
          <MetricsCard label="RUNNING" value={metrics?.running ?? 0} warn={(metrics?.running ?? 0) > 0} />
          <MetricsCard label="LOITERING" value={metrics?.loiter ?? 0} warn={(metrics?.loiter ?? 0) > 0} />
          <MetricsCard label="FALLS" value={metrics?.fall ?? 0} danger={(metrics?.fall ?? 0) > 0} />
          <MetricsCard label="FPS" value={metrics?.fps ?? 0} />
        </div>
      )}
    </div>
  )
}
