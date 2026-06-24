import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useStore } from '../store'
import { camerasApi } from '../api/client'
import VideoFeed from '../components/VideoFeed'
import MetricsCard from '../components/MetricsCard'

export default function Dashboard() {
  const cameras      = useStore(s => s.cameras)
  const setCameras   = useStore(s => s.setCameras)
  const metrics      = useStore(s => s.metrics)
  const totalAlerts  = useStore(s => s.totalAlerts)

  useEffect(() => {
    camerasApi.list().then(setCameras).catch(console.error)
  }, [setCameras])

  const totals = Object.values(metrics).reduce(
    (acc, m) => ({
      people: acc.people + m.people,
      gun:    acc.gun    + m.gun,
      knife:  acc.knife  + m.knife,
      fall:   acc.fall   + m.fall,
    }),
    { people: 0, gun: 0, knife: 0, fall: 0 }
  )

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-cyber-cyan tracking-widest">DASHBOARD</h2>
        <Link to="/cameras"
          className="text-xs font-mono border border-cyber-border text-cyber-muted px-3 py-1.5 rounded hover:border-cyber-cyan hover:text-cyber-cyan transition-colors">
          MANAGE CAMERAS
        </Link>
      </div>

      {/* Global metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricsCard label="PEOPLE" value={totals.people} />
        <MetricsCard label="WEAPONS" value={totals.gun + totals.knife} danger={totals.gun + totals.knife > 0} />
        <MetricsCard label="FALLS" value={totals.fall} danger={totals.fall > 0} />
        <MetricsCard label="TOTAL ALERTS" value={totalAlerts} warn={totalAlerts > 0} />
      </div>

      {/* Camera grid */}
      {cameras.length === 0 ? (
        <div className="border border-dashed border-cyber-border rounded-lg p-12 text-center">
          <p className="font-mono text-cyber-muted text-sm">No cameras configured yet.</p>
          <Link to="/cameras" className="mt-3 inline-block text-cyber-cyan text-sm font-mono hover:underline">
            + Add your first camera
          </Link>
        </div>
      ) : (
        <div className={`grid gap-4 ${cameras.length === 1 ? 'grid-cols-1' : cameras.length <= 4 ? 'grid-cols-2' : 'grid-cols-3'}`}>
          {cameras.map(cam => (
            <div key={cam.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-cyber-dim text-sm">{cam.name}</span>
                <Link to={`/camera/${cam.id}`}
                  className="text-xs text-cyber-cyan hover:underline font-mono">
                  EXPAND →
                </Link>
              </div>
              <VideoFeed cameraId={cam.id} cameraName={cam.name} compact />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
