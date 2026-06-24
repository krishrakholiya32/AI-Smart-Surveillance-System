import { useEffect, useState } from 'react'
import { alertsApi, type AlertRecord } from '../api/client'
import AlertBadge, { StatusBadge } from '../components/AlertBadge'
import { formatDistanceToNow } from 'date-fns'

export default function Alerts() {
  const [alerts, setAlerts]   = useState<AlertRecord[]>([])
  const [filter, setFilter]   = useState('')
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await alertsApi.list({ limit: 200, alert_type: filter || undefined })
      setAlerts(data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [filter])

  const del = async (id: number) => {
    await alertsApi.delete(id)
    setAlerts(prev => prev.filter(a => a.id !== id))
  }

  const setStatus = async (id: number, status: 'confirmed' | 'dismissed') => {
    const updated = await alertsApi.setStatus(id, status)
    setAlerts(prev => prev.map(a => a.id === id ? updated : a))
  }

  const TYPES = ['', 'weapon', 'zone_intrusion', 'running', 'loitering', 'fall', 'crowd']

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-cyber-cyan tracking-widest">ALERT HISTORY</h2>
        <span className="text-cyber-muted text-xs font-mono">{alerts.length} records</span>
      </div>

      {/* Filter */}
      <div className="flex gap-2 flex-wrap">
        {TYPES.map(t => (
          <button key={t} onClick={() => setFilter(t)}
            className={`text-xs font-mono px-3 py-1 rounded border transition-colors ${
              filter === t
                ? 'border-cyber-cyan text-cyber-cyan bg-cyber-cyan/10'
                : 'border-cyber-border text-cyber-muted hover:border-cyber-cyan/50'
            }`}>
            {t || 'ALL'}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="text-cyber-muted font-mono text-sm">Loading…</p>
      ) : alerts.length === 0 ? (
        <div className="border border-dashed border-cyber-border rounded-lg p-12 text-center">
          <p className="text-cyber-muted font-mono text-sm">No alerts found</p>
        </div>
      ) : (
        <div className="space-y-2">
          {alerts.map(a => (
            <div key={a.id}
              className="bg-cyber-surface border border-cyber-border rounded-lg p-4 flex items-start gap-4 hover:border-cyber-border/80 transition-colors">
              <div className="flex-1 min-w-0 space-y-1.5">
                <div className="flex items-center gap-2 flex-wrap">
                  <AlertBadge type={a.alert_type} />
                  <StatusBadge status={a.status} />
                  <span className="text-cyber-muted text-xs font-mono">CAM {a.camera_id}</span>
                  {a.person_id !== null && (
                    <span className="text-cyber-muted text-xs font-mono">Person #{a.person_id}</span>
                  )}
                </div>
                <p className="text-cyber-dim text-sm">{a.message}</p>
                <p className="text-cyber-muted text-xs font-mono">
                  {formatDistanceToNow(new Date(a.created_at), { addSuffix: true })}
                </p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {a.snapshot_path && (
                  <a href={alertsApi.snapshotUrl(a.id)} target="_blank" rel="noreferrer"
                    className="text-xs font-mono text-cyber-cyan hover:underline">
                    SNAPSHOT
                  </a>
                )}
                {a.status === 'pending' && (
                  <>
                    <button onClick={() => setStatus(a.id, 'confirmed')}
                      className="text-xs font-mono text-cyber-green hover:underline">
                      CONFIRM
                    </button>
                    <button onClick={() => setStatus(a.id, 'dismissed')}
                      className="text-xs font-mono text-cyber-muted hover:text-cyber-orange transition-colors">
                      DISMISS
                    </button>
                  </>
                )}
                <button onClick={() => del(a.id)}
                  className="text-xs font-mono text-cyber-muted hover:text-cyber-red transition-colors">
                  DELETE
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
