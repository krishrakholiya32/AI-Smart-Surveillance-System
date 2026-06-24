import { useEffect, useState } from 'react'
import { eventsApi, type EventRecord } from '../api/client'
import { formatDistanceToNow } from 'date-fns'

const LEVEL_COLOR: Record<string, string> = {
  alert: 'text-cyber-red',
  error: 'text-cyber-orange',
  info:  'text-cyber-muted',
}

export default function Events() {
  const [events, setEvents] = useState<EventRecord[]>([])
  const [level, setLevel]   = useState('')

  useEffect(() => {
    eventsApi.list({ limit: 200, level: level || undefined }).then(setEvents)
  }, [level])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-cyber-cyan tracking-widest">EVENT LOG</h2>
        <span className="text-cyber-muted text-xs font-mono">{events.length} entries</span>
      </div>
      <div className="flex gap-2">
        {['', 'alert', 'info', 'error'].map(l => (
          <button key={l} onClick={() => setLevel(l)}
            className={`text-xs font-mono px-3 py-1 rounded border transition-colors ${
              level === l ? 'border-cyber-cyan text-cyber-cyan bg-cyber-cyan/10' : 'border-cyber-border text-cyber-muted'
            }`}>
            {l || 'ALL'}
          </button>
        ))}
      </div>
      <div className="bg-cyber-surface border border-cyber-border rounded-lg overflow-hidden">
        {events.length === 0 ? (
          <p className="text-cyber-muted font-mono text-sm text-center p-8">No events</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-cyber-border">
                <th className="text-left text-cyber-muted font-mono text-xs px-4 py-2">TIME</th>
                <th className="text-left text-cyber-muted font-mono text-xs px-4 py-2">LEVEL</th>
                <th className="text-left text-cyber-muted font-mono text-xs px-4 py-2">CAM</th>
                <th className="text-left text-cyber-muted font-mono text-xs px-4 py-2">MESSAGE</th>
              </tr>
            </thead>
            <tbody>
              {events.map(e => (
                <tr key={e.id} className="border-b border-cyber-border/50 hover:bg-white/2 transition-colors">
                  <td className="px-4 py-2 text-cyber-muted font-mono text-xs whitespace-nowrap">
                    {formatDistanceToNow(new Date(e.created_at), { addSuffix: true })}
                  </td>
                  <td className={`px-4 py-2 font-mono text-xs ${LEVEL_COLOR[e.level] ?? 'text-cyber-muted'}`}>
                    {e.level.toUpperCase()}
                  </td>
                  <td className="px-4 py-2 text-cyber-muted font-mono text-xs">{e.camera_id ?? '-'}</td>
                  <td className="px-4 py-2 text-cyber-dim text-sm">{e.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
