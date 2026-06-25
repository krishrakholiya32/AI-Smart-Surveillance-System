import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { camerasApi, type Camera } from '../api/client'
import { useStore } from '../store'
import { Play, Square, Trash2, ExternalLink } from 'lucide-react'

export default function Cameras() {
  const cameras      = useStore(s => s.cameras)
  const setCameras   = useStore(s => s.setCameras)
  const clearMetrics = useStore(s => s.clearMetrics)
  const [running, setRunning] = useState<Record<number, boolean>>({})
  const [adding, setAdding]   = useState(false)
  const [form, setForm]       = useState({ name: '', source: 'http://host.docker.internal:8765/video' })

  const load = async () => {
    const list = await camerasApi.list()
    setCameras(list)
    const statuses = await Promise.all(list.map(c => camerasApi.status(c.id)))
    setRunning(Object.fromEntries(list.map((c, i) => [c.id, statuses[i].running])))
  }

  useEffect(() => { load() }, [])

  const addCamera = async (e: React.FormEvent) => {
    e.preventDefault()
    await camerasApi.create(form)
    setAdding(false)
    setForm({ name: '', source: 'http://host.docker.internal:8765/video' })
    load()
  }

  const toggleCamera = async (id: number) => {
    if (running[id]) {
      await camerasApi.stop(id)
      setRunning(r => ({ ...r, [id]: false }))
      clearMetrics(id)
    } else {
      await camerasApi.start(id)
      setRunning(r => ({ ...r, [id]: true }))
    }
  }

  const delCamera = async (id: number) => {
    if (!confirm('Delete this camera?')) return
    await camerasApi.delete(id)
    load()
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-cyber-cyan tracking-widest">CAMERAS</h2>
        <button onClick={() => setAdding(a => !a)}
          className="text-xs font-mono border border-cyber-cyan text-cyber-cyan px-4 py-1.5 rounded hover:bg-cyber-cyan/10 transition-colors">
          + ADD CAMERA
        </button>
      </div>

      {adding && (
        <form onSubmit={addCamera}
          className="bg-cyber-surface border border-cyber-border rounded-lg p-4 space-y-3">
          <p className="text-cyber-cyan text-xs font-mono tracking-widest">NEW CAMERA</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-mono text-cyber-muted mb-1">NAME</label>
              <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                placeholder="Front Entrance"
                className="w-full bg-cyber-bg border border-cyber-border rounded px-3 py-2 text-sm font-mono text-cyber-dim focus:outline-none focus:border-cyber-cyan" />
            </div>
            <div>
              <label className="block text-xs font-mono text-cyber-muted mb-1">SOURCE</label>
              <input value={form.source} onChange={e => setForm(f => ({ ...f, source: e.target.value }))}
                placeholder="http://host.docker.internal:8765/video | http://<phone-ip>:4747/video | rtsp://..."
                className="w-full bg-cyber-bg border border-cyber-border rounded px-3 py-2 text-sm font-mono text-cyber-dim focus:outline-none focus:border-cyber-cyan" />
            </div>
          </div>
          <div className="text-cyber-muted text-xs font-mono space-y-1">
            <p>Host webcam: run <code className="text-cyber-cyan">python scripts/webcam_server.py</code> on
              this machine, then use <code className="text-cyber-cyan">http://host.docker.internal:8765/video</code></p>
            <p>Phone camera: install the <span className="text-cyber-dim">DroidCam</span> app, connect phone +
              PC to the same Wi-Fi, then use <code className="text-cyber-cyan">http://&lt;phone-ip&gt;:4747/video</code></p>
            <p>RTSP/IP cameras: <code className="text-cyber-cyan">rtsp://user:pass@IP/stream</code></p>
            <p>Add as many cameras as you like — each runs its own independent detection worker.</p>
          </div>
          <div className="flex gap-2">
            <button type="submit"
              className="border border-cyber-green text-cyber-green text-xs font-mono px-4 py-1.5 rounded hover:bg-cyber-green/10">
              SAVE
            </button>
            <button type="button" onClick={() => setAdding(false)}
              className="border border-cyber-border text-cyber-muted text-xs font-mono px-4 py-1.5 rounded hover:border-cyber-red hover:text-cyber-red">
              CANCEL
            </button>
          </div>
        </form>
      )}

      {cameras.length === 0 ? (
        <div className="border border-dashed border-cyber-border rounded-lg p-12 text-center">
          <p className="text-cyber-muted font-mono text-sm">No cameras yet. Add one above.</p>
        </div>
      ) : (
        <div className="space-y-2">
          {cameras.map(cam => (
            <div key={cam.id}
              className="bg-cyber-surface border border-cyber-border rounded-lg px-4 py-3 flex items-center gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-sans font-semibold text-cyber-dim">{cam.name}</span>
                  <span className={`text-xs font-mono px-1.5 py-0.5 rounded border ${
                    running[cam.id]
                      ? 'text-cyber-cyan border-cyber-cyan/40 bg-cyber-cyan/10'
                      : 'text-cyber-muted border-cyber-border'
                  }`}>
                    {running[cam.id] ? 'LIVE' : 'IDLE'}
                  </span>
                </div>
                <p className="text-cyber-muted text-xs font-mono mt-0.5 truncate">{cam.source}</p>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => toggleCamera(cam.id)} title={running[cam.id] ? 'Stop' : 'Start'}
                  className={`p-2 rounded border transition-colors ${
                    running[cam.id]
                      ? 'border-cyber-red/40 text-cyber-red hover:bg-cyber-red/10'
                      : 'border-cyber-green/40 text-cyber-green hover:bg-cyber-green/10'
                  }`}>
                  {running[cam.id] ? <Square size={14} /> : <Play size={14} />}
                </button>
                <Link to={`/camera/${cam.id}`} title="View"
                  className="p-2 rounded border border-cyber-border text-cyber-muted hover:border-cyber-cyan hover:text-cyber-cyan transition-colors">
                  <ExternalLink size={14} />
                </Link>
                <button onClick={() => delCamera(cam.id)} title="Delete"
                  className="p-2 rounded border border-cyber-border text-cyber-muted hover:border-cyber-red hover:text-cyber-red transition-colors">
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
