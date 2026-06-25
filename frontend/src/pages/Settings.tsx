import { useEffect, useState } from 'react'
import { settingsApi, type SettingsInfo } from '../api/client'

type ThresholdKey = keyof SettingsInfo['thresholds']

const THRESHOLD_LABELS: [string, ThresholdKey, { step: number; integer?: boolean }][] = [
  ['Person confidence',   'CONF_PERSON',     { step: 0.05 }],
  ['Weapon confidence',   'CONF_WEAPON',     { step: 0.05 }],
  ['Run speed threshold', 'RUN_THRESH_NORM', { step: 0.01 }],
  ['Loiter seconds',      'LOITER_SECS',     { step: 1, integer: true }],
  ['Crowd limit',         'CROWD_LIMIT',     { step: 1, integer: true }],
]

const ONNX_LABELS: [string, string][] = [
  ['person', 'Person detection (YOLOv11s)'],
  ['pose',   'Pose estimation (YOLOv11n-pose)'],
  ['weapon', 'Weapon detection (custom YOLOv11)'],
]

export default function Settings() {
  const [info, setInfo] = useState<SettingsInfo | null>(null)
  const [draft, setDraft] = useState<SettingsInfo['thresholds'] | null>(null)
  const [saving, setSaving] = useState(false)
  const [savedAt, setSavedAt] = useState<number | null>(null)

  useEffect(() => {
    settingsApi.get().then(i => { setInfo(i); setDraft(i.thresholds) }).catch(console.error)
  }, [])

  const dirty = !!info && !!draft && THRESHOLD_LABELS.some(([, key]) => info.thresholds[key] !== draft[key])

  const save = async () => {
    if (!draft) return
    setSaving(true)
    try {
      const res = await settingsApi.updateThresholds(draft)
      setInfo(prev => prev ? { ...prev, thresholds: res.thresholds } : prev)
      setDraft(res.thresholds)
      setSavedAt(Date.now())
    } finally {
      setSaving(false)
    }
  }

  const reset = async () => {
    setSaving(true)
    try {
      const res = await settingsApi.resetThresholds()
      setInfo(prev => prev ? { ...prev, thresholds: res.thresholds } : prev)
      setDraft(res.thresholds)
      setSavedAt(Date.now())
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="font-mono text-cyber-cyan tracking-widest">SETTINGS</h2>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-cyber-surface border border-cyber-border rounded-lg p-5 space-y-4">
          <h3 className="font-mono text-cyber-dim text-sm tracking-wider">DETECTION THRESHOLDS</h3>
          <p className="text-cyber-muted text-xs">
            Changes apply immediately to all running cameras — no restart needed.
          </p>
          {THRESHOLD_LABELS.map(([label, key, opts]) => (
            <div key={key} className="flex items-center justify-between gap-3 text-sm">
              <span className="text-cyber-muted">{label}</span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-cyber-muted text-xs">{key}</span>
                <input
                  type="number"
                  step={opts.step}
                  value={draft ? draft[key] : ''}
                  onChange={e => {
                    const raw = e.target.value
                    setDraft(d => d && { ...d, [key]: raw === '' ? 0 : (opts.integer ? parseInt(raw, 10) : parseFloat(raw)) })
                  }}
                  disabled={!draft || saving}
                  className="w-20 bg-cyber-bg border border-cyber-border rounded px-2 py-1 text-sm font-mono text-cyber-green focus:outline-none focus:border-cyber-cyan disabled:opacity-50"
                />
              </div>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-2">
            <button onClick={save} disabled={!dirty || saving}
              className="border border-cyber-cyan text-cyber-cyan text-xs font-mono px-4 py-1.5 rounded hover:bg-cyber-cyan/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
              {saving ? 'SAVING…' : 'SAVE'}
            </button>
            <button onClick={reset} disabled={saving}
              className="border border-cyber-border text-cyber-muted text-xs font-mono px-4 py-1.5 rounded hover:border-cyber-red hover:text-cyber-red disabled:opacity-40 transition-colors">
              RESET TO DEFAULT
            </button>
            {savedAt && !dirty && (
              <span className="text-cyber-green text-xs font-mono">✓ saved</span>
            )}
          </div>
        </div>

        <div className="bg-cyber-surface border border-cyber-border rounded-lg p-5 space-y-4">
          <h3 className="font-mono text-cyber-dim text-sm tracking-wider">DETECTION CAPABILITIES</h3>
          {[
            ['Person detection',       '✓', 'YOLOv11s — COCO pretrained'],
            ['Pose estimation',        '✓', 'YOLOv11n-pose — 17 keypoints'],
            ['Weapon detection',       '✓', 'Custom YOLOv11 — Gun & Knife'],
            ['Zone intrusion',         '✓', 'Keypoint-aware polygon zones'],
            ['Running detection',      '✓', 'EMA speed-based'],
            ['Loitering detection',    '✓', 'Dwell-time based'],
            ['Crowd detection',        '✓', 'Configurable people limit'],
            ['Person re-identification','✓', 'Centroid + velocity + IoU'],
          ].map(([feat, status, desc]) => (
            <div key={feat} className="flex items-start gap-2 text-sm">
              <span className="text-cyber-green font-mono text-xs mt-0.5">{status}</span>
              <div>
                <span className="text-cyber-dim">{feat}</span>
                <span className="text-cyber-muted text-xs ml-2">{desc}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="bg-cyber-surface border border-cyber-border rounded-lg p-5 space-y-3 md:col-span-2">
          <h3 className="font-mono text-cyber-dim text-sm tracking-wider">ONNX MODEL OPTIMIZATION</h3>
          <p className="text-cyber-muted text-xs">
            ONNX exports are auto-preferred over <code className="text-cyber-cyan">.pt</code> checkpoints whenever
            present in <code className="text-cyber-cyan">models/</code> — nothing to run manually.
          </p>
          <div className="space-y-1.5">
            {ONNX_LABELS.map(([key, label]) => {
              const active = info?.onnx_active[key]
              return (
                <div key={key} className="flex items-center gap-2 text-sm">
                  <span className={`font-mono text-xs px-1.5 py-0.5 rounded border ${
                    active === undefined
                      ? 'text-cyber-muted border-cyber-border'
                      : active
                        ? 'text-cyber-green border-cyber-green/40 bg-cyber-green/10'
                        : 'text-cyber-orange border-cyber-orange/40 bg-cyber-orange/10'
                  }`}>
                    {active === undefined ? '…' : active ? 'ONNX ACTIVE' : 'USING .PT'}
                  </span>
                  <span className="text-cyber-dim">{label}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
