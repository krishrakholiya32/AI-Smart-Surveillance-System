import { useEffect, useState } from 'react'
import { settingsApi, type SettingsInfo } from '../api/client'

const THRESHOLD_LABELS: [string, keyof SettingsInfo['thresholds']][] = [
  ['Person confidence',   'CONF_PERSON'],
  ['Weapon confidence',   'CONF_WEAPON'],
  ['Run speed threshold', 'RUN_THRESH_NORM'],
  ['Loiter seconds',      'LOITER_SECS'],
  ['Crowd limit',         'CROWD_LIMIT'],
]

const ONNX_LABELS: [string, string][] = [
  ['person', 'Person detection (YOLOv11s)'],
  ['pose',   'Pose estimation (YOLOv11n-pose)'],
  ['weapon', 'Weapon detection (custom YOLOv11)'],
]

export default function Settings() {
  const [info, setInfo] = useState<SettingsInfo | null>(null)

  useEffect(() => {
    settingsApi.get().then(setInfo).catch(console.error)
  }, [])

  return (
    <div className="space-y-6">
      <h2 className="font-mono text-cyber-cyan tracking-widest">SETTINGS</h2>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-cyber-surface border border-cyber-border rounded-lg p-5 space-y-4">
          <h3 className="font-mono text-cyber-dim text-sm tracking-wider">DETECTION THRESHOLDS</h3>
          <p className="text-cyber-muted text-xs">
            Live values from the backend's <code className="text-cyber-cyan">.env</code>.
            To change them, edit <code className="text-cyber-cyan">.env</code> and restart the backend service.
          </p>
          {THRESHOLD_LABELS.map(([label, key]) => (
            <div key={key} className="flex justify-between text-sm">
              <span className="text-cyber-muted">{label}</span>
              <span className="font-mono text-cyber-cyan">
                {key} = <span className="text-cyber-green">{info ? info.thresholds[key] : '…'}</span>
              </span>
            </div>
          ))}
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
            ['Fall detection',         '✓', 'Pose angle analysis'],
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
