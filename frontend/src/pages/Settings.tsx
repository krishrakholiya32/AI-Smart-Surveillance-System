export default function Settings() {
  return (
    <div className="space-y-6">
      <h2 className="font-mono text-cyber-cyan tracking-widest">SETTINGS</h2>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-cyber-surface border border-cyber-border rounded-lg p-5 space-y-4">
          <h3 className="font-mono text-cyber-dim text-sm tracking-wider">DETECTION THRESHOLDS</h3>
          <p className="text-cyber-muted text-xs">
            Configure thresholds in the backend <code className="text-cyber-cyan">.env</code> file.
            Restart the backend service after changes.
          </p>
          {[
            ['Person confidence',   'CONF_PERSON',     '0.50'],
            ['Weapon confidence',   'CONF_WEAPON',     '0.40'],
            ['Run speed threshold', 'RUN_THRESH_NORM', '0.18'],
            ['Loiter seconds',      'LOITER_SECS',     '8'],
            ['Crowd limit',         'CROWD_LIMIT',     '5'],
          ].map(([label, key, def]) => (
            <div key={key} className="flex justify-between text-sm">
              <span className="text-cyber-muted">{label}</span>
              <span className="font-mono text-cyber-cyan">{key} = <span className="text-cyber-green">{def}</span></span>
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
            Export models to ONNX for 2–3× faster CPU inference. Run from the <code className="text-cyber-cyan">scripts/</code> directory.
          </p>
          <pre className="bg-cyber-bg rounded p-3 text-xs font-mono text-cyber-cyan overflow-x-auto">
{`# In your Python environment with ultralytics installed:
python scripts/export_onnx.py`}
          </pre>
        </div>
      </div>
    </div>
  )
}
