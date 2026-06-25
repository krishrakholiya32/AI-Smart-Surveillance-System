const TYPE_COLOR: Record<string, string> = {
  weapon:        'bg-cyber-red/20 text-cyber-red border-cyber-red/40',
  zone_intrusion:'bg-cyber-orange/20 text-cyber-orange border-cyber-orange/40',
  running:       'bg-blue-500/20 text-blue-400 border-blue-500/40',
  loitering:     'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  crowd:         'bg-cyan-500/20 text-cyber-cyan border-cyber-cyan/40',
}

const STATUS_COLOR: Record<string, string> = {
  pending:   'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  confirmed: 'bg-cyber-green/20 text-cyber-green border-cyber-green/40',
  dismissed: 'bg-white/10 text-cyber-muted border-cyber-border',
}

export default function AlertBadge({ type }: { type: string }) {
  const cls = TYPE_COLOR[type] ?? 'bg-white/10 text-cyber-dim border-cyber-border'
  return (
    <span className={`inline-block text-xs font-mono px-2 py-0.5 rounded border ${cls}`}>
      {type.replace('_', ' ').toUpperCase()}
    </span>
  )
}

export function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_COLOR[status] ?? 'bg-white/10 text-cyber-dim border-cyber-border'
  return (
    <span className={`inline-block text-xs font-mono px-2 py-0.5 rounded border ${cls}`}>
      {status.toUpperCase()}
    </span>
  )
}
