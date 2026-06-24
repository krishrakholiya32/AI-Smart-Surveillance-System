interface Props {
  label: string
  value: number | string
  danger?: boolean
  warn?: boolean
}

export default function MetricsCard({ label, value, danger, warn }: Props) {
  const color = danger ? 'text-cyber-red border-cyber-red/30' :
                warn   ? 'text-cyber-orange border-cyber-orange/30' :
                         'text-cyber-cyan border-cyber-border'
  return (
    <div className={`bg-cyber-surface rounded border p-3 ${color}`}>
      <p className="text-cyber-muted text-xs font-mono tracking-widest mb-1">{label}</p>
      <p className={`font-mono text-2xl font-bold ${color.split(' ')[0]}`}>{value}</p>
    </div>
  )
}
