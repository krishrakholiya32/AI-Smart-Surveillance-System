import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { Monitor, Bell, Settings, LogOut, Menu, X, Camera, List } from 'lucide-react'
import { useStore } from '../store'

const NAV = [
  { to: '/',        icon: Monitor, label: 'Dashboard' },
  { to: '/alerts',  icon: Bell,    label: 'Alerts'    },
  { to: '/cameras', icon: Camera,  label: 'Cameras'   },
  { to: '/events',  icon: List,    label: 'Events'    },
  { to: '/settings',icon: Settings,label: 'Settings'  },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(true)
  const navigate   = useNavigate()
  const setToken   = useStore(s => s.setToken)
  const totalAlerts = useStore(s => s.totalAlerts)

  const logout = () => { setToken(null); navigate('/login') }

  return (
    <div className="flex h-screen bg-cyber-bg overflow-hidden">
      {/* Sidebar */}
      <aside className={`flex flex-col bg-cyber-surface border-r border-cyber-border transition-all duration-200 ${open ? 'w-56' : 'w-14'}`}>
        {/* Logo */}
        <div className="flex items-center gap-3 px-3 py-4 border-b border-cyber-border">
          <div className="w-8 h-8 rounded border border-cyber-cyan flex items-center justify-center flex-shrink-0">
            <span className="text-cyber-cyan text-xs font-mono">AI</span>
          </div>
          {open && <span className="text-cyber-cyan font-mono text-sm tracking-widest truncate">SURVEILLANCE</span>}
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 space-y-1">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 mx-2 rounded transition-colors ${
                  isActive
                    ? 'bg-cyber-cyan/10 text-cyber-cyan border border-cyber-cyan/30'
                    : 'text-cyber-dim hover:bg-white/5 hover:text-cyber-cyan border border-transparent'
                }`
              }
            >
              <Icon size={16} className="flex-shrink-0" />
              {open && <span className="font-sans font-600 text-sm tracking-wide">{label}</span>}
              {!open && label === 'Alerts' && totalAlerts > 0 && (
                <span className="absolute left-8 top-1 w-2 h-2 bg-cyber-red rounded-full" />
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom */}
        <div className="border-t border-cyber-border p-2 space-y-1">
          <button onClick={logout}
            className="w-full flex items-center gap-3 px-3 py-2 rounded text-cyber-dim hover:text-cyber-red hover:bg-cyber-red/10 transition-colors">
            <LogOut size={16} className="flex-shrink-0" />
            {open && <span className="text-sm">Logout</span>}
          </button>
          <button onClick={() => setOpen(o => !o)}
            className="w-full flex items-center gap-3 px-3 py-2 rounded text-cyber-dim hover:text-cyber-cyan hover:bg-white/5 transition-colors">
            {open ? <X size={16} /> : <Menu size={16} />}
            {open && <span className="text-sm">Collapse</span>}
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        {/* Top bar */}
        <header className="sticky top-0 z-10 bg-cyber-surface/80 backdrop-blur border-b border-cyber-border px-6 py-3 flex items-center justify-between">
          <h1 className="font-mono text-cyber-cyan tracking-widest text-sm">
            AI SMART SURVEILLANCE SYSTEM
          </h1>
          <div className="flex items-center gap-4">
            {totalAlerts > 0 && (
              <span className="text-xs font-mono text-cyber-red border border-cyber-red/40 rounded px-2 py-0.5">
                {totalAlerts} ALERTS
              </span>
            )}
            <span className="text-xs font-mono text-cyber-muted">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </header>

        <div className="p-6">
          {children}
        </div>
      </main>
    </div>
  )
}
