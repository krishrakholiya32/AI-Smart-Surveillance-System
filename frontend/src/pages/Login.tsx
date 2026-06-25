import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import { useStore } from '../store'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const setToken  = useStore(s => s.setToken)
  const navigate  = useNavigate()

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await api.post<{ access_token: string }>('/auth/login', { username, password })
      setToken(res.data.access_token)
      navigate('/')
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? 'Invalid credentials')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-cyber-bg flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 border border-cyber-cyan/40 rounded-lg mb-4">
            <span className="font-mono text-cyber-cyan text-2xl">AI</span>
          </div>
          <h1 className="font-mono text-cyber-cyan tracking-widest text-lg">SURVEILLANCE SYSTEM</h1>
          <p className="text-cyber-muted text-sm mt-1">Secure Access</p>
        </div>

        <form onSubmit={submit} className="bg-cyber-surface border border-cyber-border rounded-lg p-6 space-y-4">
          <div>
            <label className="block text-xs font-mono text-cyber-muted mb-1.5 tracking-widest">USERNAME</label>
            <input
              value={username}
              onChange={e => setUsername(e.target.value)}
              className="w-full bg-cyber-bg border border-cyber-border rounded px-3 py-2 text-sm font-mono text-cyber-dim focus:outline-none focus:border-cyber-cyan transition-colors"
              autoFocus
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-cyber-muted mb-1.5 tracking-widest">PASSWORD</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-cyber-bg border border-cyber-border rounded px-3 py-2 text-sm font-mono text-cyber-dim focus:outline-none focus:border-cyber-cyan transition-colors"
            />
          </div>
          {error && <p className="text-cyber-red text-xs font-mono">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full border border-cyber-cyan text-cyber-cyan font-mono text-sm py-2 rounded hover:bg-cyber-cyan/10 transition-colors disabled:opacity-50"
          >
            {loading ? 'AUTHENTICATING…' : 'LOGIN'}
          </button>
        </form>
      </div>
    </div>
  )
}
