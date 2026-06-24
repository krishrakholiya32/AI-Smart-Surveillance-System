import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useStore } from './store'
import Layout from './components/Layout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import CameraView from './pages/CameraView'
import Cameras from './pages/Cameras'
import Alerts from './pages/Alerts'
import Events from './pages/Events'
import Settings from './pages/Settings'

function Protected({ children }: { children: React.ReactNode }) {
  const token = useStore(s => s.token)
  return token ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/*" element={
          <Protected>
            <Layout>
              <Routes>
                <Route path="/"              element={<Dashboard />} />
                <Route path="/camera/:id"    element={<CameraView />} />
                <Route path="/cameras"       element={<Cameras />} />
                <Route path="/alerts"        element={<Alerts />} />
                <Route path="/events"        element={<Events />} />
                <Route path="/settings"      element={<Settings />} />
              </Routes>
            </Layout>
          </Protected>
        } />
      </Routes>
    </BrowserRouter>
  )
}
