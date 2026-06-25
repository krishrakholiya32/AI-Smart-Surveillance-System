import { create } from 'zustand'
import type { Camera, AlertRecord } from '../api/client'

interface StreamMetrics {
  people: number; gun: number; knife: number
  running: number; loiter: number; fps: number
}

interface AppState {
  token: string | null
  cameras: Camera[]
  activeCameraId: number | null
  metrics: Record<number, StreamMetrics>
  recentAlerts: AlertRecord[]
  totalAlerts: number

  setToken: (t: string | null) => void
  setCameras: (c: Camera[]) => void
  setActiveCamera: (id: number | null) => void
  updateMetrics: (cameraId: number, m: StreamMetrics) => void
  clearMetrics: (cameraId: number) => void
  addAlert: (a: AlertRecord) => void
  bumpAlertCount: () => void
}

export const useStore = create<AppState>((set) => ({
  token: localStorage.getItem('token'),
  cameras: [],
  activeCameraId: null,
  metrics: {},
  recentAlerts: [],
  totalAlerts: 0,

  setToken: (t) => {
    if (t) localStorage.setItem('token', t)
    else localStorage.removeItem('token')
    set({ token: t })
  },

  setCameras: (c) => set({ cameras: c }),
  setActiveCamera: (id) => set({ activeCameraId: id }),

  updateMetrics: (cameraId, m) =>
    set((s) => ({ metrics: { ...s.metrics, [cameraId]: m } })),

  clearMetrics: (cameraId) =>
    set((s) => {
      const { [cameraId]: _removed, ...rest } = s.metrics
      return { metrics: rest }
    }),

  addAlert: (a) =>
    set((s) => ({
      recentAlerts: [a, ...s.recentAlerts].slice(0, 100),
      totalAlerts: s.totalAlerts + 1,
    })),

  bumpAlertCount: () => set((s) => ({ totalAlerts: s.totalAlerts + 1 })),
}))
