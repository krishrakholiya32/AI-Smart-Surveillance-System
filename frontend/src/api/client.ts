import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export interface Camera {
  id: number; name: string; source: string; is_active: boolean; width: number; height: number
}
export interface Zone {
  id: number; camera_id: number; name: string; points: number[][]; color: string; is_active: boolean
}
export interface AlertRecord {
  id: number; camera_id: number; alert_type: string; message: string
  person_id: number | null; snapshot_path: string | null; meta: Record<string,unknown> | null
  status: string; created_at: string
}
export interface EventRecord {
  id: number; camera_id: number | null; level: string; message: string; created_at: string
}

export const camerasApi = {
  list:   ()                  => api.get<Camera[]>('/cameras').then(r => r.data),
  get:    (id: number)        => api.get<Camera>(`/cameras/${id}`).then(r => r.data),
  create: (d: Partial<Camera>)=> api.post<Camera>('/cameras', d).then(r => r.data),
  update: (id: number, d: Partial<Camera>) => api.patch<Camera>(`/cameras/${id}`, d).then(r => r.data),
  delete: (id: number)        => api.delete(`/cameras/${id}`),
  start:  (id: number)        => api.post(`/cameras/${id}/start`).then(r => r.data),
  stop:   (id: number)        => api.post(`/cameras/${id}/stop`).then(r => r.data),
  status: (id: number)        => api.get<{running:boolean}>(`/cameras/${id}/status`).then(r => r.data),
}

export const zonesApi = {
  list:   (camera_id?: number) => api.get<Zone[]>('/zones', { params: { camera_id } }).then(r => r.data),
  create: (d: Partial<Zone>)   => api.post<Zone>('/zones', d).then(r => r.data),
  update: (id: number, d: Partial<Zone>) => api.patch<Zone>(`/zones/${id}`, d).then(r => r.data),
  delete: (id: number)         => api.delete(`/zones/${id}`),
}

export const alertsApi = {
  list: (params?: {camera_id?:number; alert_type?:string; limit?:number; offset?:number}) =>
    api.get<AlertRecord[]>('/alerts', { params }).then(r => r.data),
  delete: (id: number) => api.delete(`/alerts/${id}`),
  setStatus: (id: number, status: 'confirmed' | 'dismissed') =>
    api.patch<AlertRecord>(`/alerts/${id}`, { status }).then(r => r.data),
  snapshotUrl: (id: number) => `/api/alerts/${id}/snapshot`,
}

export const eventsApi = {
  list: (params?: {camera_id?:number; level?:string; limit?:number}) =>
    api.get<EventRecord[]>('/events', { params }).then(r => r.data),
}

export interface SettingsInfo {
  thresholds: {
    CONF_PERSON: number
    CONF_WEAPON: number
    RUN_THRESH_NORM: number
    LOITER_SECS: number
    CROWD_LIMIT: number
  }
  onnx_active: Record<string, boolean>
}

export const settingsApi = {
  get: () => api.get<SettingsInfo>('/settings').then(r => r.data),
}
