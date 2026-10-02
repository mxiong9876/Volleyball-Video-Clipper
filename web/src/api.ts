const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export type Health = {
  status: 'ok' | 'degraded'
  db: boolean
  redis: boolean
}

// /health returns 503 with a JSON body when a dependency is down,
// so read the body for both 200 and 503 instead of throwing.
export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${API_URL}/health`)
  if (res.status !== 200 && res.status !== 503) {
    throw new Error(`Unexpected status ${res.status}`)
  }
  return res.json()
}
