import { useQuery } from '@tanstack/react-query'
import { fetchHealth } from './api'

function StatusRow({ label, ok }: { label: string; ok: boolean | undefined }) {
  const color = ok === undefined ? 'bg-gray-300' : ok ? 'bg-green-500' : 'bg-red-500'
  return (
    <li className="flex items-center justify-between py-2">
      <span>{label}</span>
      <span className={`h-3 w-3 rounded-full ${color}`} aria-label={ok ? 'up' : 'down'} />
    </li>
  )
}

export default function App() {
  const { data, error, isPending } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 5000,
  })

  let summary = 'Checking…'
  if (error) summary = 'API unreachable'
  else if (data) summary = data.status === 'ok' ? 'All systems go' : 'Degraded'

  return (
    <main className="mx-auto max-w-md p-8 font-sans">
      <h1 className="text-2xl font-bold">Volley Breakdown</h1>
      <section className="mt-6 rounded-lg border border-gray-200 p-4">
        <h2 className="font-semibold">{summary}</h2>
        <ul className="mt-2 divide-y divide-gray-100">
          <StatusRow label="API" ok={isPending ? undefined : !error} />
          <StatusRow label="Postgres" ok={data?.db} />
          <StatusRow label="Redis" ok={data?.redis} />
        </ul>
      </section>
    </main>
  )
}
