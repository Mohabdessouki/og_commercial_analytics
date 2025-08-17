const base = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export async function runAnalysis(body: any) {
  const res = await fetch(`${base}/api/run-analysis`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function optimize(body: any) {
  const res = await fetch(`${base}/api/optimize`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function fetchNews(hours = 168, max_items = 30) {
  const res = await fetch(`${base}/api/news`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ hours, max_items })
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}