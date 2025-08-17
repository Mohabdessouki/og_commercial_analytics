import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function SensitivityBar({ rows }: { rows?: any[] }) {
  if (!rows || !rows.length) return null
  const data = rows.map(r => ({ factor: r.factor, elasticity: r.elasticity_mean }))
  return (
    <div className="card">
      <div className="section-title">Local Sensitivity (Elasticity)</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} layout="vertical">
          <XAxis type="number" />
          <YAxis type="category" dataKey="factor" width={200} />
          <Tooltip />
          <Bar dataKey="elasticity" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}