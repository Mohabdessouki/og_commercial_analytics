import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LabelList } from 'recharts'

export default function PerformanceBar({ perf }: { perf?: any }) {
  if (!perf) return null
  const data = Object.keys(perf).map(k => ({ model: k, MAPE: perf[k].MAPE }))
  return (
    <div className="card">
      <div className="section-title">Model Performance (MAPE)</div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="MAPE">
            <LabelList dataKey="MAPE" position="top" formatter={(v:any)=>v.toFixed(2)} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}