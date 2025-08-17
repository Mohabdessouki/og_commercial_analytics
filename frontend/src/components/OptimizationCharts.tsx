import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'

export default function OptimizationCharts({ rows }: { rows?: any[] }) {
  if (!rows || !rows.length) return null
  return (
    <div className="grid grid-2">
      <div className="card">
        <div className="section-title">Optimized vs Forecast Price</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={rows}>
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="forecast_ref_price" dot={false} />
            <Line type="monotone" dataKey="optimal_price" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="card">
        <div className="section-title">Projected Daily Profit</div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows}>
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="total_profit" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}