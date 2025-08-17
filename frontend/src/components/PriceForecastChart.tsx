import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'

type Props = { data: any[]; forecasts?: Record<string, number[]> }
export default function PriceForecastChart({ data, forecasts }: Props) {
  const lastDate = data.length ? new Date(data[data.length-1].date) : null
  const future: any[] = []
  if (forecasts && lastDate) {
    const names = Object.keys(forecasts)
    const maxLen = Math.max(...names.map(n => forecasts[n].length))
    for (let i=0;i<maxLen;i++) {
      const d = new Date(lastDate); d.setDate(d.getDate()+i+1)
      const row:any = { date: d.toISOString().slice(0,10) }
      names.forEach(n => { row[n] = forecasts[n][i] })
      future.push(row)
    }
  }
  return (
    <div className="card">
      <div className="section-title">Price History & Forecast</div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={[...data.map(d=>({ date:d.date, Historical:d.gasoline_price || d.diesel_price })), ...future]}>
          <XAxis dataKey="date" hide/>
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Historical" dot={false} strokeWidth={2} />
          {forecasts && Object.keys(forecasts).map((k, i) => (
            <Line key={k} type="monotone" dataKey={k} dot={false} strokeWidth={2} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}