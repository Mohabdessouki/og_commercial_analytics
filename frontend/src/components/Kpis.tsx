export default function Kpis({ kpis }: { kpis?: any }) {
  if (!kpis) return null
  const { current_price, current_demand, current_cost } = kpis
  const marginPct = current_price ? ((current_price - current_cost)/current_price*100) : 0
  return (
    <div className="grid grid-3">
      <div className="card"><div className="label">Current Price</div><div style={{ fontSize:24, fontWeight:700 }}>${current_price.toFixed(2)}</div></div>
      <div className="card"><div className="label">Current Demand (bpd)</div><div style={{ fontSize:24, fontWeight:700 }}>{Math.round(current_demand).toLocaleString()}</div></div>
      <div className="card"><div className="label">Current Cost</div><div style={{ fontSize:24, fontWeight:700 }}>${current_cost.toFixed(2)} <span className="badge">Margin {marginPct.toFixed(1)}%</span></div></div>
    </div>
  )
}