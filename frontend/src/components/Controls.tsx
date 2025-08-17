import React from 'react'

type Props = {
  onRun: (payload: any) => void
  onOptimize: (payload: any) => void
  busy: boolean
}

export default function Controls({ onRun, onOptimize, busy }: Props) {
  const [target, setTarget] = React.useState<'gasoline'|'diesel'>('gasoline')
  const [profile, setProfile] = React.useState<'corporate'|'typical_refinery'|'regional_refinery'>('typical_refinery')
  const [useReal, setUseReal] = React.useState(true)
  const [start, setStart] = React.useState('2022-01-01')
  const [trainRatio, setTrainRatio] = React.useState(0.8)
  const [horizon, setHorizon] = React.useState(30)

  return (
    <div className="card">
      <div className="section-title">Run Analysis</div>
      <div className="grid grid-3">
        <div>
          <div className="label">Product</div>
          <select value={target} onChange={e=>setTarget(e.target.value as any)}>
            <option value="gasoline">Gasoline</option>
            <option value="diesel">Diesel</option>
          </select>
        </div>
        <div>
          <div className="label">Profile</div>
          <select value={profile} onChange={e=>setProfile(e.target.value as any)}>
            <option value="corporate">Corporate</option>
            <option value="typical_refinery">Typical Refinery</option>
            <option value="regional_refinery">Regional Refinery</option>
          </select>
        </div>
        <div>
          <div className="label">Start Date</div>
          <input className="input" type="date" value={start} onChange={e=>setStart(e.target.value)} />
        </div>
        <div>
          <div className="label">Train Ratio</div>
          <input className="input" type="number" step="0.05" min={0.5} max={0.95} value={trainRatio}
                 onChange={e=>setTrainRatio(parseFloat(e.target.value))} />
        </div>
        <div>
          <div className="label">Horizon (days)</div>
          <input className="input" type="number" min={7} max={120} value={horizon}
                 onChange={e=>setHorizon(parseInt(e.target.value))} />
        </div>
        <div>
          <div className="label">Use Real Data</div>
          <input type="checkbox" checked={useReal} onChange={e=>setUseReal(e.target.checked)} />
        </div>
      </div>
      <div style={{ display:'flex', gap:8, marginTop:12 }}>
        <button className="btn" disabled={busy} onClick={() => onRun({
          target_product: target, profile, use_real_data: useReal, start_date: start,
          train_ratio: trainRatio, forecast_horizon: horizon
        })}>Run</button>
        <button className="btn secondary" disabled={busy} onClick={() => onOptimize({
          target_product: target, cost_scenario: 'base', competitive_scenario: 'base', horizon
        })}>Optimize Only</button>
      </div>
    </div>
  )
}