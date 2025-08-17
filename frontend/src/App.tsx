import React from 'react'
import Controls from './components/Controls'
import Kpis from './components/Kpis'
import PriceForecastChart from './components/PriceForecastChart'
import PerformanceBar from './components/PerformanceBar'
import SensitivityBar from './components/SensitivityBar'
import OptimizationCharts from './components/OptimizationCharts'
import NewsCard from './components/NewsCard'
import { runAnalysis, optimize, fetchNews } from './lib/api'

export default function App() {
  const [busy, setBusy] = React.useState(false)
  const [kpis, setKpis] = React.useState<any>()
  const [data, setData] = React.useState<any[]>([])
  const [forecasts, setForecasts] = React.useState<Record<string, number[]>>()
  const [perf, setPerf] = React.useState<any>()
  const [sens, setSens] = React.useState<any[]>()
  const [opt, setOpt] = React.useState<any[]>()
  const [news, setNews] = React.useState<string>('')
  const [newsItems, setNewsItems] = React.useState<any[]>([])

  async function doRun(payload: any) {
    try {
      setBusy(true)
      const res = await runAnalysis(payload)
      setKpis(res.kpis)
      setData(res.data)
      setForecasts(res.forecasts)
      setPerf(res.model_performance)
      setSens(res.sensitivity)
      setOpt(res.optimization)
      const ns = await fetchNews()
      setNews(ns.summary_md)
      setNewsItems(ns.items || [])
    } catch (e) {
      alert('Run failed: ' + (e as Error).message)
    } finally { setBusy(false) }
  }

  async function doOptimize(payload: any) {
    try {
      setBusy(true)
      const res = await optimize(payload)
      setOpt(res.optimization)
    } catch (e) {
      alert('Optimize failed: ' + (e as Error).message)
    } finally { setBusy(false) }
  }

  return (
    <div className="container">
      <h1>Oil & Gas Pricing Dashboard</h1>
      <p className="label">Forecasting • Sensitivity • Optimization • News</p>

      <Controls busy={busy} onRun={doRun} onOptimize={doOptimize} />
      <Kpis kpis={kpis} />

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <PriceForecastChart data={data} forecasts={forecasts} />
        <PerformanceBar perf={perf} />
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <SensitivityBar rows={sens} />
        <OptimizationCharts rows={opt} />
      </div>

      <div style={{ marginTop: 16 }}>
        <NewsCard md={news} items={newsItems} />
      </div>
    </div>
  )
}