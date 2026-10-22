import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Dispatch, DragEvent, FormEvent, ReactNode, SetStateAction } from 'react'
import {
  Activity, Anchor, ArrowDownRight, ArrowUpRight, BarChart3, CheckCircle2,
  CircleAlert, CloudUpload, Compass, Database, FileSpreadsheet,
  Gauge, LayoutDashboard, LoaderCircle, Menu, ShieldCheck, Ship, Waves, X,
} from 'lucide-react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line,
  LineChart, Pie, PieChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import {
  buildPeriodRange, createRequestGate, dashboardMatchesDataset,
  datasetsMatch, describePeriodCoverage, latestValidObservation, sampleHistory,
} from './dashboardData.mjs'
import type { DashboardSummary, HistoryRow } from './dashboardData.mjs'
import './App.css'

const API = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const PORT_ID = 'port-of-los-angeles'
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024
type Page = 'overview' | 'port' | 'predict' | 'delay' | 'performance' | 'methodology' | 'dataset'
type DatePeriod = '1m' | '6m' | '1y' | '2y' | '3y' | 'custom'
type Summary = DashboardSummary
type Metadata = {
  project_name: string; port_id: string; port_name: string; target_definition: string
  prediction_horizon: string; data_status: string; synthetic: boolean
  observation_start: string | null; observation_end: string | null
  processed_rows: number; last_processed_at: string | null; limitations: string[]
  source_url: string | null; source_name: string | null; acquisition_date: string | null
  geographic_scope: {
    port_id: string; label: string; identified: boolean
    identification_method: string
    coordinate_coverage: {
      latitude: { min: number; max: number }; longitude: { min: number; max: number }
    } | null
  }
  target_status: string; target_source: string | null; target_rule: string | null
  model_version: string | null
  dataset_sha256: string | null
  evaluation_dataset: {
    data_status: string | null; source_name: string | null
    filename: string | null; sha256: string | null; identity_recorded?: boolean
  }
  model_training_dataset: Metadata['evaluation_dataset'] | null
  evaluation_dataset_matches_active: boolean | null
  prediction_compatible: boolean | null; prediction_incompatibility: string | null
}
type MetricSet = {
  sample_count: number; class_distribution: { negative: number; positive: number }
  classification_threshold: number
  accuracy: number | null; precision: number | null; recall: number | null
  f1: number | null; roc_auc: number | null; roc_auc_reason?: string | null
  brier_score: number | null; log_loss: number | null; confusion_matrix: number[][]
}
type Metrics = {
  available: boolean; selected_model?: string; synthetic_warning?: string | null
  data_status?: string; training_dataset?: Metadata['evaluation_dataset']
  target_status?: string; target_definition?: string; target_source?: string | null
  calibration?: { status: string; calibration_fit_brier_score: number | null; sample_count: number }
  splits?: {
    train?: { sample_count: number; period: { start: string; end: string } }
    calibration?: { sample_count: number; period: { start: string; end: string } }
    validation?: { sample_count: number; period: { start: string; end: string } }
    test: { sample_count: number; period: { start: string; end: string } }
  }
  comparison_test_raw_models?: Record<string, MetricSet>
  comparison_test_calibrated_models?: Record<string, MetricSet>
  threshold_selection?: {
    method: string; partition: string; sample_count: number
    selected_model: string; selected_threshold: number
    model_thresholds: Record<string, {
      threshold: number; precision: number; recall: number; f1: number
      predicted_positive: number
    }>
  }
  test?: {
    selected_model: MetricSet
    selected_model_is_calibrated: boolean
    risk_distribution?: { low: number; moderate: number; high: number }
  }
  calibration_curve?: {
    points: { mean_predicted_probability: number; observed_event_fraction: number }[] | null
    unavailable_reason: string | null
  }
  global_feature_importance?: Record<string, number>
}
type DelayMetrics = {
  available: boolean
  model_availability?: 'available' | 'unavailable'
  reason?: string
  model?: string
  target?: string
  target_definition?: string
  probability_calibration?: string
  probability_calibration_note?: string
  dataset_identity?: { data_status?: string | null; source_name?: string | null; filename?: string | null; sha256?: string | null }
  training_dataset?: Metadata['evaluation_dataset'] | null
  target_distribution?: { not_delayed: number; delayed: number }
  features?: string[]
  split_strategy?: string
  train?: { sample_count: number; period: { start: string; end: string } }
  test?: {
    sample_count: number; period: { start: string; end: string }
    precision: number | null; recall: number | null; f1: number | null
    roc_auc: number | null; brier_score: number | null; log_loss: number | null
    confusion_matrix: number[][]
  }
}
type DelayPrediction = {
  status: 'available' | 'unavailable'
  model_availability: 'available' | 'unavailable'
  reason: string | null
  evaluation_warning: string | null
  predicted_outcome: 'delayed' | 'not_delayed' | null
  delay_probability_percent: number | null
  risk_category: 'Low' | 'Medium' | 'High' | null
  estimated_delay_minutes: number | null
  duration_status: string
  probability_calibration: string | null
  probability_note: string | null
  risk_thresholds: Record<string, number> | null
  main_factors: {
    feature: string; observed_value: number | string; reference_value: number | string
    probability_change: number; direction: 'increases' | 'decreases' | 'no_change'; method: string
  }[]
  recommendations: string[]
  target_definition: string
  dataset_provenance: {
    data_status?: string | null; source_name?: string | null
    filename?: string | null; sha256?: string | null; identity_recorded?: boolean
  }
  model_version: string | null
}
type Prediction = {
  target_definition: string; prediction_horizon: string; congestion_probability: number
  target_status?: string | null; target_source?: string | null
  risk_category: 'low' | 'moderate' | 'high'; classification_threshold: number
  probability_calibration: string
  prediction_reliability: 'high' | 'moderate' | 'low' | 'insufficient_evidence'
  reliability_reasons: string[]
  local_explanations: {
    feature: string; observed_value: number; reference_value: number
    probability_change: number; direction: 'increases' | 'decreases' | 'no_change'
  }[]
  recommended_actions: string[]; data_quality_warnings: string[]
}
type Port = { port_id: string; name: string; data_available: boolean; synthetic: boolean }
type DatasetValidation = {
  valid: boolean; errors: string[]; warnings: string[]; input_rows: number
  columns: string[]; missing_columns: string[]; missing_values: Record<string, number>
  invalid_dates: number; invalid_numeric_values: Record<string, number>
  invalid_mmsi_values: number; duplicate_records: number
  preview: Record<string, string | null>[]
}
type ActiveDataset = {
  available: boolean; processing_status?: string; source_name?: string | null; source_url?: string | null
  filename?: string | null; data_status?: string; synthetic?: boolean
  input_rows?: number | null; processed_rows?: number; date_coverage?: { start: string; end: string } | null
  last_processed_at?: string | null; model_note?: string | null; sha256?: string | null
  geographic_scope?: Metadata['geographic_scope']
  target_status?: string; target_source?: string | null
}
type UploadResult = {
  status: string; filename: string; source_name: string; input_rows: number
  processed_rows: number; date_coverage: { start: string; end: string }
  last_processed_at: string; sha256: string; model_retrained: boolean; model_note: string
}
type FormValues = Record<string, string>

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API}${path}`
  const method = init?.method ?? 'GET'
  let response: Response
  try {
    response = await fetch(url, { ...init, cache: 'no-store' })
  } catch (caught) {
    const reason = caught instanceof Error ? ` (${caught.message})` : ''
    throw new Error(
      `${method} ${url} failed before receiving a response${reason}. Check API reachability and CORS for ${window.location.origin}.`,
      { cause: caught },
    )
  }
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${method} ${url} returned ${response.status} ${response.statusText}${detail ? `: ${detail}` : ''}`)
  }
  try {
    return await response.json() as T
  } catch (caught) {
    const reason = caught instanceof Error ? `: ${caught.message}` : ''
    throw new Error(`${method} ${url} returned an invalid JSON response${reason}`, { cause: caught })
  }
}
const percent = (value: number | null | undefined) => value == null ? 'N/A' : `${(value * 100).toFixed(1)}%`
const metric = (value: number | null | undefined) => value == null ? 'N/A' : value.toFixed(3)
const titleCase = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase())

function App() {
  const [page, setPage] = useState<Page>('overview')
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [delayMetrics, setDelayMetrics] = useState<DelayMetrics | null>(null)
  const [ports, setPorts] = useState<Port[]>([])
  const [summary, setSummary] = useState<Summary | null>(null)
  const [history, setHistory] = useState<HistoryRow[]>([])
  const [activeDataset, setActiveDataset] = useState<ActiveDataset | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [rangeLoading, setRangeLoading] = useState(false)
  const requestGate = useRef(createRequestGate())
  const [period, setPeriod] = useState<DatePeriod>('3y')
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd] = useState('')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [formError, setFormError] = useState('')
  const [delayPrediction, setDelayPrediction] = useState<DelayPrediction | null>(null)
  const [delayPredicting, setDelayPredicting] = useState(false)
  const [delayFormError, setDelayFormError] = useState('')
  const [delayForm, setDelayForm] = useState<FormValues>({
    vessel_arrivals_daily: '', queue_length_vessels: '', port_capacity_berths: '',
    weather_condition: '', wind_speed_kmh: '',
    day_of_week: String(new Date().getUTCDay() === 0 ? 6 : new Date().getUTCDay() - 1),
    month: String(new Date().getUTCMonth() + 1),
  })
  const [form, setForm] = useState<FormValues>({
    vessel_density: '', anchorage_vessels: '', anchorage_dwell_hours: '',
    day_of_week: '', month: '',
  })
  const coverageEnd = metadata?.observation_end ?? null
  const coverageStart = metadata?.observation_start ?? ''
  const selectedRange = useMemo(
    () => buildPeriodRange(period, coverageStart, coverageEnd, customStart, customEnd),
    [period, coverageEnd, customStart, customEnd, coverageStart],
  )
  const visibleHistory = history
  useEffect(() => {
    if (!mobileNavOpen) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMobileNavOpen(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [mobileNavOpen])
  const refreshDashboard = useCallback(async (expectedSha256?: string) => {
    const version = requestGate.current.next()
    setError('')
    setRangeLoading(true)
    try {
      const snapshot = await requestJson<{
        summary: Summary; metadata: Metadata; metrics: Metrics
        ports: Port[]; history: HistoryRow[]; active_dataset: ActiveDataset
      }>(
        `/api/v1/dashboard${selectedRange.start && selectedRange.end
          ? `?${new URLSearchParams({ start_date: selectedRange.start, end_date: selectedRange.end })}`
          : ''}`,
      )
      if (!requestGate.current.isCurrent(version)) return
      if (expectedSha256 && !dashboardMatchesDataset(snapshot, expectedSha256)) {
        throw new Error(
          `The upload completed, but the refreshed dashboard does not match dataset ${expectedSha256}. The previous dashboard values were retained.`,
        )
      }
      setMetadata(snapshot.metadata)
      setMetrics(snapshot.metrics)
      setPorts(snapshot.ports)
      setSummary(snapshot.summary)
      setHistory(snapshot.history)
      setActiveDataset(snapshot.active_dataset)
      try {
        setDelayMetrics(await requestJson<DelayMetrics>('/api/v1/delay/metrics'))
      } catch (caught) {
        const reason = caught instanceof Error ? caught.message : 'Shipment-delay metrics request failed.'
        setDelayMetrics({ available: false, model_availability: 'unavailable', reason })
      }
    } catch (caught) {
      const errorMessage = caught instanceof Error ? caught.message : 'Unable to refresh the active dataset.'
      if (requestGate.current.isCurrent(version)) {
        if (!expectedSha256) {
          setSummary(null)
          setHistory([])
        }
        setError(errorMessage)
      }
      throw new Error(errorMessage, { cause: caught })
    } finally {
      if (requestGate.current.isCurrent(version)) {
        setLoading(false)
        setRangeLoading(false)
      }
    }
  }, [selectedRange])
  useEffect(() => {
    const gate = requestGate.current
    const timer = window.setTimeout(() => {
      void refreshDashboard().catch(() => undefined)
    }, 0)
    return () => {
      window.clearTimeout(timer)
      gate.invalidate()
    }
  }, [refreshDashboard])
  const evaluationMatches = metadata
    ? datasetsMatch(activeDataset, metadata.evaluation_dataset)
    : null
  const datasetMismatchNotice = evaluationMatches === false
    ? `Historical analytics use ${metadata?.source_name ?? 'the active dataset'}, but model evaluation remains tied to ${metadata?.evaluation_dataset.data_status ?? 'a different dataset'}. Uploading data does not retrain the model.`
    : ''
  const navItems: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'port', label: 'Port analysis', icon: Anchor },
    { id: 'predict', label: 'Predict congestion', icon: Gauge },
    { id: 'delay', label: 'Shipment delay', icon: Ship },
    { id: 'performance', label: 'Model performance', icon: BarChart3 },
    { id: 'methodology', label: 'Methodology', icon: Compass },
    { id: 'dataset', label: 'Dataset manager', icon: FileSpreadsheet },
  ]
  async function submitPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setFormError(''); setPrediction(null)
    const values = Object.fromEntries(Object.entries(form).map(([key, value]) => [key, Number(value)]))
    if (Object.values(values).some((value) => !Number.isFinite(value))) {
      setFormError('Enter valid numeric values for every field.'); return
    }
    if (values.anchorage_vessels > values.vessel_density) {
      setFormError('Anchorage vessel count cannot exceed total observed vessel density.'); return
    }
    setPredicting(true)
    try {
      const result = await requestJson<Prediction>('/api/v1/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values),
      })
      setPrediction(result)
    } catch (caught) {
      setFormError(caught instanceof Error ? caught.message : 'Prediction failed.')
    } finally { setPredicting(false) }
  }
  async function submitDelayPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setDelayFormError(''); setDelayPrediction(null)
    const values: Record<string, number | string> = { weather_condition: delayForm.weather_condition.trim() }
    for (const key of ['vessel_arrivals_daily', 'queue_length_vessels', 'port_capacity_berths', 'wind_speed_kmh', 'day_of_week', 'month']) {
      values[key] = Number(delayForm[key])
    }
    if (Object.entries(values).some(([key, value]) => key === 'weather_condition'
      ? typeof value !== 'string' || !value
      : typeof value !== 'number' || !Number.isFinite(value))) {
      setDelayFormError('Enter valid values for every shipment and port condition.'); return
    }
    setDelayPredicting(true)
    try {
      const result = await requestJson<DelayPrediction>('/api/v1/delay/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values),
      })
      setDelayPrediction(result)
    } catch (caught) {
      setDelayFormError(caught instanceof Error ? caught.message : 'Shipment-delay prediction failed.')
    } finally { setDelayPredicting(false) }
  }
  const isSynthetic = activeDataset?.synthetic ?? false
  const syntheticNotice = isSynthetic
    ? 'The active dataset is synthetic development data, not verified maritime observations.'
    : ''
  return (
    <div className="app-shell">
      {mobileNavOpen && <button className="drawer-backdrop" aria-label="Close navigation menu" onClick={() => setMobileNavOpen(false)} />}
      <aside className={`sidebar ${mobileNavOpen ? 'sidebar-open' : ''}`}>
        <button className="brand" onClick={() => { setPage('overview'); setMobileNavOpen(false) }}>
          <span className="brand-mark"><Waves size={22} /></span><span><strong>Port Congestion</strong><small>AND SHIPMENT DELAY PREDICTION SYSTEM</small></span>
        </button>
        {mobileNavOpen && <button className="drawer-close" onClick={() => setMobileNavOpen(false)}><X size={16} />Close navigation</button>}
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">{navItems.map(({ id, label, icon: Icon }) => (
          <button className={`nav-link ${page === id ? 'active' : ''}`} key={id} aria-current={page === id ? 'page' : undefined} onClick={() => { setPage(id); setMobileNavOpen(false) }}>
            <Icon size={18} strokeWidth={1.8} /><span>{label}</span>{page === id && <span className="nav-active-dot" />}
          </button>
        ))}</nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-port"><div className="sidebar-port-icon"><Ship size={17} /></div><div><span>SELECTED SCOPE</span><strong>{metadata?.port_name ?? 'Geographic scope unavailable'}</strong><small>{metadata?.geographic_scope.identification_method ?? 'Awaiting dataset metadata'}</small></div><span className="online-dot" /></div>
        <div className="sidebar-footer">LOCAL ANALYTICS <span>v1.0</span></div>
      </aside>
      <main className="main-content">
        <header className="topbar"><div className="topbar-start"><button className="menu-button" aria-label={mobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'} aria-expanded={mobileNavOpen} onClick={() => setMobileNavOpen((open) => !open)}>{mobileNavOpen ? <X size={19} /> : <Menu size={19} />}</button><div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{navItems.find((item) => item.id === page)?.label}</strong></div></div><div className="topbar-status"><span className="status-dot" />LOCAL ENVIRONMENT</div></header>
        <div className="page-container">
          {syntheticNotice && <div className="synthetic-banner" role="note"><CircleAlert size={17} /><span><strong>Synthetic data notice</strong> — {syntheticNotice}</span></div>}
          {datasetMismatchNotice && <div className="synthetic-banner" role="note"><CircleAlert size={17} /><span><strong>Evaluation dataset differs</strong> — {datasetMismatchNotice}</span></div>}
          {error && <div className="error-banner" role="alert"><CircleAlert size={18} /><span>{error}</span></div>}
          {(loading || rangeLoading) && <div className="loading-state"><LoaderCircle className="spin" size={24} />{loading ? 'Connecting to local analytics API…' : 'Refreshing the active dataset…'}</div>}
          {!loading && !metadata && !error && <div className="empty-chart">No dashboard data is available.</div>}
          {!loading && metadata && page === 'overview' && <Overview summary={summary} metadata={metadata} metrics={metrics} history={visibleHistory} loadHistory={() => setPage('port')} period={period} setPeriod={setPeriod} range={selectedRange} setCustomStart={setCustomStart} setCustomEnd={setCustomEnd} rangeLoading={rangeLoading} coverageStart={coverageStart} coverageEnd={coverageEnd} />}
          {!loading && metadata && page === 'port' && <PortAnalysis portList={ports} summary={summary} rows={visibleHistory} loading={rangeLoading} datasetAvailable={activeDataset?.available ?? false} period={period} setPeriod={setPeriod} range={selectedRange} setCustomStart={setCustomStart} setCustomEnd={setCustomEnd} coverageStart={coverageStart} coverageEnd={coverageEnd} />}
          {!loading && metadata && page === 'predict' && <PredictionPage form={form} setForm={setForm} onSubmit={submitPrediction} result={prediction} submitting={predicting} error={formError} metadata={metadata} />}
          {!loading && metadata && page === 'delay' && <ShipmentDelayPage form={delayForm} setForm={setDelayForm} onSubmit={submitDelayPrediction} result={delayPrediction} submitting={delayPredicting} error={delayFormError} metrics={delayMetrics} />}
          {!loading && metadata && page === 'performance' && <Performance metrics={metrics} delayMetrics={delayMetrics} activeDataset={activeDataset} metadata={metadata} />}
          {!loading && metadata && page === 'methodology' && <Methodology metadata={metadata} />}
          {!loading && metadata && page === 'dataset' && <DatasetManager active={activeDataset} metadata={metadata} onUploaded={refreshDashboard} />}
        </div>
        <footer className="page-footer"><span>Port Congestion · AIS congestion proxy</span></footer>
      </main>
    </div>
  )
}

function PageHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="page-heading"><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
}
function PanelTitle({ title, subtitle, action }: { title: string; subtitle: string; action?: ReactNode }) {
  return <div className="panel-title"><div><h2>{title}</h2><p>{subtitle}</p></div>{action}</div>
}
function EmptyChart({ text }: { text: string }) { return <div className="empty-chart"><BarChart3 size={25} /><span>{text}</span></div> }
function StatCard({ icon: Icon, label, value, detail }: { icon: typeof Activity; label: string; value: string; detail: string }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon size={17} /></span></div><strong className="stat-value">{value}</strong><span className="stat-detail">{detail}</span></article>
}

function DateRangeSelector({ period, setPeriod, range, setCustomStart, setCustomEnd, coverageStart, coverageEnd }: {
  period: DatePeriod; setPeriod: Dispatch<SetStateAction<DatePeriod>>
  range: { start: string; end: string }
  setCustomStart: Dispatch<SetStateAction<string>>; setCustomEnd: Dispatch<SetStateAction<string>>
  coverageStart: string; coverageEnd: string | null
}) {
  return <div className="date-range-selector">
    <label className="control-field"><span className="field-label">DATE RANGE</span>
      <select value={period} onChange={(event) => setPeriod(event.target.value as DatePeriod)}>
        <option value="1m">1 month</option><option value="6m">6 months</option>
        <option value="1y">1 year</option><option value="2y">2 years</option>
        <option value="3y">3 years</option><option value="custom">Custom range</option>
      </select>
    </label>
    {period === 'custom' && <>
      <label className="control-field"><span className="field-label">FROM</span><input type="date" value={range.start} min={coverageStart || undefined} max={coverageEnd || undefined} onChange={(event) => setCustomStart(event.target.value)} /></label>
      <label className="control-field"><span className="field-label">TO</span><input type="date" value={range.end} min={coverageStart || undefined} max={coverageEnd || undefined} onChange={(event) => setCustomEnd(event.target.value)} /></label>
    </>}
    <span className="period-caption">{describePeriodCoverage(period, coverageStart, coverageEnd, range)}</span>
    {range.start && range.end && range.start > range.end && <span className="date-range-error" role="alert">Start date must be on or before end date.</span>}
  </div>
}

function Overview({ summary, metadata, metrics, history, loadHistory, period, setPeriod, range, setCustomStart, setCustomEnd, rangeLoading, coverageStart, coverageEnd }: {
  summary: Summary | null; metadata: Metadata | null; metrics: Metrics | null
  history: HistoryRow[]; loadHistory: () => void
  period: DatePeriod; setPeriod: Dispatch<SetStateAction<DatePeriod>>
  range: { start: string; end: string }
  setCustomStart: Dispatch<SetStateAction<string>>; setCustomEnd: Dispatch<SetStateAction<string>>
  rangeLoading: boolean; coverageStart: string; coverageEnd: string | null
}) {
  const score = metrics?.test?.selected_model
  const distribution = metrics?.test?.risk_distribution ? [
    { name: 'Low', value: metrics.test.risk_distribution.low, color: '#5c9c82' },
    { name: 'Moderate', value: metrics.test.risk_distribution.moderate, color: '#d9a24e' },
    { name: 'High', value: metrics.test.risk_distribution.high, color: '#c97059' },
  ] : []
  const rows = sampleHistory(history, 30)
  return <>
    <PageHeading eyebrow="PORT INTELLIGENCE / OVERVIEW" title="Congestion overview" description="Historical activity and model context for the selected port scope." />
    <DateRangeSelector {...{ period, setPeriod, range, setCustomStart, setCustomEnd, coverageStart, coverageEnd }} />
    <div className="scope-card"><div className="scope-icon"><Anchor size={19} /></div><div className="scope-copy"><span>SELECTED GEOGRAPHIC SCOPE</span><strong>{summary?.port_name ?? metadata?.port_name ?? 'Geographic scope unavailable'}</strong><small>{metadata?.geographic_scope.identification_method}</small><small>{summary?.coverage ? `${summary.coverage.start} — ${summary.coverage.end} UTC` : 'No processed history available'}{metadata?.last_processed_at ? ` · Updated ${new Date(metadata.last_processed_at).toLocaleString()}` : ''}</small><small>Dataset: {metadata?.source_name ?? ((metadata?.synthetic ?? summary?.synthetic) ? 'Synthetic development data' : 'Not reported')}{metadata?.acquisition_date ? ` · Acquired ${metadata.acquisition_date}` : ''}{metadata?.source_url && <> · <a href={metadata.source_url} target="_blank" rel="noreferrer">Source details</a></>}</small>{metadata?.geographic_scope.coordinate_coverage && <small>Coordinate extent: {metadata.geographic_scope.coordinate_coverage.latitude.min.toFixed(3)}–{metadata.geographic_scope.coordinate_coverage.latitude.max.toFixed(3)}° lat, {metadata.geographic_scope.coordinate_coverage.longitude.min.toFixed(3)}–{metadata.geographic_scope.coordinate_coverage.longitude.max.toFixed(3)}° lon</small>}</div><div className={`data-badge ${summary?.data_available ? 'ready' : ''}`}><span />{summary?.data_available ? 'Data processed' : 'No processed data'}</div></div>
    <div className="stats-grid">
      <StatCard icon={Database} label="Daily observations" value={summary?.data_available ? summary.observations.toLocaleString() : '—'} detail="Observed UTC days in selected range" />
      <StatCard icon={Activity} label="Model-target congestion days" value={summary?.congestion_days?.toLocaleString() ?? '—'} detail={!summary?.target_available ? 'Target labels unavailable for this geographic scope' : summary.congestion_days === 0 ? 'No positive target labels in this date range' : `${summary.target_observations} valid target labels`} />
      <StatCard icon={Gauge} label="Model-target event rate" value={percent(summary?.congestion_rate)} detail={summary?.target_available ? `Across ${summary.target_observations} valid target labels` : 'No valid target labels in this range'} />
      <StatCard icon={ShieldCheck} label="Saved-model test samples" value={metrics?.splits?.test.sample_count?.toLocaleString() ?? '—'} detail={metrics?.splits?.test.period ? `Saved evaluation test period · ${metrics.splits.test.period.start} to ${metrics.splits.test.period.end}` : 'Evaluation not available'} />
    </div>
    {rangeLoading && <div className="range-status"><LoaderCircle className="spin" size={15} />Updating selected dates…</div>}
    <div className="content-grid overview-grid">
      <section className="panel chart-panel"><PanelTitle title="Vessel activity across selected range" subtitle={`Up to ${rows.length} actual observation days · ${history.length} days in range`} action={<button className="text-action" onClick={loadHistory}>View port analysis <span>→</span></button>} />
        {rows.length ? <div className="chart-wrap" role="img" aria-label="Daily observed vessel and candidate anchorage counts over the selected date range"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rows} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <defs><linearGradient id="densityFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#246b8a" stopOpacity={0.18} /><stop offset="95%" stopColor="#246b8a" stopOpacity={0} /></linearGradient></defs>
          <CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e5eaf0" /><XAxis dataKey="date" tickFormatter={(value: string) => value.slice(5)} tickLine={false} axisLine={false} minTickGap={24} /><YAxis tickLine={false} axisLine={false} width={38} /><Tooltip labelFormatter={(value) => `${value} UTC`} /><Legend />
          <Area type="monotone" dataKey="vessel_density" name="Observed vessels" stroke="#318b9d" fill="url(#densityFill)" strokeWidth={2} /><Area type="monotone" dataKey="anchorage_vessels" name="Candidate anchorage" stroke="#e19a58" fill="transparent" strokeWidth={2} dot={false} />
        </AreaChart></ResponsiveContainer></div> : <EmptyChart text={summary?.dataset_available ? `No observations in this range. Available data: ${coverageStart || 'unknown'} — ${coverageEnd || 'unknown'}.` : 'No processed daily history is available.'} />}
      </section>
      <section className="panel model-panel"><PanelTitle title="Held-out risk categories" subtitle="Selected-model test predictions · not live port status" />
        {distribution.length ? <><div className="donut-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={64} outerRadius={88} paddingAngle={3} stroke="none">{distribution.map((item) => <Cell key={item.name} fill={item.color} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer><div className="donut-center"><strong>{score?.sample_count ?? '—'}</strong><span>samples</span></div></div><div className="distribution-legend">{distribution.map((item) => <div key={item.name}><span style={{ backgroundColor: item.color }} />{item.name}<strong>{item.value}</strong></div>)}</div></> : <EmptyChart text="Evaluation becomes available after model training." />}
      </section>
    </div>
    <section className="note-strip"><div className="note-icon"><CircleAlert size={18} /></div><div><strong>{metadata?.target_status === 'unavailable' ? 'Congestion target unavailable' : 'Interpret the target correctly'}</strong><p>{metadata?.target_definition ?? 'Next UTC day crosses the configured candidate-anchorage vessel-count threshold.'}{metadata?.target_rule ? ` Candidate-anchorage rule: ${metadata.target_rule}.` : ''}{metadata?.target_source ? ` ${metadata.target_source}` : ''} AIS-derived candidate anchorage counts and proxy labels are kept separately from synthetic target labels. None are verified shipment delivery outcomes.</p></div><span className="horizon-pill">HORIZON <b>1 DAY</b></span></section>
  </>
}

function PortAnalysis({ portList, summary, rows, loading, datasetAvailable, period, setPeriod, range, setCustomStart, setCustomEnd, coverageStart, coverageEnd }: {
  portList: Port[]; summary: Summary | null; rows: HistoryRow[]; loading: boolean; datasetAvailable: boolean
  period: DatePeriod; setPeriod: Dispatch<SetStateAction<DatePeriod>>
  range: { start: string; end: string }
  setCustomStart: Dispatch<SetStateAction<string>>; setCustomEnd: Dispatch<SetStateAction<string>>
  coverageStart: string; coverageEnd: string | null
}) {
  const data = rows
  const chartData = sampleHistory(data, 500)
  const tableRows = data.slice(0, 14)
  const latestDensity = latestValidObservation(data, 'vessel_density')
  const latestAnchorage = latestValidObservation(data, 'anchorage_vessels')
  const latestDwell = latestValidObservation(data, 'anchorage_dwell_hours')
  return <>
    <PageHeading eyebrow="PORT INTELLIGENCE / HISTORICAL ANALYSIS" title="Port activity analysis" description="Explore observed vessel activity and derived congestion labels for the available geographic scope." />
    <div className="analysis-controls panel"><div className="control-field"><label className="field-label" htmlFor="port-select">GEOGRAPHIC SCOPE</label><select id="port-select" value={portList[0]?.port_id ?? PORT_ID} disabled={portList.length <= 1}>{portList.map((port) => <option key={port.port_id} value={port.port_id}>{port.name}</option>)}</select></div><DateRangeSelector {...{ period, setPeriod, range, setCustomStart, setCustomEnd, coverageStart, coverageEnd }} /></div>
    {loading ? <div className="loading-state"><LoaderCircle className="spin" size={22} />Loading port history…</div> : rows.length ? <>
      <div className="stats-grid compact-stats"><StatCard icon={Ship} label="Latest observed vessels" value={latestDensity?.vessel_density.toLocaleString() ?? '—'} detail={latestDensity ? `On ${latestDensity.date}` : 'No valid observations in this range'} /><StatCard icon={Anchor} label="Latest candidate anchorage" value={latestAnchorage?.anchorage_vessels?.toLocaleString() ?? '—'} detail={latestAnchorage ? `Distinct MMSIs · ${latestAnchorage.date} · configured geofence rule` : 'No valid observations in this range'} /><StatCard icon={Activity} label="Dwell proxy" value={latestDwell?.anchorage_dwell_hours == null ? '—' : `${latestDwell.anchorage_dwell_hours.toFixed(1)} h`} detail={latestDwell ? `Within-day AIS observation span · ${latestDwell.date}` : 'No sufficient dwell observations in this range'} /></div>
      <div className="stats-grid compact-stats"><StatCard icon={Activity} label="Model-target congestion days" value={summary?.congestion_days?.toLocaleString() ?? '—'} detail={!summary?.target_available ? 'Target labels unavailable for this geographic scope' : summary.congestion_days === 0 ? 'No positive target labels in this date range' : `${summary.target_observations} valid target labels`} /><StatCard icon={Gauge} label="Model-target event rate" value={percent(summary?.congestion_rate)} detail={summary?.target_available ? `Across ${summary.target_observations} valid target labels` : 'No valid target labels in this range'} /></div>
      <section className="panel chart-panel large-chart"><PanelTitle title="Observed and derived vessel activity" subtitle={`${chartData.length} actual dates sampled from ${data.length} observed days in range`} /><div className="chart-wrap tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData} margin={{ top: 18, right: 12, left: -14, bottom: 8 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9edf1" /><XAxis dataKey="date" tickFormatter={(value: string) => value.slice(5)} tickLine={false} axisLine={false} minTickGap={22} /><YAxis tickLine={false} axisLine={false} /><Tooltip /><Legend /><Line type="monotone" dataKey="vessel_density" name="Observed vessels" stroke="#318b9d" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="anchorage_vessels" name="Candidate anchorage vessels" stroke="#e19a58" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div></section>
      <section className="panel chart-panel"><PanelTitle title="Synthetic label and observation proxy" subtitle="Separate series · 0 = event not flagged · 1 = event flagged" />{data.some((row) => row.ais_congestion_proxy != null || row.synthetic_congestion_label != null) ? <div className="chart-wrap short" role="img" aria-label="Daily synthetic congestion labels and independently calculated observation proxy"><ResponsiveContainer width="100%" height="100%"><BarChart data={sampleHistory(data, 30)}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e5eaf0" /><XAxis dataKey="date" tickFormatter={(value: string) => value.slice(5)} tickLine={false} axisLine={false} minTickGap={20} /><YAxis domain={[0, 1]} ticks={[0, 1]} tickLine={false} axisLine={false} /><Tooltip labelFormatter={(value) => `${value} UTC`} /><Legend /><Bar dataKey="synthetic_congestion_label" name="Synthetic congestion label" fill="#bd7a20" radius={[3, 3, 0, 0]} /><Bar dataKey="ais_congestion_proxy" name="Observation-derived 25-vessel proxy" fill="#318b9d" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div> : <EmptyChart text="No separate synthetic labels or observation-derived congestion proxy are available." />}</section>
      <section className="panel history-table-panel"><PanelTitle title="Daily observations in selected range" subtitle={`First ${Math.min(14, tableRows.length)} of ${data.length} observed rows · no missing dates are filled`} /><div className="table-scroll"><table><thead><tr><th scope="col">UTC date</th><th scope="col">Observed vessels</th><th scope="col">Candidate anchorage</th><th scope="col">Dwell proxy (h)</th><th scope="col">Model target</th><th scope="col">Synthetic label</th><th scope="col">AIS proxy</th></tr></thead><tbody>{tableRows.slice().reverse().map((row) => <tr key={row.date}><th scope="row">{row.date}</th><td>{row.vessel_density.toLocaleString()}</td><td>{row.anchorage_vessels?.toLocaleString() ?? '—'}</td><td>{row.anchorage_dwell_hours == null ? '—' : row.anchorage_dwell_hours.toFixed(1)}</td><td>{row.next_day_congestion == null ? '—' : row.next_day_congestion ? 'Event' : 'No event'}</td><td>{row.synthetic_congestion_label == null ? '—' : row.synthetic_congestion_label ? 'Event' : 'No event'}</td><td>{row.ais_congestion_proxy == null ? '—' : row.ais_congestion_proxy ? 'Event' : 'No event'}</td></tr>)}</tbody></table></div></section>
    </> : <section className="panel"><EmptyChart text={datasetAvailable ? `No observations in this range. Available data: ${coverageStart || 'unknown'} — ${coverageEnd || 'unknown'}.` : 'No processed history is available for this port scope.'} /></section>}
  </>
}

function DatasetManager({ active, metadata, onUploaded }: {
  active: ActiveDataset | null; metadata: Metadata
  onUploaded: (sha256: string) => Promise<void>
}) {
  const [file, setFile] = useState<File | null>(null)
  const [sourceName, setSourceName] = useState('')
  const [validation, setValidation] = useState<DatasetValidation | null>(null)
  const [validating, setValidating] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState<UploadResult | null>(null)
  const [dragging, setDragging] = useState(false)

  async function inspectFile(candidate: File) {
    setError(''); setValidation(null); setResult(null); setFile(null)
    if (!candidate.name.toLowerCase().endsWith('.csv')) {
      setError('Choose a CSV file. Required columns: MMSI, BaseDateTime, LAT, LON, SOG.')
      return
    }
    if (candidate.size > MAX_UPLOAD_BYTES) {
      setError(`The selected file is ${formatBytes(candidate.size)}. The upload limit is 25 MiB.`)
      return
    }
    if (candidate.size === 0) {
      setError('The selected CSV file is empty.')
      return
    }
    setFile(candidate)
    setValidating(true)
    try {
      const body = new FormData()
      body.append('file', candidate)
      const url = `${API}/api/v1/datasets/validate`
      let response: Response
      try {
        response = await fetch(url, { method: 'POST', body })
      } catch (caught) {
        const reason = caught instanceof Error ? ` (${caught.message})` : ''
        throw new Error(`POST ${url} failed before receiving a response${reason}. Check API reachability and CORS for ${window.location.origin}.`, { cause: caught })
      }
      let payload: unknown
      try {
        payload = await response.json() as unknown
      } catch (caught) {
        const reason = caught instanceof Error ? `: ${caught.message}` : ''
        throw new Error(`POST ${url} returned invalid JSON (${response.status})${reason}`, { cause: caught })
      }
      if (response.ok) {
        setValidation(payload as DatasetValidation)
      } else {
        const detail = payload && typeof payload === 'object' && 'detail' in payload
          ? (payload as { detail: { validation?: DatasetValidation; message?: string } }).detail
          : null
        if (detail?.validation) setValidation(detail.validation)
        else throw new Error(detail?.message ?? `POST ${url} returned ${response.status} ${response.statusText}.`)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to validate the selected CSV.')
      setFile(null)
    } finally {
      setValidating(false)
    }
  }

  function uploadFile(candidate: File, source: string): Promise<UploadResult> {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest()
      request.open('POST', `${API}/api/v1/datasets/upload`)
      request.upload.onprogress = (event) => {
        if (event.lengthComputable) setProgress(Math.round(event.loaded / event.total * 100))
      }
      request.onload = () => {
        let payload: unknown
        try { payload = JSON.parse(request.responseText) as unknown } catch {
          reject(new Error(`POST ${API}/api/v1/datasets/upload returned invalid JSON (${request.status}).`))
          return
        }
        if (request.status >= 200 && request.status < 300) {
          resolve(payload as UploadResult)
          return
        }
        const detail = payload && typeof payload === 'object' && 'detail' in payload
          ? (payload as { detail: string | { message?: string; validation?: DatasetValidation } }).detail
          : null
        const message = typeof detail === 'string' ? detail : detail?.message
        reject(new Error(`POST ${API}/api/v1/datasets/upload returned ${request.status}${message ? `: ${message}` : ''}`))
      }
      request.onerror = () => reject(new Error(`POST ${API}/api/v1/datasets/upload failed before receiving a response.`))
      request.ontimeout = () => reject(new Error(`POST ${API}/api/v1/datasets/upload timed out.`))
      const body = new FormData()
      body.append('file', candidate)
      if (source.trim()) body.append('source_name', source.trim())
      request.send(body)
    })
  }

  async function processDataset() {
    if (!file || !validation?.valid) return
    setError(''); setResult(null); setUploading(true); setProgress(0)
    try {
      const uploaded = await uploadFile(file, sourceName)
      setResult(uploaded)
      try {
        await onUploaded(uploaded.sha256)
      } catch (caught) {
        const reason = caught instanceof Error ? caught.message : 'unknown refresh error'
        setError(`The dataset was activated, but the dashboard could not confirm and display it: ${reason}`)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Dataset processing failed. The active dataset was not replaced.')
    } finally {
      setUploading(false)
    }
  }

  function dropFile(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files[0]
    if (dropped) void inspectFile(dropped)
  }

  const previewColumns = validation?.columns.slice(0, 6) ?? []
  return <>
    <PageHeading eyebrow="DATA / DATASET MANAGER" title="Manage dataset" description="Validate and process raw AIS-message CSV data. The configured Los Angeles geofence is identified only when all uploaded coordinates fall within it." />
    <div className="dataset-grid">
      <section className="panel dataset-upload-panel">
        <PanelTitle title="Upload CSV" subtitle="The current dataset remains active unless the new file validates and processes successfully." />
        <div className="required-columns"><strong>Required AIS columns</strong><span>MMSI · BaseDateTime · LAT · LON · SOG</span><small>Use nine-digit MMSI identifiers, parseable timestamps, numeric latitude/longitude and 0–102.3 knot speed values. Raw AIS messages only; processed features or shipment-delay labels are not accepted.</small></div>
        <label className={`drop-zone ${dragging ? 'drop-zone-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={dropFile}>
          <input type="file" accept=".csv,text/csv" onChange={(event) => { const selected = event.target.files?.[0]; if (selected) void inspectFile(selected); event.currentTarget.value = '' }} />
          <CloudUpload size={27} /><strong>{file ? file.name : 'Drop a CSV file here'}</strong><span>{file ? formatBytes(file.size) : 'or browse your device · maximum 25 MiB'}</span>
          <span className="secondary-button">Choose CSV</span>
        </label>
        <label className="input-field source-input"><span>Dataset source name <small>optional</small></span><input type="text" maxLength={120} value={sourceName} onChange={(event) => setSourceName(event.target.value)} placeholder="e.g. AIS archive, collection date" /></label>
        {validating && <div className="progress-message"><LoaderCircle className="spin" size={16} />Validating columns and records…</div>}
        {validation && <div className={`validation-summary ${validation.valid ? 'validation-valid' : 'validation-invalid'}`} role="status">
          {validation.valid ? <CheckCircle2 size={17} /> : <CircleAlert size={17} />}
          <strong>{validation.valid ? 'Validation passed' : 'Validation needs attention'}</strong>
          <span>{validation.input_rows.toLocaleString()} input rows · {validation.columns.length} columns</span>
        </div>}
        {validation && (validation.errors.length > 0 || validation.warnings.length > 0) && <div className="validation-report">
          {validation.errors.map((item) => <p className="validation-error" key={item}><CircleAlert size={14} />{item}</p>)}
          {validation.warnings.map((item) => <p className="validation-warning" key={item}><CircleAlert size={14} />{item}</p>)}
        </div>}
        {validation?.valid && <div className="quality-grid">
          <span>Missing values<strong>{Object.values(validation.missing_values).reduce((sum, count) => sum + count, 0)}</strong></span>
          <span>Invalid dates<strong>{validation.invalid_dates}</strong></span>
          <span>Duplicate records<strong>{validation.duplicate_records}</strong></span>
        </div>}
        {validation?.preview.length ? <div className="preview-table"><strong>Preview · first {validation.preview.length} rows</strong><div className="table-scroll"><table><thead><tr>{previewColumns.map((column) => <th scope="col" key={column}>{column}</th>)}</tr></thead><tbody>{validation.preview.map((row, index) => <tr key={`${index}-${row.MMSI ?? 'row'}`}>{previewColumns.map((column) => <td key={column}>{row[column] ?? '—'}</td>)}</tr>)}</tbody></table></div></div> : null}
        {error && <div className="form-error" role="alert"><CircleAlert size={16} />{error}</div>}
        {uploading && <div className="upload-progress" role="status"><span>Uploading and processing · {progress}%</span><progress max="100" value={progress} /></div>}
        <button className="primary-button" type="button" onClick={() => void processDataset()} disabled={!file || !validation?.valid || validating || uploading}>
          {uploading ? <LoaderCircle size={17} className="spin" /> : <Database size={17} />}{uploading ? 'Processing dataset…' : 'Process and activate dataset'}
        </button>
        <p className="upload-note">Processing does not retrain the model. Predictions and evaluation continue to use the existing model artifact.</p>
        <p className="upload-note">Historical analytics derive the next-day proxy from MMSI, BaseDateTime, LAT, LON, and SOG. Additional columns such as SyntheticCongestionLabel, queue, weather, and delay minutes are not consumed by the current pipeline or model.</p>
      </section>
      <aside className="panel active-dataset-panel">
        <PanelTitle title="Active dataset" subtitle="Currently used for dashboard history and summary queries." />
        {active?.available ? <>
          <div className="active-dataset-status"><span className="status-dot" />Active · {active.processing_status ?? 'status unavailable'}</div>
          <dl className="dataset-details">
            <div><dt>Source</dt><dd>{active.source_name ?? 'Not reported'}</dd></div>
            <div><dt>File</dt><dd>{active.filename ?? 'Not reported'}</dd></div>
            <div><dt>Geographic scope</dt><dd>{active.geographic_scope?.label ?? 'Not verified'}</dd></div>
            <div><dt>Input rows</dt><dd>{active.input_rows?.toLocaleString() ?? '—'}</dd></div>
            <div><dt>Processed days</dt><dd>{active.processed_rows?.toLocaleString() ?? '—'}</dd></div>
            <div><dt>Date coverage</dt><dd>{active.date_coverage ? `${active.date_coverage.start} — ${active.date_coverage.end} UTC` : 'Unavailable'}</dd></div>
            <div><dt>Last processed</dt><dd>{active.last_processed_at ? new Date(active.last_processed_at).toLocaleString() : 'Unavailable'}</dd></div>
          </dl>
          {active.geographic_scope && <p className="upload-note">{active.geographic_scope.identification_method}</p>}
          {active.target_status === 'unavailable' && <p className="upload-note">Congestion target labels are unavailable for this dataset's geographic scope; no events or event rate are inferred.</p>}
          {active.model_note && <p className="upload-note">{active.model_note}</p>}
          <div className="dataset-model-line"><strong>Loaded model trained on:</strong> {metadata.model_training_dataset?.source_name ?? metadata.model_training_dataset?.data_status ?? 'Dataset identity not recorded'}{metadata.model_training_dataset?.identity_recorded === false ? ' · exact file identity not recorded' : ''}</div>
          <div className="dataset-model-line"><strong>Saved evaluation dataset:</strong> {metadata.evaluation_dataset.source_name ?? metadata.evaluation_dataset.data_status ?? 'Dataset identity not recorded'}{metadata.evaluation_dataset.identity_recorded === false ? ' · exact file identity not recorded' : ''}</div>
        </> : <div className="empty-dataset"><Database size={22} /><span>No active processed dataset is available.</span></div>}
        {result && <div className="upload-success" role="status"><CheckCircle2 size={18} /><div><strong>Dataset processed and activated</strong><span>{result.processed_rows.toLocaleString()} daily rows · {result.date_coverage.start} — {result.date_coverage.end}</span></div></div>}
      </aside>
    </div>
  </>
}

function formatBytes(bytes: number) {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MiB` : `${Math.max(1, Math.round(bytes / 1024))} KiB`
}

function PredictionPage({ form, setForm, onSubmit, result, submitting, error, metadata }: {
  form: FormValues; setForm: Dispatch<SetStateAction<FormValues>>
  onSubmit: (event: FormEvent<HTMLFormElement>) => void; result: Prediction | null
  submitting: boolean; error: string; metadata: Metadata
}) {
  const fields = [
    { key: 'vessel_density', label: 'Observed vessel density', unit: 'vessels', min: 0, max: 100000, step: 1, help: 'Distinct MMSIs observed in the current UTC day.' },
    { key: 'anchorage_vessels', label: 'Candidate anchorage vessels', unit: 'vessels', min: 0, max: 100000, step: 1, help: 'Distinct MMSIs matching the configured geofence and speed rule.' },
    { key: 'anchorage_dwell_hours', label: 'Median dwell proxy', unit: 'hours', min: 0, max: 8760, step: 0.1, help: 'Median within-day AIS observation span; not true waiting time.' },
    { key: 'day_of_week', label: 'Prediction day of week', unit: '0–6', min: 0, max: 6, step: 1, help: 'UTC weekday for the following day (Monday = 0).' },
    { key: 'month', label: 'Prediction month', unit: '1–12', min: 1, max: 12, step: 1, help: 'UTC calendar month for the following day.' },
  ]
  const incompatible = metadata.prediction_compatible === false
  return <>
    <PageHeading eyebrow="PORT INTELLIGENCE / NEXT-DAY ESTIMATE" title="Predict congestion" description="Estimate the defined next-day congestion proxy using current-day indicators." />
    <div className="prediction-layout"><form className="panel prediction-form" onSubmit={onSubmit}><PanelTitle title="Prediction inputs" subtitle="Enter indicators available at the end of the current UTC day." />
      <div className="form-grid">{fields.map((field) => <label className="input-field" key={field.key}><span>{field.label}<small>{field.unit}</small></span><input type="number" required min={field.min} max={field.max} step={field.step} value={form[field.key]} onChange={(event) => setForm((current) => ({ ...current, [field.key]: event.target.value }))} /><small className="input-help">{field.help}</small></label>)}</div>
        <div className="form-footnote"><CircleAlert size={15} />Prototype only — no live port data feed is configured. Model inputs remain manual; the uploaded dataset does not retrain the model.</div>
      {incompatible && <div className="form-error" role="alert"><CircleAlert size={16} />{metadata.prediction_incompatibility ?? 'The active dataset is incompatible with the current model feature schema.'}</div>}
      {error && <div className="form-error" role="alert"><CircleAlert size={16} />{error}</div>}
      <button className="primary-button" type="submit" disabled={submitting || incompatible}>{submitting ? <LoaderCircle size={17} className="spin" /> : <Activity size={17} />}{submitting ? 'Calculating…' : 'Generate prediction'}</button>
    </form><div className="prediction-side">{result ? <PredictionResult prediction={result} /> : <section className="panel prediction-placeholder"><div className="placeholder-icon"><Gauge size={26} /></div><h2>Prediction output</h2><p>Submit input values to view event probability and reliability separately.</p></section>}
      <div className="target-definition panel"><div className="target-icon"><Compass size={17} /></div><div><strong>Target and horizon</strong><p>Next UTC day crosses the fixed candidate-anchorage count threshold. This is not verified shipment delay.</p></div></div>
    </div></div>
  </>
}

function PredictionResult({ prediction }: { prediction: Prediction }) {
  const data = prediction.local_explanations.map((item) => ({ ...item, display: item.probability_change * 100 }))
  return <section className="panel result-panel">
    <div className="result-topline"><span className="result-kicker">NEXT-DAY CONGESTION PROXY</span><span className={`risk-pill risk-${prediction.risk_category}`}>{prediction.risk_category} risk</span></div>
    <p className="prediction-target-copy">{prediction.target_definition} · Horizon: {prediction.prediction_horizon}.</p>
    <div className="probability-row"><strong>{percent(prediction.congestion_probability)}</strong><span>estimated event probability</span></div>
    <div className="probability-track"><span className={`track-fill fill-${prediction.risk_category}`} style={{ width: `${Math.max(2, prediction.congestion_probability * 100)}%` }} /></div>
    <div className="threshold-line">Classification threshold <strong>{percent(prediction.classification_threshold)}</strong><span> · {prediction.probability_calibration.replaceAll('_', ' ')}</span></div>
    <div className="reliability-box"><div className="reliability-heading"><span><ShieldCheck size={16} /> Prediction reliability</span><strong className={`reliability-${prediction.prediction_reliability}`}>{titleCase(prediction.prediction_reliability)}</strong></div><ul>{prediction.reliability_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>
    <div className="explanation-section"><h3>Model sensitivity indicators</h3><p>Probability change when one input is replaced with its training median; not causal or additive.</p><div className="sensitivity-chart"><ResponsiveContainer width="100%" height="100%"><BarChart layout="vertical" data={data} margin={{ top: 4, right: 14, left: 18, bottom: 0 }}><CartesianGrid strokeDasharray="3 5" horizontal={false} stroke="#edf0f2" /><XAxis type="number" tickFormatter={(value: number) => `${value.toFixed(0)}%`} tickLine={false} axisLine={false} /><YAxis type="category" dataKey="feature" tickFormatter={titleCase} width={126} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => [`${Number(value).toFixed(2)} pp`, 'Change from reference']} /><Bar dataKey="display" name="Probability change" radius={[0, 3, 3, 0]}>{data.map((item) => <Cell key={item.feature} fill={item.display >= 0 ? '#318b9d' : '#e19a58'} />)}</Bar></BarChart></ResponsiveContainer></div>
      <div className="explanation-values">{prediction.local_explanations.slice(0, 3).map((item) => <div key={item.feature}><span>{titleCase(item.feature)}</span><strong>{item.observed_value}</strong><small>{item.direction === 'increases' ? <ArrowUpRight size={13} /> : item.direction === 'decreases' ? <ArrowDownRight size={13} /> : null}{item.probability_change >= 0 ? '+' : ''}{(item.probability_change * 100).toFixed(1)} pp vs reference</small></div>)}</div>
    </div>
    <div className="action-list"><h3>Suggested review actions</h3>{prediction.recommended_actions.map((action) => <p key={action}><CheckCircle2 size={15} />{action}</p>)}</div>
    {prediction.data_quality_warnings.length > 0 && <div className="warning-list">{prediction.data_quality_warnings.map((warning) => <p key={warning}><CircleAlert size={14} />{warning}</p>)}</div>}
  </section>
}

function ShipmentDelayPage({ form, setForm, onSubmit, result, submitting, error, metrics }: {
  form: FormValues; setForm: Dispatch<SetStateAction<FormValues>>
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  result: DelayPrediction | null; submitting: boolean; error: string
  metrics: DelayMetrics | null
}) {
  const fields = [
    { key: 'vessel_arrivals_daily', label: 'Daily vessel arrivals', unit: 'vessels', min: 0, max: 100000, step: 1 },
    { key: 'queue_length_vessels', label: 'Queue length', unit: 'vessels', min: 0, max: 100000, step: 1 },
    { key: 'port_capacity_berths', label: 'Port capacity', unit: 'berths', min: 1, max: 100000, step: 1 },
    { key: 'wind_speed_kmh', label: 'Wind speed', unit: 'km/h', min: 0, max: 500, step: 0.1 },
    { key: 'day_of_week', label: 'Target day of week', unit: '0–6 · Monday = 0', min: 0, max: 6, step: 1 },
    { key: 'month', label: 'Target month', unit: '1–12', min: 1, max: 12, step: 1 },
  ]
  return <>
    <PageHeading eyebrow="PORT INTELLIGENCE / SHIPMENT DELAY" title="Shipment Delay Prediction" description="Estimate the supplied synthetic delay-label outcome from operational conditions and the preceding UTC day's context." />
    <div className="synthetic-banner" role="note"><CircleAlert size={17} /><span><strong>Synthetic-data limitation</strong> — The classifier is trained on SyntheticDelayLabel. Its scores are not evidence of real-world shipping performance, and probability is not model confidence.</span></div>
    {!metrics?.available && <div className="form-error" role="status"><CircleAlert size={16} />Shipment-delay model unavailable: {metrics?.reason ?? 'Checking model availability from the backend.'}</div>}
    <div className="prediction-layout">
      <form className="panel prediction-form" onSubmit={onSubmit}>
        <PanelTitle title="Shipment and port conditions" subtitle="Enter the target UTC day and operating conditions from the preceding day." />
        <div className="form-grid">
          {fields.map((field) => <label className="input-field" key={field.key}>
            <span>{field.label}<small>{field.unit}</small></span>
            <input type="number" required min={field.min} max={field.max} step={field.step} value={form[field.key]} onChange={(event) => setForm((current) => ({ ...current, [field.key]: event.target.value }))} />
          </label>)}
          <label className="input-field"><span>Weather condition<small>category</small></span><input type="text" required maxLength={80} list="delay-weather-options" value={form.weather_condition} onChange={(event) => setForm((current) => ({ ...current, weather_condition: event.target.value }))} /><datalist id="delay-weather-options"><option value="Clear" /><option value="Cloudy" /><option value="Rain" /><option value="Storm" /><option value="Fog" /></datalist></label>
        </div>
        <div className="form-footnote"><CircleAlert size={15} />Delay probabilities are uncalibrated; risk bands are Low &lt;30%, Medium 30–&lt;60%, and High ≥60%. Duration is unavailable because no duration model has been evaluated.</div>
        {error && <div className="form-error" role="alert"><CircleAlert size={16} />{error}</div>}
        <button className="primary-button" type="submit" disabled={submitting}>{submitting ? <LoaderCircle size={17} className="spin" /> : <Activity size={17} />}{submitting ? 'Predicting…' : 'Predict shipment delay'}</button>
      </form>
      <div className="prediction-side">
        {result ? <ShipmentDelayResult prediction={result} /> : <section className="panel prediction-placeholder"><div className="placeholder-icon"><Ship size={26} /></div><h2>Shipment-delay output</h2><p>Submit conditions to receive the actual backend classifier response, if the model is available.</p></section>}
        <div className="target-definition panel"><div className="target-icon"><Compass size={17} /></div><div><strong>Target and provenance</strong><p>{metrics?.target_definition ?? 'SyntheticDelayLabel aggregated by target UTC day.'} {metrics?.dataset_identity?.source_name ? `Dataset: ${metrics.dataset_identity.source_name}.` : ''}</p></div></div>
      </div>
    </div>
  </>
}

function ShipmentDelayResult({ prediction }: { prediction: DelayPrediction }) {
  if (prediction.status === 'unavailable') {
    return <section className="panel result-panel" role="status"><div className="result-topline"><span className="result-kicker">MODEL UNAVAILABLE</span></div><p>{prediction.reason ?? 'No valid delay model is loaded.'}</p><p>Predictions are not generated until valid labeled data is available.</p></section>
  }
  return <section className="panel result-panel">
    <div className="result-topline"><span className="result-kicker">SHIPMENT DELAY CLASSIFICATION</span><span className={`risk-pill risk-${prediction.risk_category?.toLowerCase()}`}>{prediction.risk_category} risk</span></div>
    <p className="prediction-target-copy">{prediction.target_definition}</p>
    {prediction.evaluation_warning && <div className="warning-list"><p><CircleAlert size={14} />{prediction.evaluation_warning}</p></div>}
    <div className="probability-row"><strong>{prediction.predicted_outcome === 'delayed' ? 'Delayed' : 'Not delayed'}</strong><span>predicted outcome</span></div>
    <div className="probability-row"><strong>{prediction.delay_probability_percent?.toFixed(1)}%</strong><span>delay probability · {prediction.probability_calibration}</span></div>
    <p className="calibration-note">{prediction.probability_note} Risk thresholds: Low &lt;{(prediction.risk_thresholds?.low ?? 0.3) * 100}%, Medium below {(prediction.risk_thresholds?.medium ?? 0.6) * 100}% after the Low threshold, High at or above {(prediction.risk_thresholds?.medium ?? 0.6) * 100}%.</p>
    <div className="probability-track"><span className={`track-fill fill-${prediction.risk_category?.toLowerCase() === 'medium' ? 'moderate' : prediction.risk_category?.toLowerCase()}`} style={{ width: `${Math.max(2, Math.min(100, prediction.delay_probability_percent ?? 0))}%` }} /></div>
    <div className="threshold-line">Estimated delay duration <strong>{prediction.estimated_delay_minutes == null ? 'Unavailable' : `${prediction.estimated_delay_minutes.toFixed(0)} min`}</strong><span> · {prediction.duration_status}</span></div>
    <div className="explanation-section"><h3>Main contributing factors</h3><p>One-feature-at-a-time probability sensitivity against training reference values; not causal or additive.</p><div className="explanation-values">{prediction.main_factors.map((factor) => <div key={factor.feature}><span>{titleCase(factor.feature)}</span><strong>{String(factor.observed_value)}</strong><small>{factor.direction.replace('_', ' ')} · {factor.probability_change >= 0 ? '+' : ''}{(factor.probability_change * 100).toFixed(1)} pp</small></div>)}</div></div>
    <div className="action-list"><h3>Practical recommendations</h3>{prediction.recommendations.map((recommendation) => <p key={recommendation}><CheckCircle2 size={15} />{recommendation}</p>)}</div>
    <p className="calibration-note">Training source: {prediction.dataset_provenance.source_name ?? prediction.dataset_provenance.data_status ?? 'not recorded'} · Model {prediction.model_version}. Synthetic-data results do not establish real shipping performance.</p>
  </section>
}

function DelayPerformance({ metrics, activeDataset }: {
  metrics: DelayMetrics | null; activeDataset: ActiveDataset | null
}) {
  if (!metrics?.available || !metrics.test) {
    return <section className="panel chart-panel"><PanelTitle title="Shipment-delay model evaluation" subtitle="Separate chronological holdout metrics" /><EmptyChart text={metrics?.reason ?? 'Shipment-delay evaluation is unavailable.'} /></section>
  }
  const test = metrics.test
  const trainingSha = metrics.training_dataset?.sha256 ?? metrics.dataset_identity?.sha256
  const datasetMismatch = Boolean(activeDataset?.sha256 && trainingSha && activeDataset.sha256 !== trainingSha)
  return <section className="panel chart-panel delay-performance">
    <PanelTitle title="Shipment-delay classification" subtitle="Separate model · supplied SyntheticDelayLabel · chronological final test" />
    <p className="calibration-note">Dataset: <strong>{metrics.dataset_identity?.source_name ?? metrics.dataset_identity?.data_status ?? 'not recorded'}</strong>{metrics.dataset_identity?.filename ? ` · ${metrics.dataset_identity.filename}` : ''}. Target: {metrics.target_definition} Probabilities are {metrics.probability_calibration}; Brier and log-loss are reported on that uncalibrated output.</p>
    {datasetMismatch && <p className="calibration-note">The active dataset differs from this shipment-delay model's training/evaluation dataset. Dataset uploads do not automatically retrain the delay model.</p>}
    <div className="performance-meta panel"><div><span>MODEL</span><strong>{titleCase(metrics.model ?? '—')}</strong></div><div><span>FINAL TEST PERIOD</span><strong>{test.period.start} — {test.period.end}</strong></div><div><span>TEST SAMPLES</span><strong>{test.sample_count.toLocaleString()}</strong></div><div><span>TRAIN SAMPLES</span><strong>{metrics.train?.sample_count.toLocaleString() ?? '—'}</strong></div></div>
    <div className="metric-grid">{[['Precision', test.precision], ['Recall', test.recall], ['F1 score', test.f1], ['ROC-AUC', test.roc_auc], ['Brier score', test.brier_score], ['Log loss', test.log_loss]].map(([name, value]) => <div className="metric-card" key={String(name)}><span>{name}</span><strong>{metric(value as number | null)}</strong></div>)}</div>
    <div className="matrix"><span /><span>Predicted not delayed</span><span>Predicted delayed</span><strong>Actual not delayed</strong><b>{test.confusion_matrix[0]?.[0] ?? '—'}</b><b>{test.confusion_matrix[0]?.[1] ?? '—'}</b><strong>Actual delayed</strong><b>{test.confusion_matrix[1]?.[0] ?? '—'}</b><b>{test.confusion_matrix[1]?.[1] ?? '—'}</b></div>
    <p className="calibration-note">Target distribution: {metrics.target_distribution?.not_delayed ?? '—'} not delayed, {metrics.target_distribution?.delayed ?? '—'} delayed. No duration regression was evaluated.</p>
  </section>
}

function Performance({ metrics, delayMetrics, activeDataset, metadata }: {
  metrics: Metrics | null; delayMetrics: DelayMetrics | null
  activeDataset: ActiveDataset | null; metadata: Metadata
}) {
  const selected = metrics?.test?.selected_model
  const importance = Object.entries(metrics?.global_feature_importance ?? {}).map(([feature, value]) => ({ feature: titleCase(feature), value }))
  const comparisonMetrics = metrics?.comparison_test_calibrated_models ?? metrics?.comparison_test_raw_models
  const comparison = Object.entries(comparisonMetrics ?? {}).map(([name, item]) => ({
    model: name === 'logistic_regression' ? 'Logistic Regression'
      : name === 'logistic_regression_balanced' ? 'Balanced Logistic'
        : 'Random Forest',
    Accuracy: item.accuracy,
    Precision: item.precision,
    Recall: item.recall,
    F1: item.f1,
  }))
  const thresholdSummary = Object.entries(metrics?.threshold_selection?.model_thresholds ?? {})
    .map(([name, item]) => `${titleCase(name)} ${percent(item.threshold)}`)
    .join(' · ')
  return <>
    <PageHeading eyebrow="PORT INTELLIGENCE / VALIDATION" title="Model performance" description="Chronological holdout evaluation, calibration evidence, and global model associations." />
    {!metrics?.available || !selected ? <section className="panel"><EmptyChart text="Model metrics are unavailable until evaluation has been generated. The API has not provided final test results." /></section> : <>
      <p className="calibration-note">Loaded model training dataset: <strong>{metadata.model_training_dataset?.source_name ?? metadata.model_training_dataset?.data_status ?? 'identity not recorded'}</strong>{metadata.model_training_dataset?.filename ? ` · ${metadata.model_training_dataset.filename}` : ''}. Saved evaluation dataset: <strong>{metrics.training_dataset?.source_name ?? metrics.training_dataset?.data_status ?? metrics.data_status ?? 'not recorded'}</strong>{metrics.training_dataset?.filename ? ` · ${metrics.training_dataset.filename}` : ''}{metrics.training_dataset?.identity_recorded === false ? ' · exact file identity was not recorded in this artifact' : ''}. These metrics remain the saved model evaluation; uploads do not retrain or recalculate them.</p>
      <div className="performance-meta panel"><div><span>SELECTED MODEL</span><strong>{titleCase(metrics.selected_model ?? '—')}</strong></div><div><span>SAVED EVALUATION TEST PERIOD</span><strong>{metrics.splits?.test.period.start} — {metrics.splits?.test.period.end}</strong></div><div><span>TEST SAMPLES</span><strong>{metrics.splits?.test.sample_count}</strong></div><div><span>CALIBRATION</span><strong>{metrics.calibration?.status.replaceAll('_', ' ')}</strong></div></div>
      <p className="calibration-note">Saved model target: <strong>{metrics.target_status === 'provided' ? 'Supplied congestion label' : 'AIS-derived congestion proxy'}</strong>. {metrics.target_definition}{metrics.target_source ? ` Source: ${metrics.target_source}` : ''}</p>
      <div className="metric-grid">{[['Accuracy', selected.accuracy], ['Precision', selected.precision], ['Recall', selected.recall], ['F1 score', selected.f1], ['ROC-AUC', selected.roc_auc], ['Brier score', selected.brier_score], ['Log loss', selected.log_loss]].map(([name, value]) => <div className="metric-card" key={String(name)}><span>{name}</span><strong>{metric(value as number | null)}</strong>{name === 'ROC-AUC' && selected.roc_auc == null && <small>{selected.roc_auc_reason ?? 'Unavailable for this test sample.'}</small>}</div>)}</div>
      <section className="evaluation-partitions panel" aria-label="Evaluation data partitions">
        <div><span>TRAINING</span><strong>{metrics.splits?.train?.sample_count?.toLocaleString() ?? '—'}</strong><small>{metrics.splits?.train?.period ? `${metrics.splits.train.period.start} — ${metrics.splits.train.period.end}` : 'Not reported'}</small></div>
        <div><span>CALIBRATION</span><strong>{metrics.splits?.calibration?.sample_count?.toLocaleString() ?? '—'}</strong><small>{metrics.splits?.calibration?.period ? `${metrics.splits.calibration.period.start} — ${metrics.splits.calibration.period.end}` : 'Not reported'}</small></div>
        <div><span>VALIDATION</span><strong>{metrics.splits?.validation?.sample_count?.toLocaleString() ?? '—'}</strong><small>{metrics.splits?.validation?.period ? `${metrics.splits.validation.period.start} — ${metrics.splits.validation.period.end}` : 'Not reported'}</small></div>
        <div><span>FINAL TEST</span><strong>{metrics.splits?.test.sample_count.toLocaleString() ?? '—'}</strong><small>{metrics.splits?.test.period ? `${metrics.splits.test.period.start} — ${metrics.splits.test.period.end}` : 'Not reported'}</small></div>
      </section>
      <div className="content-grid performance-grid">
        <section className="panel chart-panel"><PanelTitle title="Baseline vs candidate" subtitle={metrics.comparison_test_calibrated_models ? 'Final test metrics · calibrated scores · thresholds fixed on validation' : 'Final test metrics · raw classifiers · fixed 0.5 cutoff'} />{comparison.length ? <><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={comparison} margin={{ top: 10, right: 8, left: -18, bottom: 14 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9edf1" /><XAxis dataKey="model" interval={0} tick={{ fontSize: 10 }} tickLine={false} axisLine={false} /><YAxis domain={[0, 1]} tickLine={false} axisLine={false} /><Tooltip /><Legend /><Bar dataKey="Accuracy" fill="#688299" radius={[3, 3, 0, 0]} /><Bar dataKey="Precision" fill="#348b9c" radius={[3, 3, 0, 0]} /><Bar dataKey="Recall" fill="#e19a58" radius={[3, 3, 0, 0]} /><Bar dataKey="F1" fill="#596d83" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div>{thresholdSummary && <p className="calibration-note">Validation-tuned thresholds (not chosen on the final test): {thresholdSummary}.</p>}</> : <EmptyChart text="Comparable model test results unavailable." />}</section>
        <section className="panel chart-panel"><PanelTitle title="Global feature importance" subtitle="Ranked normalized selected-model importance · not causal" />{importance.length ? <><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={importance} layout="vertical" margin={{ top: 8, right: 16, left: 22, bottom: 0 }}><CartesianGrid strokeDasharray="3 5" horizontal={false} stroke="#e9edf1" /><XAxis type="number" domain={[0, 'dataMax']} tickLine={false} axisLine={false} /><YAxis type="category" dataKey="feature" width={135} tickLine={false} axisLine={false} /><Tooltip /><Bar dataKey="value" name="Importance" fill="#318b9d" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div><p className="feature-interpretation">Strongest ranked association: <strong>{importance[0].feature}</strong> ({(importance[0].value * 100).toFixed(1)}%). Importance describes model association; it does not establish causality.</p></> : <EmptyChart text="Feature importance unavailable." />}</section>
      </div>
      <div className="content-grid performance-grid">
        <section className="panel chart-panel"><PanelTitle title="Probability calibration" subtitle="Final held-out test data · closer to the diagonal is better" />{metrics.calibration_curve?.points?.length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={metrics.calibration_curve.points} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}><CartesianGrid strokeDasharray="3 5" vertical={false} stroke="#e9edf1" /><XAxis type="number" dataKey="mean_predicted_probability" domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} /><YAxis type="number" dataKey="observed_event_fraction" domain={[0, 1]} tickFormatter={(value: number) => `${Math.round(value * 100)}%`} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} /><Legend /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#95a2aa" strokeDasharray="4 4" name="Ideal calibration" /><Line type="linear" dataKey="observed_event_fraction" name="Observed event rate" stroke="#318b9d" strokeWidth={2} dot={{ r: 4 }} /></LineChart></ResponsiveContainer></div> : <EmptyChart text={metrics.calibration_curve?.unavailable_reason ?? 'Calibration curve unavailable.'} />}<p className="calibration-note"><ShieldCheck size={15} />Calibration fit Brier score: {metric(metrics.calibration?.calibration_fit_brier_score)} (in-sample on calibration data); final test Brier: {metric(selected.brier_score)}.</p></section>
        <section className="panel confusion-panel"><PanelTitle title="Confusion matrix" subtitle={`Final test set · actual rows × predicted columns · validation threshold ${percent(selected.classification_threshold)}`} /><div className="matrix"><span /><span>Predicted no event</span><span>Predicted event</span><strong>Actual no event</strong><b>{selected.confusion_matrix?.[0]?.[0] ?? '—'}</b><b>{selected.confusion_matrix?.[0]?.[1] ?? '—'}</b><strong>Actual event</strong><b>{selected.confusion_matrix?.[1]?.[0] ?? '—'}</b><b>{selected.confusion_matrix?.[1]?.[1] ?? '—'}</b></div><p className="calibration-note">The selected model's fixed-0.5 test result is separately retained in the evaluation artifact.</p></section>
      </div>
    </>}
    <DelayPerformance metrics={delayMetrics} activeDataset={activeDataset} />
  </>
}

function MethodCard({ number, title, icon: Icon, children }: { number: string; title: string; icon: typeof Activity; children: ReactNode }) {
  return <article className="method-card panel"><div className="method-card-top"><span>{number}</span><Icon size={18} /></div><h2>{title}</h2>{children}</article>
}
function Methodology({ metadata }: { metadata: Metadata | null }) {
  const limitations = metadata?.limitations ?? ['No verified delivery labels.', 'Synthetic metrics are not real-world evidence.', 'Geofence and threshold require operational validation.']
  return <>
    <PageHeading eyebrow="PORT INTELLIGENCE / TRANSPARENCY" title="Methodology & limitations" description="What is measured, what is inferred, and what must not be claimed from this prototype." />
    <section className="method-hero panel"><div className="method-icon"><Compass size={21} /></div><div><span>MODEL TARGET</span><h2>{metadata?.target_status === 'provided' ? 'Supplied congestion label' : 'Next-day AIS-derived congestion proxy'}</h2><p>{metadata?.target_definition ?? 'Next UTC day has at least 25 observed MMSIs in the candidate-anchorage geofence.'}{metadata?.target_source ? ` Source: ${metadata.target_source}` : ''}</p><div className="method-pills"><span>Horizon: following UTC day</span><span>Cutoff: end of observed UTC day</span><span>Classification threshold: validation-tuned</span></div></div></section>
    <div className="method-grid">
      <MethodCard number="01" title="Data and provenance" icon={Database}><p>Dataset status: <strong>{metadata?.data_status ?? 'not processed'}</strong>. {metadata?.synthetic ? 'The current dataset is synthetic development data generated to exercise the pipeline; it is not real AIS evidence.' : metadata?.data_status === 'public' ? 'The current processed dataset is identified as public-source data.' : 'No processed dataset provenance is currently reported.'}</p><p>Source: {metadata?.source_name ?? 'Not reported'}{metadata?.source_url && <> · <a href={metadata.source_url} target="_blank" rel="noreferrer">View source</a></>}{metadata?.acquisition_date ? ` · Acquired ${metadata.acquisition_date}` : ''}. Provenance tracks the source, import date, file hash, schema, coverage, missingness, duplicates, and processing counts.</p></MethodCard>
      <MethodCard number="02" title="Target construction" icon={Anchor}><p>A positive label means the following day has at least 25 unique MMSIs matching a coarse coordinate geofence and speed ≤0.5 knots. The development zone is not a verified port authority boundary.</p><p>No shipment schedules, actual delivery times, berth queues, or validated waiting labels are available.</p></MethodCard>
      <MethodCard number="03" title="Features and leakage" icon={Activity}><p>Features: same-day observed MMSI count, candidate-anchorage MMSI count, median within-day anchorage message span, and known next-day weekday/month. Vessel identity and future observations are excluded.</p><p>Missing dwell is encoded as zero and is not measured zero waiting.</p></MethodCard>
      <MethodCard number="04" title="Training and evaluation" icon={BarChart3}><p>Unweighted and balanced Logistic Regression plus balanced Random Forest use chronological 60%/15%/15%/10% training, calibration, validation, and final test partitions. Model choice uses validation Brier score; sigmoid calibration uses the earlier calibration period; the classification threshold maximizes validation F1 after calibration. The final test is not used to choose the model, calibrator, or threshold.</p><p>Unsupported class metrics are unavailable rather than fabricated.</p></MethodCard>
      <MethodCard number="05" title="Probability vs reliability" icon={ShieldCheck}><p>Congestion probability estimates the defined event. Prediction reliability separately reports high, moderate, low, or insufficient evidence based on sample coverage, calibration evidence, and input range checks.</p><p>Reliability is not the probability that the model is correct, nor 1 minus congestion probability.</p></MethodCard>
      <MethodCard number="06" title="Explanations and use" icon={Waves}><p>Global feature importance is association, not causality. Local sensitivity changes one input to its training median and measures model probability change; effects are not additive.</p><p>Recommendations support human review only. Do not use this prototype to reroute vessels or make consequential decisions.</p></MethodCard>
    </div>
    <section className="limitations-panel panel"><div><CircleAlert size={18} /><strong>Key limitations</strong></div><ul>{limitations.map((item) => <li key={item}>{item}</li>)}</ul><p>Dataset status: <strong>{metadata?.data_status ?? 'not processed'}</strong> · Processed observations: <strong>{metadata?.processed_rows?.toLocaleString() ?? '—'}</strong>{metadata?.model_version ? <> · Model version: <strong>{metadata.model_version}</strong></> : null}</p></section>
  </>
}
export default App
