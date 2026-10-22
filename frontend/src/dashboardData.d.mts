export type DatePeriod = '1m' | '6m' | '1y' | '2y' | '3y' | 'custom'

export type DateRange = { start: string; end: string }

export type HistoryRow = {
  date: string
  vessel_density: number
  anchorage_vessels: number | null
  anchorage_dwell_hours: number | null
  ais_congestion_proxy?: number | null
  synthetic_congestion_label?: number | null
  next_day_congestion: number | null
}

export type DashboardSummary = {
  data_available: boolean
  dataset_available: boolean
  port_id: string
  port_name: string
  observations: number
  target_observations: number
  target_available: boolean
  congestion_days: number | null
  congestion_rate: number | null
  coverage: DateRange | null
  synthetic: boolean
}

export function buildPeriodRange(
  period: DatePeriod,
  coverageStart: string,
  coverageEnd: string | null,
  customStart: string,
  customEnd: string,
): DateRange

export function describePeriodCoverage(
  period: DatePeriod,
  coverageStart: string,
  coverageEnd: string | null,
  range: DateRange,
): string

export function filterHistory(rows: HistoryRow[], range: DateRange): HistoryRow[]

export function sampleHistory(rows: HistoryRow[], maximum?: number): HistoryRow[]

export function latestValidObservation(
  rows: HistoryRow[],
  field: 'vessel_density' | 'anchorage_vessels' | 'anchorage_dwell_hours',
): HistoryRow | undefined

export function summarizeHistory(
  rows: HistoryRow[],
  portId: string,
  portName: string,
  datasetAvailable: boolean,
  synthetic: boolean,
): DashboardSummary

export function datasetsMatch(
  activeDataset: { sha256?: string | null; data_status?: string | null } | null,
  evaluationDataset: { sha256?: string | null; data_status?: string | null } | null,
): boolean | null

export function dashboardMatchesDataset(
  snapshot: { active_dataset?: { sha256?: string | null } | null },
  expectedSha256: string,
): boolean

export function createRequestGate(): {
  next(): number
  isCurrent(token: number): boolean
  invalidate(): void
}
