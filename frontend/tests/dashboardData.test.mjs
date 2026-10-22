import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildPeriodRange,
  createRequestGate,
  dashboardMatchesDataset,
  describePeriodCoverage,
  datasetsMatch,
  filterHistory,
  latestValidObservation,
  sampleHistory,
  summarizeHistory,
} from '../src/dashboardData.mjs'

const rows = [
  { date: '2024-12-01', vessel_density: 8, anchorage_vessels: 1, anchorage_dwell_hours: 0, next_day_congestion: 0 },
  { date: '2024-12-02', vessel_density: 12, anchorage_vessels: 2, anchorage_dwell_hours: 2, next_day_congestion: 0 },
  { date: '2024-12-03', vessel_density: 15, anchorage_vessels: 3, anchorage_dwell_hours: 4, next_day_congestion: 1 },
]

test('date ranges stay inside the active dataset coverage', () => {
  assert.deepEqual(
    buildPeriodRange('3y', '2024-12-01', '2024-12-30', '', ''),
    { start: '2024-12-01', end: '2024-12-30' },
  )
  assert.deepEqual(
    buildPeriodRange('custom', '2024-12-01', '2024-12-30', '2024-11-01', '2025-01-01'),
    { start: '2024-12-01', end: '2024-12-30' },
  )
})

test('one- and six-month ranges use latest dataset date and available observations', () => {
  const oneMonth = buildPeriodRange('1m', '2024-01-01', '2024-08-01', '', '')
  const sixMonths = buildPeriodRange('6m', '2024-01-01', '2024-08-01', '', '')
  assert.deepEqual(oneMonth, { start: '2024-07-01', end: '2024-08-01' })
  assert.deepEqual(sixMonths, { start: '2024-02-01', end: '2024-08-01' })
  const sparseObservedDates = [
    { date: '2024-02-01' },
    { date: '2024-07-01' },
    { date: '2024-08-01' },
  ]
  assert.equal(filterHistory(sparseObservedDates, oneMonth).length, 2)
  assert.equal(filterHistory(sparseObservedDates, sixMonths).length, 3)
  assert.deepEqual(sampleHistory(sparseObservedDates, 2), [
    sparseObservedDates[0],
    sparseObservedDates[2],
  ])
  assert.equal(
    describePeriodCoverage('6m', '2024-01-01', '2024-08-01', sixMonths),
    '2024-02-01 — 2024-08-01 UTC · observed dates only',
  )
})

test('a longer selected period reports short coverage without fabricating dates', () => {
  const range = buildPeriodRange('6m', '2025-05-01', '2025-05-31', '', '')
  assert.deepEqual(range, { start: '2025-05-01', end: '2025-05-31' })
  assert.match(
    describePeriodCoverage('6m', '2025-05-01', '2025-05-31', range),
    /Showing available coverage only/,
  )
  assert.equal(
    filterHistory(rows, range).length,
    0,
  )
})

test('selected range summary and chart rows derive from the same active history', () => {
  const range = buildPeriodRange('custom', '2024-12-01', '2024-12-30', '2024-12-01', '2024-12-02')
  const visible = filterHistory(rows, range)
  const summary = summarizeHistory(visible, 'port-of-los-angeles', 'Los Angeles', true, false)
  assert.deepEqual(visible.map((row) => row.date), ['2024-12-01', '2024-12-02'])
  assert.equal(summary.observations, visible.length)
  assert.equal(summary.coverage.start, visible[0].date)
  assert.equal(summary.coverage.end, visible.at(-1).date)
  assert.equal(summary.congestion_days, 0)
  assert.equal(summary.congestion_rate, 0)
})

test('empty ranges do not report fabricated event counts', () => {
  const summary = summarizeHistory([], 'port-of-los-angeles', 'Los Angeles', true, false)
  assert.equal(summary.data_available, false)
  assert.equal(summary.observations, 0)
  assert.equal(summary.congestion_days, null)
  assert.equal(summary.congestion_rate, null)
  assert.equal(summary.coverage, null)
})

test('unlabeled history reports unavailable congestion metrics rather than zero events', () => {
  const unlabeledRows = [
    { date: '2025-01-01', vessel_density: 3, anchorage_vessels: null, anchorage_dwell_hours: null, next_day_congestion: null },
  ]
  const summary = summarizeHistory(
    unlabeledRows,
    'uploaded-ais-area',
    'Uploaded AIS dataset — geographic area not verified',
    true,
    false,
  )
  assert.equal(summary.observations, 1)
  assert.equal(summary.target_observations, 0)
  assert.equal(summary.target_available, false)
  assert.equal(summary.congestion_days, null)
  assert.equal(summary.congestion_rate, null)
})

test('range summaries and latest valid observations follow each selected range', () => {
  const variedRows = [
    { date: '2024-01-01', vessel_density: 10, anchorage_vessels: 2, anchorage_dwell_hours: 1, next_day_congestion: 0 },
    { date: '2024-01-02', vessel_density: 20, anchorage_vessels: 0, anchorage_dwell_hours: null, next_day_congestion: 1 },
    { date: '2024-01-03', vessel_density: 30, anchorage_vessels: null, anchorage_dwell_hours: 3, next_day_congestion: null },
    { date: '2024-01-04', vessel_density: 40, anchorage_vessels: 4, anchorage_dwell_hours: 4, next_day_congestion: 1 },
  ]
  const firstRange = filterHistory(variedRows, { start: '2024-01-01', end: '2024-01-02' })
  const secondRange = filterHistory(variedRows, { start: '2024-01-02', end: '2024-01-03' })
  const thirdRange = filterHistory(variedRows, { start: '2024-01-04', end: '2024-01-04' })
  const summarize = (selected) => summarizeHistory(
    selected, 'port-of-los-angeles', 'Los Angeles', true, false,
  )

  assert.deepEqual(
    [firstRange, secondRange, thirdRange].map((selected) => {
      const result = summarize(selected)
      return [result.observations, result.target_observations, result.congestion_days, result.congestion_rate]
    }),
    [[2, 2, 1, 0.5], [2, 1, 1, 1], [1, 1, 1, 1]],
  )
  assert.equal(latestValidObservation(secondRange, 'vessel_density')?.date, '2024-01-03')
  assert.equal(latestValidObservation(secondRange, 'anchorage_vessels')?.date, '2024-01-02')
  assert.equal(latestValidObservation(secondRange, 'anchorage_vessels')?.anchorage_vessels, 0)
  assert.equal(latestValidObservation(secondRange, 'anchorage_dwell_hours')?.date, '2024-01-03')
  assert.equal(latestValidObservation([], 'vessel_density'), undefined)
})

test('evaluation identity is compared with the active dataset when evidence exists', () => {
  assert.equal(
    datasetsMatch(
      { data_status: 'uploaded', sha256: 'new-upload' },
      { data_status: 'synthetic', sha256: 'original-model-data' },
    ),
    false,
  )
  assert.equal(
    datasetsMatch(
      { data_status: 'uploaded', sha256: 'same' },
      { data_status: 'uploaded', sha256: 'same' },
    ),
    true,
  )
  assert.equal(datasetsMatch({ data_status: 'public' }, { data_status: 'public' }), null)
})

test('a superseded dashboard refresh cannot commit over a newer snapshot', () => {
  const gate = createRequestGate()
  const older = gate.next()
  const newer = gate.next()
  assert.equal(gate.isCurrent(older), false)
  assert.equal(gate.isCurrent(newer), true)
  gate.invalidate()
  assert.equal(gate.isCurrent(newer), false)
})

test('an upload refresh accepts its new snapshot and rejects the previous dataset', () => {
  const staleSnapshot = { active_dataset: { sha256: 'original-data' } }
  const uploadedSnapshot = {
    active_dataset: { sha256: 'uploaded-data' },
    history: [
      { date: '2025-05-02', vessel_density: 40, anchorage_vessels: 31, anchorage_dwell_hours: 12, next_day_congestion: 1 },
    ],
  }
  assert.equal(dashboardMatchesDataset(staleSnapshot, 'uploaded-data'), false)
  assert.equal(dashboardMatchesDataset(uploadedSnapshot, 'uploaded-data'), true)
  assert.equal(dashboardMatchesDataset(uploadedSnapshot, ''), false)
  const summary = summarizeHistory(
    uploadedSnapshot.history,
    'port-of-los-angeles',
    'Los Angeles',
    true,
    false,
  )
  assert.equal(uploadedSnapshot.history[0].vessel_density, 40)
  assert.equal(uploadedSnapshot.history[0].anchorage_vessels, 31)
  assert.equal(summary.observations, 1)
  assert.equal(summary.congestion_days, 1)
})
