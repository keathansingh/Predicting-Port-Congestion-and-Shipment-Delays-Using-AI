export function buildPeriodRange(period, coverageStart, coverageEnd, customStart, customEnd) {
  if (!coverageStart || !coverageEnd) return { start: '', end: '' }

  if (period === 'custom') {
    const start = customStart || coverageStart
    const end = customEnd || coverageEnd
    return {
      start: start < coverageStart ? coverageStart : start,
      end: end > coverageEnd ? coverageEnd : end,
    }
  }

  const months = { '1m': 1, '6m': 6, '1y': 12, '2y': 24, '3y': 36 }
  const selectedStart = subtractMonths(coverageEnd, months[period])
  return { start: selectedStart < coverageStart ? coverageStart : selectedStart, end: coverageEnd }
}

function subtractMonths(dateString, count) {
  const end = new Date(`${dateString}T00:00:00Z`)
  const day = end.getUTCDate()
  end.setUTCDate(1)
  end.setUTCMonth(end.getUTCMonth() - count)
  const lastDay = new Date(Date.UTC(end.getUTCFullYear(), end.getUTCMonth() + 1, 0)).getUTCDate()
  end.setUTCDate(Math.min(day, lastDay))
  return end.toISOString().slice(0, 10)
}

export function describePeriodCoverage(period, coverageStart, coverageEnd, range) {
  if (!range.start || !range.end) return 'Date coverage unavailable'
  if (period === 'custom') return `Showing available observations · ${range.start} — ${range.end} UTC`
  const requestedStart = subtractMonths(coverageEnd, { '1m': 1, '6m': 6, '1y': 12, '2y': 24, '3y': 36 }[period])
  if (range.start > requestedStart || coverageStart > requestedStart) {
    return `Showing available coverage only · ${range.start} — ${range.end} UTC · no dates are interpolated`
  }
  return `${range.start} — ${range.end} UTC · observed dates only`
}

export function filterHistory(rows, range) {
  if (!range.start || !range.end || range.start > range.end) return []
  return rows.filter((row) => row.date >= range.start && row.date <= range.end)
}

export function sampleHistory(rows, maximum = 30) {
  if (maximum <= 0 || rows.length === 0) return []
  if (rows.length <= maximum) return rows
  if (maximum === 1) return [rows[0]]
  return Array.from({ length: maximum }, (_, index) => (
    rows[Math.round(index * (rows.length - 1) / (maximum - 1))]
  ))
}

export function latestValidObservation(rows, field) {
  return rows.reduce((latest, row) => {
    const value = row[field]
    if (value == null || !Number.isFinite(Number(value))) return latest
    return !latest || row.date > latest.date ? row : latest
  }, undefined)
}

export function summarizeHistory(rows, portId, portName, datasetAvailable, synthetic) {
  const labels = rows
    .map((row) => row.next_day_congestion)
    .filter((label) => label === 0 || label === 1)
  const congestionDays = labels.reduce((total, label) => total + label, 0)
  return {
    data_available: rows.length > 0,
    dataset_available: datasetAvailable,
    port_id: portId,
    port_name: portName,
    observations: rows.length,
    target_observations: labels.length,
    target_available: labels.length > 0,
    congestion_days: labels.length ? congestionDays : null,
    congestion_rate: labels.length ? congestionDays / labels.length : null,
    coverage: rows.length ? { start: rows[0].date, end: rows.at(-1).date } : null,
    synthetic,
  }
}

export function datasetsMatch(activeDataset, evaluationDataset) {
  if (!activeDataset || !evaluationDataset) return null
  if (activeDataset.sha256 && evaluationDataset.sha256) {
    return activeDataset.sha256 === evaluationDataset.sha256
  }
  if (
    activeDataset.data_status &&
    evaluationDataset.data_status &&
    activeDataset.data_status !== evaluationDataset.data_status
  ) {
    return false
  }
  return null
}

export function dashboardMatchesDataset(snapshot, expectedSha256) {
  return Boolean(expectedSha256 && snapshot?.active_dataset?.sha256 === expectedSha256)
}

export function createRequestGate() {
  let revision = 0
  return {
    next() {
      revision += 1
      return revision
    },
    isCurrent(token) {
      return token === revision
    },
    invalidate() {
      revision += 1
    },
  }
}
