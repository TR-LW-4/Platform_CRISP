import type { ProgressRecord } from '../types'

interface MetricChartProps {
  records: ProgressRecord[]
  label: string
}

export function MetricChart({ records, label }: MetricChartProps) {
  const values = records
    .map((record, index) => ({ index, value: record.metric }))
    .filter((item): item is { index: number; value: number } => item.value !== null)

  if (!values.length) {
    return <div className="empty-visual">Waiting for progress data</div>
  }

  const width = 800
  const height = 340
  const pad = 36
  const minimum = Math.min(...values.map((item) => item.value))
  const maximum = Math.max(...values.map((item) => item.value))
  const range = maximum - minimum || 1
  const denominator = Math.max(records.length - 1, 1)
  const points = values
    .map(({ index, value }) => {
      const x = pad + (index / denominator) * (width - pad * 2)
      const y = height - pad - ((value - minimum) / range) * (height - pad * 2)
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} chart`}>
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="axis" />
        <line
          x1={pad}
          y1={height - pad}
          x2={width - pad}
          y2={height - pad}
          className="axis"
        />
        <line
          x1={pad}
          y1={height / 2}
          x2={width - pad}
          y2={height / 2}
          className="grid-line"
        />
        <polyline points={points} className="metric-line" />
        <text x={pad} y={20} className="chart-label">
          {maximum.toFixed(3)}
        </text>
        <text x={pad} y={height - 8} className="chart-label">
          {minimum.toFixed(3)}
        </text>
      </svg>
      <div className="chart-caption">
        <span>{label}</span>
        <strong>{values.at(-1)?.value.toFixed(3)}</strong>
      </div>
    </div>
  )
}
