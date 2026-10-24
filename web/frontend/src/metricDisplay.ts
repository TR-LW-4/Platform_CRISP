import type { ProblemInfo } from './types'

interface ClassRow {
  label: string
  display_label?: string
  objective_group?: string
}

// Metric of the selected crane-time model (CRP-Time `time_model` option).
const TIME_MODEL_METRIC: Record<string, string> = {
  f2: 'crane_time_f2',
  f2_vertical: 'crane_time_vertical',
  f2_vertical_practice: 'crane_time_vertical',
}

export function displayMetricsOf(problem: ProblemInfo | undefined): string[] {
  return problem?.display_metrics ?? []
}

/** Display metrics with the optimised crane-time metric of the time model first. */
export function displayMetricsFor(problem: ProblemInfo | undefined, timeModel: unknown): string[] {
  const metrics = displayMetricsOf(problem)
  const main = TIME_MODEL_METRIC[String(timeModel ?? '')]
  return main && metrics.includes(main) ? [main, ...metrics.filter((key) => key !== main)] : metrics
}

export function metricLabel(problem: ProblemInfo | undefined, key: string): string {
  return problem?.metric_labels?.[key] ?? key
}

/** Label of the optimised crane-time metric, or null when the problem has no display labels. */
export function objectiveMetricLabel(
  problem: ProblemInfo | undefined,
  timeModel: unknown,
): string | null {
  const key = TIME_MODEL_METRIC[String(timeModel ?? '')]
  return key && problem?.metric_labels?.[key] ? problem.metric_labels[key] : null
}

/**
 * Readable class label per grouping label ("3×5 · f2"). The objective hash is
 * appended only where two groups in the same table would otherwise read alike.
 */
export function classLabels(rows: ClassRow[]): Map<string, string> {
  const counts = new Map<string, number>()
  for (const row of rows) {
    const text = row.display_label ?? row.label
    counts.set(text, (counts.get(text) ?? 0) + 1)
  }
  return new Map(
    rows.map((row) => {
      const text = row.display_label ?? row.label
      const hash = row.objective_group?.split('#')[1]
      const clash = (counts.get(text) ?? 0) > 1
      return [row.label, clash && hash ? `${text} #${hash}` : text]
    }),
  )
}
