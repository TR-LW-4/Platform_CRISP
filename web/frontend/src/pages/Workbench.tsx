import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { defaultsFor } from '../config'
import { InstanceSourcePanel } from '../components/InstanceSourcePanel'
import { MetricChart } from '../components/MetricChart'
import { ParameterForm } from '../components/ParameterForm'
import type {
  AlgorithmFidelity,
  AlgorithmGeometry,
  AlgorithmInfo,
  AlgorithmRestriction,
  BatchClassSummary,
  BenchmarkQueueBlock,
  Catalog,
  ConfigValues,
  JobStatus,
  JobSummary,
  ProgressRecord,
} from '../types'

interface WorkbenchProps {
  catalog: Catalog
  resumeJobId?: string | null
  onResumeHandled?: () => void
}

const terminalStatuses: JobStatus[] = ['completed', 'stopped', 'failed']

const LAYOUT_HIDDEN_KEYS = [
  'num_bays',
  'num_rows',
  'max_tiers',
  'num_containers',
  'num_groups',
  'seed',
]

function compatible(algorithm: AlgorithmInfo, problemName: string) {
  return (
    algorithm.compatible_problems.length === 0 ||
    algorithm.compatible_problems.includes(problemName)
  )
}

type KindFilter = 'all' | 'exact' | 'heuristic' | 'solver'
type RestrictionFilter = 'all' | 'restricted' | 'unrestricted'

const ALL_FAMILIES = 'All'

const KIND_BUTTONS: { id: KindFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'exact', label: 'Exact' },
  { id: 'heuristic', label: 'Heuristic' },
  { id: 'solver', label: 'Solver / IP' },
]

const RESTRICTION_BUTTONS: { id: RestrictionFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'restricted', label: 'Restricted' },
  { id: 'unrestricted', label: 'Unrestricted' },
]

const RESTRICTION_LABELS: Record<AlgorithmRestriction, string> = {
  restricted: 'Restricted relocation',
  unrestricted: 'Unrestricted relocation',
  both: 'Restricted + unrestricted',
}

/**
 * Kind is derived from fields the catalog already has. "Exact" follows the
 * class `category` (not method_group, which files search-based exact methods
 * under "Tree Search"); "Solver / IP" follows method_group or requires_solver.
 * An exact Gurobi model therefore appears under both Exact and Solver / IP.
 */
function matchesKind(algorithm: AlgorithmInfo, kind: KindFilter) {
  const category = algorithm.category.toLowerCase()
  switch (kind) {
    case 'exact':
      return category === 'exact'
    case 'heuristic':
      return category === 'heuristic' || algorithm.method_group === 'Heuristic'
    case 'solver':
      return algorithm.method_group === 'Solver / IP' || algorithm.requires_solver
    default:
      return true
  }
}

function matchesRestriction(algorithm: AlgorithmInfo, filter: RestrictionFilter) {
  if (filter === 'all') return true
  const declared = algorithm.restriction
  return declared === filter || declared === 'both'
}

const GEOMETRY_LABELS: Record<AlgorithmGeometry, string> = {
  'single-bay': 'Single-bay (2D)',
  'multi-bay': 'Multi-bay (3D)',
}

const FIDELITY_LABELS: Record<AlgorithmFidelity, string> = {
  faithful: 'Faithful reproduction',
  adapted: 'Platform adaptation',
  degenerate: 'Degenerate baseline',
}

// Objective modes under which crane time is part of what is minimised.
const TIME_AWARE_MODES = new Set(['crane_time', 'weighted'])

interface Assessment {
  status: 'ok' | 'degraded' | 'incompatible'
  notes: string[]
}

function objectiveLabel(algorithm: AlgorithmInfo) {
  if (algorithm.objectives.includes('fixed-rule')) return 'Fixed rule'
  return algorithm.objectives
    .map((item) => (item === 'crane_time' ? 'Crane time' : 'Relocations'))
    .join(' → ')
}

/**
 * Mirror of core.algorithm_meta.assess_compatibility. A yard only counts as
 * multi-bay when both horizontal axes exceed one: the single-bay benchmarks
 * store their S stacks as S bays with one row each.
 */
function assess(
  algorithm: AlgorithmInfo,
  problemName: string,
  objectiveMode: string,
  numBays: number,
  numRows: number,
  restrictedRelocation?: boolean,
): Assessment {
  if (!compatible(algorithm, problemName)) {
    return {
      status: 'incompatible',
      notes: [`Not registered for ${problemName}.`],
    }
  }

  const notes: string[] = []
  let status: Assessment['status'] = 'ok'
  const timeAware = TIME_AWARE_MODES.has(objectiveMode)
  const fixedRule = algorithm.objectives.includes('fixed-rule')
  const optimises =
    !fixedRule &&
    algorithm.objectives.includes(timeAware ? 'crane_time' : 'relocations')

  if (!optimises) {
    status = 'degraded'
    if (fixedRule) {
      notes.push(
        'Deterministic rule: the selected objective is measured afterwards, ' +
          'never optimised. Use as a baseline only.',
      )
    } else if (timeAware) {
      notes.push(
        'Minimises relocation count only; crane time is reported but not optimised.',
      )
    } else {
      notes.push(
        'Does not minimise relocation count; the reported value is a ' +
          'by-product of a different objective.',
      )
    }
  }

  if (timeAware && numBays > 1 && numRows > 1 && algorithm.geometry === 'single-bay') {
    status = 'degraded'
    notes.push(
      `Single-bay method on a ${numBays}×${numRows} yard: stacks are flattened, ` +
        'so gantry travel between bays is not part of its decisions.',
    )
  }

  if (
    restrictedRelocation !== undefined &&
    algorithm.restriction &&
    algorithm.restriction !== 'both' &&
    (algorithm.restriction === 'restricted') !== restrictedRelocation
  ) {
    status = 'degraded'
    notes.push(
      algorithm.restriction === 'restricted'
        ? 'This algorithm solves the restricted problem but the instance is set to unrestricted relocation; compare only against restricted references.'
        : 'This algorithm solves the unrestricted problem but the instance is set to restricted relocation; compare only against unrestricted references.',
    )
  }

  if (algorithm.fidelity === 'adapted') {
    notes.push(
      'Platform adaptation rather than a line-by-line reproduction; ' +
        'published numbers are not expected to match.',
    )
  } else if (algorithm.fidelity === 'degenerate') {
    status = 'degraded'
    notes.push('Degenerate baseline: does not represent the published method.')
  }

  return { status, notes }
}

function displayNumber(value: number | null | undefined) {
  return value === null || value === undefined ? '—' : value.toFixed(3)
}

const HIDDEN_SUMMARY_METRICS = new Set([
  'retrievals',
  'total_moves',
  'steps',
  'progress',
  'optimal_proven',
])

function summaryMetricKeys(classes: BatchClassSummary[]): string[] {
  const keys = new Set<string>()
  for (const row of classes) {
    Object.keys(row.metrics || {}).forEach((key) => {
      if (!HIDDEN_SUMMARY_METRICS.has(key)) keys.add(key)
    })
  }
  // Prefer literature-facing metrics first when present.
  const preferred = [
    'objective_value',
    'relocations',
    'crane_time',
    'crane_time_f2',
    'crane_time_vertical',
    'crane_time_rmgc',
    'time',
    'moves',
    'shifters',
  ]
  const ordered = preferred.filter((key) => keys.has(key))
  for (const key of [...keys].sort()) {
    if (!ordered.includes(key)) ordered.push(key)
  }
  return ordered
}

export function Workbench({
  catalog,
  resumeJobId,
  onResumeHandled,
}: WorkbenchProps) {
  const [problemName, setProblemName] = useState(catalog.problems[0]?.name ?? '')
  const [methodGroup, setMethodGroup] = useState(ALL_FAMILIES)
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [restrictionFilter, setRestrictionFilter] = useState<RestrictionFilter>('all')
  const [algorithmName, setAlgorithmName] = useState('')
  const [problemValues, setProblemValues] = useState<ConfigValues>({})
  const [algorithmValues, setAlgorithmValues] = useState<ConfigValues>({})
  const [instanceSource, setInstanceSource] = useState('random')
  const [benchmarkQueue, setBenchmarkQueue] = useState<BenchmarkQueueBlock[]>([])
  const [zhuDupAlpha, setZhuDupAlpha] = useState('alpha=0.2')
  const [firstOnly, setFirstOnly] = useState(false)
  const [resolvedCount, setResolvedCount] = useState(0)
  const [job, setJob] = useState<JobSummary | null>(null)
  const [records, setRecords] = useState<ProgressRecord[]>([])
  const [error, setError] = useState('')
  const sequence = useRef(0)
  const preserveResumedConfig = useRef(false)
  const handleResolvedCount = useCallback((count: number) => {
    setResolvedCount(count)
  }, [])

  const problem = catalog.problems.find((item) => item.name === problemName)
  const compatibleAlgorithms = useMemo(
    () => catalog.algorithms.filter((item) => compatible(item, problemName)),
    [catalog.algorithms, problemName],
  )
  // Restriction buttons only make sense when some algorithm declares a rule.
  const hasRestrictionInfo = useMemo(
    () => compatibleAlgorithms.some((item) => item.restriction),
    [compatibleAlgorithms],
  )
  const filteredAlgorithms = useMemo(
    () =>
      compatibleAlgorithms.filter(
        (item) =>
          matchesKind(item, kindFilter) &&
          (!hasRestrictionInfo || matchesRestriction(item, restrictionFilter)),
      ),
    [compatibleAlgorithms, kindFilter, restrictionFilter, hasRestrictionInfo],
  )
  const methodGroups = useMemo(
    () => [...new Set(filteredAlgorithms.map((item) => item.method_group))].sort(),
    [filteredAlgorithms],
  )
  const groupAlgorithms = useMemo(
    () =>
      methodGroup === ALL_FAMILIES
        ? filteredAlgorithms
        : filteredAlgorithms.filter((item) => item.method_group === methodGroup),
    [filteredAlgorithms, methodGroup],
  )
  const undeclaredRestriction = useMemo(
    () => compatibleAlgorithms.filter((item) => !item.restriction).length,
    [compatibleAlgorithms],
  )
  const algorithm = catalog.algorithms.find((item) => item.name === algorithmName)
  const running = Boolean(job && !terminalStatuses.includes(job.status))
  const jobId = job?.id
  const jobStatus = job?.status
  const usingLayout = instanceSource !== 'random'

  // Comparison axes the fairness check depends on. Benchmark layouts always
  // land on a single row, so their geometry is fixed regardless of the form.
  const objectiveMode = String(problemValues.objective_mode ?? 'relocations')
  const numBays = Number(problemValues.num_bays ?? 1)
  const numRows = usingLayout ? 1 : Number(problemValues.num_rows ?? 1)
  const restrictedRelocation =
    typeof problemValues.restricted_relocation === 'boolean'
      ? problemValues.restricted_relocation
      : undefined
  const assessment = useMemo(
    () =>
      algorithm
        ? assess(
            algorithm,
            problemName,
            objectiveMode,
            numBays,
            numRows,
            restrictedRelocation,
          )
        : null,
    [algorithm, problemName, objectiveMode, numBays, numRows, restrictedRelocation],
  )
  const batchMetricKeys = useMemo(
    () => summaryMetricKeys(job?.batch_summary?.classes ?? []),
    [job?.batch_summary],
  )

  useEffect(() => {
    if (!resumeJobId) return
    let cancelled = false
    const resume = async () => {
      try {
        const resumed = await api.job(resumeJobId)
        if (cancelled) return
        preserveResumedConfig.current = true
        setProblemName(resumed.problem_name)
        setAlgorithmName(resumed.algorithm_name)
        const resumedAlgorithm = catalog.algorithms.find(
          (item) => item.name === resumed.algorithm_name,
        )
        setKindFilter('all')
        setRestrictionFilter('all')
        setMethodGroup(resumedAlgorithm?.method_group ?? ALL_FAMILIES)
        setProblemValues(resumed.problem_config)
        setAlgorithmValues(resumed.algorithm_config)
        setInstanceSource(resumed.instance_source ?? 'random')
        setRecords(resumed.records ?? [])
        sequence.current = resumed.records?.at(-1)?.sequence ?? 0
        setJob(resumed)
        setError(resumed.error ?? '')
      } catch (resumeError) {
        if (!cancelled) {
          setError(
            resumeError instanceof Error ? resumeError.message : String(resumeError),
          )
        }
      } finally {
        if (!cancelled) onResumeHandled?.()
      }
    }
    void resume()
    return () => {
      cancelled = true
    }
  }, [catalog.algorithms, resumeJobId, onResumeHandled])

  useEffect(() => {
    if (methodGroup !== ALL_FAMILIES && !methodGroups.includes(methodGroup)) {
      setMethodGroup(ALL_FAMILIES)
    }
  }, [methodGroups, methodGroup])

  useEffect(() => {
    if (!groupAlgorithms.some((item) => item.name === algorithmName)) {
      setAlgorithmName(groupAlgorithms[0]?.name ?? '')
    }
  }, [groupAlgorithms, algorithmName])

  useEffect(() => {
    if (problem) {
      if (preserveResumedConfig.current) return
      setProblemValues(defaultsFor(problem.config_schema))
      setBenchmarkQueue([])
      setFirstOnly(false)
    }
  }, [problem])

  // Picking an algorithm that solves one relocation rule sets the matching
  // problem flag. Declared after the problem-defaults effect so it wins when
  // both fire together; skipped while a resumed job restores its own config.
  useEffect(() => {
    if (!algorithm || !problem || preserveResumedConfig.current) return
    if (!('restricted_relocation' in problem.config_schema)) return
    if (algorithm.restriction === 'restricted') {
      setProblemValues((current) => ({ ...current, restricted_relocation: true }))
    } else if (algorithm.restriction === 'unrestricted') {
      setProblemValues((current) => ({ ...current, restricted_relocation: false }))
    }
  }, [algorithm, problem])

  useEffect(() => {
    if (algorithm) {
      if (preserveResumedConfig.current) {
        preserveResumedConfig.current = false
        return
      }
      setAlgorithmValues(defaultsFor(algorithm.config_schema))
    }
  }, [algorithm])

  useEffect(() => {
    if (!jobId || !jobStatus || terminalStatuses.includes(jobStatus)) return

    let cancelled = false
    const poll = async () => {
      try {
        const response = await api.records(jobId, sequence.current)
        if (cancelled) return
        if (response.records.length) {
          setRecords((current) => [...current, ...response.records])
          sequence.current = response.next_sequence
        }
        setJob((current) =>
          current
            ? {
                ...current,
                status: response.status,
                error: response.error,
                result_file: response.result_file,
                result_files: response.result_files,
                batch_summary: response.batch_summary ?? current.batch_summary,
                batch_total: response.batch_total,
                batch_index: response.batch_index,
                completed_count: response.completed_count,
                pending_count: response.pending_count,
                instance_source: response.instance_source,
              }
            : current,
        )
        if (response.error) setError(response.error)
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : String(pollError))
        }
      }
    }
    void poll()
    const timer = window.setInterval(poll, 500)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [jobId, jobStatus])

  // Final refresh so batch_summary is present even if the last poll raced completion.
  useEffect(() => {
    if (!jobId || !jobStatus || !terminalStatuses.includes(jobStatus)) return
    if (job?.batch_summary?.classes?.length) return
    let cancelled = false
    const refresh = async () => {
      try {
        const latestJob = await api.job(jobId)
        if (cancelled || !latestJob.batch_summary) return
        setJob((current) =>
          current ? { ...current, batch_summary: latestJob.batch_summary } : current,
        )
      } catch {
        // Ignore refresh errors; per-instance results are already visible.
      }
    }
    void refresh()
    return () => {
      cancelled = true
    }
  }, [jobId, jobStatus, job?.batch_summary?.classes?.length])

  const start = async () => {
    if (!problem || !algorithm) return
    if (
      problem.name === 'CRP-Time' &&
      problemValues.objective_mode === 'weighted' &&
      Number(problemValues.relocation_weight ?? 0) === 0 &&
      Number(problemValues.time_weight ?? 0) === 0
    ) {
      setError('Relocation weight and time weight cannot both be zero.')
      return
    }
    if (usingLayout && resolvedCount <= 0) {
      setError('Add at least one instance block before starting a benchmark run.')
      return
    }
    setError('')
    setRecords([])
    sequence.current = 0
    try {
      setJob(
        await api.startJob({
          problemName: problem.name,
          algorithmName: algorithm.name,
          problemConfig: problemValues,
          algorithmConfig: algorithmValues,
          instanceSource,
          benchmarkQueue,
          zhuDupAlpha: instanceSource === 'zhu_dup' ? zhuDupAlpha : undefined,
          firstOnly,
        }),
      )
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : String(startError))
    }
  }

  const stop = async () => {
    if (!job) return
    try {
      setJob(await api.stopJob(job.id))
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : String(stopError))
    }
  }

  const continueRun = async () => {
    if (!job) return
    setError('')
    try {
      setJob(await api.continueJob(job.id))
    } catch (continueError) {
      setError(
        continueError instanceof Error ? continueError.message : String(continueError),
      )
    }
  }

  const latest = records.at(-1) ?? job?.latest
  const primaryMetric = problem?.metric_names[0] ?? 'Metric'
  const hasBatchSummary = Boolean(job?.batch_summary?.classes?.length)
  const showBatchSummarySlot = usingLayout || hasBatchSummary
  const layoutLabel = latest?.extra?.layout_file
    ? String(latest.extra.layout_file).split('/').slice(-2).join('/')
    : null
  const batchLabel =
    job?.batch_total && job.batch_total > 0
      ? `${(job.batch_index ?? 0) + 1} / ${job.batch_total}`
      : null
  const canContinue = Boolean(
    job
    && (job.status === 'stopped' || job.status === 'failed')
    && (job.batch_total ?? 0) > 0
    && (job.pending_count ?? 0) > 0,
  )

  return (
    <main className="page workbench">
      <header className="page-header">
        <div>
          <span className="eyebrow">Research execution</span>
          <h1>Test workbench</h1>
          <p>Configure once. Progress updates without rebuilding the page.</p>
        </div>
        <div className={`status-badge status-${job?.status ?? 'idle'}`}>
          {job?.status ?? 'idle'}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <strong>Run error</strong>
          <pre>{error}</pre>
        </div>
      )}

      <div className="workbench-grid">
        <aside className="config-panel card">
          <label className="field">
            <span>Problem</span>
            <select
              value={problemName}
              disabled={running}
              onChange={(event) => {
                setProblemName(event.target.value)
                setKindFilter('all')
                setRestrictionFilter('all')
                setMethodGroup(ALL_FAMILIES)
              }}
            >
              {catalog.problems.map((item) => (
                <option key={item.name}>{item.name}</option>
              ))}
            </select>
          </label>
          {problem && (
            <>
              <p className="description">{problem.description}</p>
              <div className="tags">
                {problem.tags.map((tag) => <span key={tag}>{tag}</span>)}
              </div>
            </>
          )}

          <InstanceSourcePanel
            problemName={problemName}
            disabled={running}
            source={instanceSource}
            queue={benchmarkQueue}
            alpha={zhuDupAlpha}
            firstOnly={firstOnly}
            onSourceChange={setInstanceSource}
            onQueueChange={setBenchmarkQueue}
            onAlphaChange={setZhuDupAlpha}
            onFirstOnlyChange={setFirstOnly}
            onResolvedCount={handleResolvedCount}
          />

          {hasRestrictionInfo && (
            <div className="field">
              <span>Relocation rule</span>
              <div className="segmented" role="group" aria-label="Relocation rule">
                {RESTRICTION_BUTTONS.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className={restrictionFilter === item.id ? 'active' : ''}
                    aria-pressed={restrictionFilter === item.id}
                    disabled={running}
                    onClick={() => setRestrictionFilter(item.id)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
              {restrictionFilter !== 'all' && undeclaredRestriction > 0 && (
                <small className="description">
                  {undeclaredRestriction} algorithm(s) do not declare a relocation
                  rule and are hidden; choose All to see them.
                </small>
              )}
            </div>
          )}
          <label className="field">
            <span>Algorithm family</span>
            <select
              value={methodGroup}
              disabled={running}
              onChange={(event) => setMethodGroup(event.target.value)}
            >
              <option value={ALL_FAMILIES}>{ALL_FAMILIES}</option>
              {methodGroups.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label className="field">
            <span>Algorithm ({groupAlgorithms.length})</span>
            <select
              value={algorithmName}
              disabled={running || !groupAlgorithms.length}
              onChange={(event) => setAlgorithmName(event.target.value)}
            >
              {!groupAlgorithms.length && (
                <option value="">No algorithm matches these filters</option>
              )}
              {(methodGroup === ALL_FAMILIES ? methodGroups : [methodGroup]).map((group) => {
                const members = groupAlgorithms.filter((item) => item.method_group === group)
                if (!members.length) return null
                return (
                  <optgroup key={group} label={group}>
                    {members.map((item) => {
                      const itemStatus = assess(
                        item,
                        problemName,
                        objectiveMode,
                        numBays,
                        numRows,
                        restrictedRelocation,
                      ).status
                      return (
                        <option key={item.name} value={item.name}>
                          {itemStatus === 'degraded' ? `⚠ ${item.name}` : item.name}
                        </option>
                      )
                    })}
                  </optgroup>
                )
              })}
            </select>
          </label>
          {algorithm && (
            <>
              <div className="tags">
                <span>{algorithm.method_group}</span>
                {algorithm.category !== algorithm.method_group && (
                  <span>{algorithm.category}</span>
                )}
                {algorithm.restriction && (
                  <span>{RESTRICTION_LABELS[algorithm.restriction]}</span>
                )}
                <span>{GEOMETRY_LABELS[algorithm.geometry]}</span>
                <span>{objectiveLabel(algorithm)}</span>
                {algorithm.fidelity !== 'faithful' && (
                  <span>{FIDELITY_LABELS[algorithm.fidelity]}</span>
                )}
              </div>
              {assessment && assessment.notes.length > 0 && (
                <ul className={`comparison-notes ${assessment.status}`}>
                  {assessment.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              )}
              <p className="description">
                {algorithm.requires_solver && (
                  <strong>Requires {algorithm.solver_backend ?? 'external solver'}. </strong>
                )}
                {algorithm.description}
              </p>
            </>
          )}

          <details open>
            <summary>
              Problem parameters
              {usingLayout ? ' (layout overrides geometry)' : ''}
            </summary>
            {problem && (
              <ParameterForm
                schema={problem.config_schema}
                values={problemValues}
                disabled={running}
                hideKeys={usingLayout ? LAYOUT_HIDDEN_KEYS : []}
                onChange={setProblemValues}
              />
            )}
          </details>
          <details open>
            <summary>Algorithm parameters</summary>
            {algorithm && (
              <ParameterForm
                schema={algorithm.config_schema}
                values={algorithmValues}
                disabled={running}
                onChange={setAlgorithmValues}
              />
            )}
          </details>

          <div className="run-actions">
            <button
              className="primary"
              disabled={
                running
                || !algorithm
                || (usingLayout && resolvedCount <= 0)
              }
              onClick={start}
            >
              {usingLayout
                ? `Start batch${resolvedCount ? ` (${resolvedCount})` : ''}`
                : 'Start run'}
            </button>
            <button disabled={!running || job?.status === 'stopping'} onClick={stop}>
              Stop
            </button>
            {canContinue && (
              <button className="primary" onClick={() => void continueRun()}>
                Continue
                {job?.pending_count
                  ? ` (${job.pending_count} left)`
                  : ''}
              </button>
            )}
          </div>
        </aside>

        <section className="run-panel">
          <div className="run-panel-sticky">
            <div className="metric-cards">
              <article className="metric-card">
                <span>{algorithm?.step_label ?? 'Step'}</span>
                <strong>{latest?.step ?? '—'}</strong>
              </article>
              <article className="metric-card">
                <span>Current metric</span>
                <strong>{displayNumber(latest?.metric)}</strong>
              </article>
              <article className="metric-card">
                <span>Best metric</span>
                <strong>{displayNumber(latest?.best_metric)}</strong>
              </article>
              <article className="metric-card">
                <span>{batchLabel ? 'Batch' : 'Updates'}</span>
                <strong>{batchLabel ?? records.length}</strong>
              </article>
            </div>

            {layoutLabel && (
              <div className="layout-banner">Current layout: {layoutLabel}</div>
            )}

            <div className="progress-track">
              <div style={{ width: `${(latest?.progress ?? 0) * 100}%` }} />
            </div>
          </div>

          <div className="run-panel-body">
            {showBatchSummarySlot && (
              <article className="visual-card card batch-summary-card">
                <div className="card-heading">
                  <div>
                    <span className="eyebrow">Benchmark summary</span>
                    <h2>Mean ± std by class</h2>
                  </div>
                  <span>
                    {hasBatchSummary
                      ? `${job?.batch_summary?.total_runs ?? 0} runs`
                      : running
                        ? 'Running…'
                        : 'Waiting for batch'}
                    {job?.batch_summary?.ungrouped
                      ? ` · ${job.batch_summary.ungrouped} ungrouped`
                      : ''}
                  </span>
                </div>
                {hasBatchSummary ? (
                  <div className="table-card batch-summary-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Class</th>
                          <th>n</th>
                          {batchMetricKeys.flatMap((key) => [
                            <th key={`${key}-mean`}>{key} mean</th>,
                            <th key={`${key}-std`}>{key} std</th>,
                          ])}
                        </tr>
                      </thead>
                      <tbody>
                        {job!.batch_summary!.classes.map((row) => (
                          <tr key={`${row.source}-${row.label}`}>
                            <td><code>{row.label}</code></td>
                            <td>{row.n}</td>
                            {batchMetricKeys.flatMap((key) => {
                              const agg = row.metrics?.[key]
                              return [
                                <td key={`${row.label}-${key}-mean`}>
                                  {agg ? displayNumber(agg.mean) : '—'}
                                </td>,
                                <td key={`${row.label}-${key}-std`}>
                                  {agg ? displayNumber(agg.std) : '—'}
                                </td>,
                              ]
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="description">
                    Class-wise mean/std appears here when a Caserta/Zhu batch finishes
                    (2+ layout files). Single-instance runs skip this table.
                  </p>
                )}
                {job?.batch_summary?.error && (
                  <div className="error-banner">{job.batch_summary.error}</div>
                )}
              </article>
            )}

            <article className="visual-card card convergence-card">
              <div className="card-heading">
                <div>
                  <span className="eyebrow">Convergence</span>
                  <h2>{primaryMetric}</h2>
                </div>
              </div>
              <div className="convergence-scroll">
                <MetricChart records={records} label={primaryMetric} />
              </div>
            </article>

            {job?.result_file && (
              <div className="success-banner">
                Saved {job.result_files?.length && job.result_files.length > 1
                  ? `${job.result_files.length} results (latest: ${job.result_file})`
                  : `to ${job.result_file}`}
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  )
}