import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { Catalog, CompareFolder, CompareTable } from '../types'

interface CompareProps {
  catalog: Catalog
  /** When true (page visible), refresh folder list from disk. */
  active?: boolean
}

function fmt(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

function toCsv(table: CompareTable): string {
  const header = ['Class', ...table.algorithms.flatMap((a) => [
    `${a} mean`,
    `${a} std`,
    `${a} n`,
  ])]
  const lines = [header.join(',')]
  for (const row of table.rows) {
    const cells = [row.label]
    for (const algo of table.algorithms) {
      const c = row.cells[algo]
      cells.push(
        c?.mean == null ? '' : String(c.mean),
        c?.std == null ? '' : String(c.std),
        c ? String(c.n) : '0',
      )
    }
    lines.push(cells.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','))
  }
  return lines.join('\n')
}

export function Compare({ catalog, active = true }: CompareProps) {
  const problems = catalog.problems.map((p) => p.name)
  const [problem, setProblem] = useState(problems[0] ?? 'CRP-R')
  const [folders, setFolders] = useState<CompareFolder[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [metric, setMetric] = useState('relocations')
  const [dedup, setDedup] = useState(true)
  const [table, setTable] = useState<CompareTable | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const loadFolders = async (prob: string) => {
    setError('')
    try {
      const list = await api.compareFolders(prob)
      setFolders(list)
      setSelected((prev) => prev.filter((name) => list.some((f) => f.algorithm === name)))
    } catch (err) {
      setFolders([])
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  // Keep-alive page: refresh folders when becoming visible or problem changes.
  useEffect(() => {
    if (!active) return
    void loadFolders(problem)
  }, [active, problem])

  useEffect(() => {
    if (table?.available_metrics?.length && !table.available_metrics.includes(metric)) {
      setMetric(table.available_metrics[0])
    }
  }, [table, metric])

  const toggle = (algorithm: string) => {
    setSelected((prev) =>
      prev.includes(algorithm)
        ? prev.filter((a) => a !== algorithm)
        : [...prev, algorithm],
    )
  }

  const selectTop = (n: number) => {
    setSelected(folders.slice(0, n).map((f) => f.algorithm))
  }

  const runCompare = async () => {
    if (!selected.length) {
      setError('Select at least one algorithm folder.')
      return
    }
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const result = await api.compare({
        problem,
        algorithms: selected,
        metric,
        dedup,
      })
      setTable(result)
      if (result.metric !== metric) setMetric(result.metric)
    } catch (err) {
      setTable(null)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  const deleteFolder = async (algorithm: string, nRuns: number) => {
    const ok = window.confirm(
      `Delete all ${nRuns} saved run file(s) for\n"${algorithm}"\nunder ${problem}?\n\n`
      + 'This removes JSON under results/ and cannot be undone.',
    )
    if (!ok) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const res = await api.deleteResultFolder(problem, algorithm)
      setNotice(`Deleted ${res.removed} file(s) for ${algorithm}.`)
      setSelected((prev) => prev.filter((a) => a !== algorithm))
      if (table?.algorithms.includes(algorithm)) setTable(null)
      await loadFolders(problem)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const deleteSelectedFolders = async () => {
    if (!selected.length) return
    const total = folders
      .filter((f) => selected.includes(f.algorithm))
      .reduce((sum, f) => sum + f.n_runs, 0)
    const ok = window.confirm(
      `Delete saved runs for ${selected.length} algorithm folder(s) `
      + `(~${total} files) under ${problem}?`,
    )
    if (!ok) return
    setBusy(true)
    setError('')
    setNotice('')
    let removed = 0
    try {
      for (const algorithm of selected) {
        const res = await api.deleteResultFolder(problem, algorithm)
        removed += res.removed
      }
      setNotice(`Deleted ${removed} file(s) across ${selected.length} folder(s).`)
      setSelected([])
      setTable(null)
      await loadFolders(problem)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const downloadCsv = () => {
    if (!table) return
    const blob = new Blob([toCsv(table)], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `compare_${table.problem}_${table.metric}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const metricOptions = useMemo(() => {
    const hidden = new Set([
      'retrievals',
      'total_moves',
      'steps',
      'progress',
      'optimal_proven',
    ])
    const raw = table?.available_metrics?.length
      ? table.available_metrics
      : ['relocations', 'crane_time', 'time', 'steps']
    return raw.filter((m) => !hidden.has(m))
  }, [table])

  return (
    <main className="page compare-page">
      <header className="page-header">
        <div>
          <span className="eyebrow">PlatEMO-style experiment table</span>
          <h1>Compare</h1>
          <p>
            Select algorithm folders under <code>results/</code>, compare class-wise
            mean ± std, or delete whole folders to clean disk.
          </p>
        </div>
        <button
          className="primary"
          disabled={loading || busy || !selected.length}
          onClick={runCompare}
        >
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </header>

      {error && (
        <div className="error-banner">
          <strong>Compare error</strong>
          <pre>{error}</pre>
        </div>
      )}
      {notice && <div className="success-banner">{notice}</div>}

      <div className="compare-controls card">
        <label className="field">
          <span>Problem</span>
          <select value={problem} onChange={(e) => setProblem(e.target.value)}>
            {problems.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Metric</span>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {metricOptions.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={dedup}
            onChange={(e) => setDedup(e.target.checked)}
          />
          Dedup latest run per layout (for legacy timestamped files)
        </label>
        <div className="compare-quick">
          <button type="button" onClick={() => selectTop(3)} disabled={!folders.length || busy}>
            Select 3 newest
          </button>
          <button type="button" onClick={() => setSelected([])} disabled={busy}>Clear</button>
          <button type="button" onClick={() => void loadFolders(problem)} disabled={busy}>
            Refresh folders
          </button>
          <button
            type="button"
            className="danger-ghost"
            disabled={busy || !selected.length}
            onClick={() => void deleteSelectedFolders()}
          >
            Delete selected folders
          </button>
        </div>
      </div>

      <article className="card compare-folders">
        <div className="card-heading">
          <div>
            <span className="eyebrow">Result folders</span>
            <h2>{problem}</h2>
          </div>
          <span>{selected.length} selected · {folders.length} available</span>
        </div>
        {!folders.length && (
          <p className="description">
            No saved runs under <code>results/{problem}/</code>. Run batches in
            Workbench first.
          </p>
        )}
        <div className="folder-list">
          {folders.map((folder) => {
            const checked = selected.includes(folder.algorithm)
            return (
              <div className={`folder-row${checked ? ' selected' : ''}`} key={folder.folder}>
                <label className="folder-main">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(folder.algorithm)}
                  />
                  <div>
                    <strong>{folder.algorithm}</strong>
                    <span>
                      {folder.category || '—'} · {folder.n_runs} files
                      {folder.latest ? ` · ${folder.latest.slice(0, 16)}` : ''}
                    </span>
                  </div>
                </label>
                <code>{folder.folder_name}</code>
                <button
                  type="button"
                  className="danger-ghost"
                  disabled={busy}
                  onClick={() => void deleteFolder(folder.algorithm, folder.n_runs)}
                >
                  Delete
                </button>
              </div>
            )
          })}
        </div>
      </article>

      {table && (
        <article className="card compare-table-card">
          <div className="card-heading">
            <div>
              <span className="eyebrow">Mean ± std by class</span>
              <h2>{table.metric}</h2>
            </div>
            <div className="compare-actions">
              <span>
                {table.lower_is_better ? 'Lower is better' : 'Higher is better'}
                {table.dedup ? ' · deduped' : ''}
              </span>
              <button type="button" onClick={downloadCsv}>Export CSV</button>
            </div>
          </div>
          <div className="table-card batch-summary-table">
            <table>
              <thead>
                <tr>
                  <th>Class</th>
                  {table.algorithms.map((algo) => (
                    <th key={algo} colSpan={2}>{algo}</th>
                  ))}
                </tr>
                <tr>
                  <th />
                  {table.algorithms.flatMap((algo) => [
                    <th key={`${algo}-ms`}>mean ± std</th>,
                    <th key={`${algo}-n`}>n</th>,
                  ])}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row) => (
                  <tr key={row.label}>
                    <td><code>{row.label}</code></td>
                    {table.algorithms.flatMap((algo) => {
                      const cell = row.cells[algo]
                      return [
                        <td
                          key={`${row.label}-${algo}-v`}
                          className={cell?.best ? 'best-cell' : undefined}
                        >
                          {cell?.mean == null
                            ? '—'
                            : `${fmt(cell.mean)} ± ${fmt(cell.std)}`}
                        </td>,
                        <td key={`${row.label}-${algo}-n`}>{cell?.n ?? 0}</td>,
                      ]
                    })}
                  </tr>
                ))}
                {!table.rows.length && (
                  <tr>
                    <td colSpan={1 + table.algorithms.length * 2}>
                      No class-grouped runs (need layout paths in saved results).
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="description">
            Highlighted cells are best for <code>{table.metric}</code>
            {' '}({table.lower_is_better ? 'minimum' : 'maximum'} mean).
            Files used:{' '}
            {table.algorithms.map((a) => `${a}=${table.n_files[a] ?? 0}`).join(' · ')}
          </p>
        </article>
      )}
    </main>
  )
}
