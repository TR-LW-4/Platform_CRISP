import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type {
  BenchmarkQueueBlock,
  BenchmarkSourceInfo,
  BenchmarkSourcesResponse,
  SNPair,
  StowClass,
} from '../types'

interface InstanceSourcePanelProps {
  problemName: string
  disabled?: boolean
  source: string
  queue: BenchmarkQueueBlock[]
  alpha: string
  firstOnly: boolean
  onSourceChange: (source: string) => void
  onQueueChange: (queue: BenchmarkQueueBlock[]) => void
  onAlphaChange: (alpha: string) => void
  onFirstOnlyChange: (value: boolean) => void
  onResolvedCount: (count: number) => void
}

function pairKey(pair: SNPair) {
  return `${pair.s}-${pair.n}`
}

function stowClassKey(item: StowClass) {
  return `${item.vs}-${item.ys}-${item.yt}`
}

export function InstanceSourcePanel({
  problemName,
  disabled = false,
  source,
  queue,
  alpha,
  firstOnly,
  onSourceChange,
  onQueueChange,
  onAlphaChange,
  onFirstOnlyChange,
  onResolvedCount,
}: InstanceSourcePanelProps) {
  const [catalog, setCatalog] = useState<BenchmarkSourcesResponse | null>(null)
  const [error, setError] = useState('')
  const [height, setHeight] = useState<number | ''>('')
  const [selectedWs, setSelectedWs] = useState<number[]>([])
  const [selectedPairs, setSelectedPairs] = useState<string[]>([])
  const [selectedStowClasses, setSelectedStowClasses] = useState<string[]>([])
  const [resolvedCount, setResolvedCount] = useState(0)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const payload = await api.benchmarks(problemName)
        if (cancelled) return
        setCatalog(payload)
        setError('')
        const available = payload.sources.filter((item) => item.available)
        if (!available.some((item) => item.id === source)) {
          onSourceChange(available[0]?.id ?? '')
        }
        const dup = payload.sources.find((item) => item.id === 'zhu_dup')
        if (dup?.alphas?.length && !dup.alphas.includes(alpha)) {
          onAlphaChange(dup.alphas[0])
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : String(loadError))
        }
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [problemName])

  const activeSource: BenchmarkSourceInfo | undefined = catalog?.sources.find(
    (item) => item.id === source,
  )

  const snIndex = useMemo(() => {
    if (!activeSource) return null
    if (source === 'zhu_dup') {
      return activeSource.alpha_indexes?.[alpha] ?? null
    }
    if (source === 'zhu') {
      return {
        heights: activeSource.heights ?? [],
        sn_by_height: activeSource.sn_by_height ?? {},
      }
    }
    return null
  }, [activeSource, alpha, source])

  const heights = useMemo(() => {
    if (source === 'caserta' || source === 'crp_stow') {
      return activeSource?.heights ?? []
    }
    return snIndex?.heights ?? []
  }, [activeSource, snIndex, source])

  useEffect(() => {
    if (!heights.length) {
      setHeight('')
      return
    }
    if (height === '' || !heights.includes(height)) {
      setHeight(heights[0])
      setSelectedWs([])
      setSelectedPairs([])
      setSelectedStowClasses([])
    }
  }, [heights, height])

  const wsOptions = useMemo(() => {
    if (source !== 'caserta' || height === '' || !activeSource?.ws_by_height) return []
    return activeSource.ws_by_height[String(height)] ?? []
  }, [activeSource, height, source])

  const pairOptions = useMemo(() => {
    if (!snIndex || height === '') return []
    return snIndex.sn_by_height[String(height)] ?? []
  }, [height, snIndex])

  const stowClassOptions = useMemo(() => {
    if (
      source !== 'crp_stow' ||
      height === '' ||
      !activeSource?.stow_classes_by_height
    ) return []
    return activeSource.stow_classes_by_height[String(height)] ?? []
  }, [activeSource, height, source])

  useEffect(() => {
    if (source === 'random') {
      setResolvedCount(0)
      onResolvedCount(0)
      return
    }
    let cancelled = false
    const resolve = async () => {
      try {
        const result = await api.resolveBenchmark({
          source,
          queue,
          alpha: source === 'zhu_dup' ? alpha : undefined,
          first_only: firstOnly,
        })
        if (cancelled) return
        setResolvedCount(result.count)
        onResolvedCount(result.count)
        setError('')
      } catch (resolveError) {
        if (!cancelled) {
          setResolvedCount(0)
          onResolvedCount(0)
          setError(
            resolveError instanceof Error ? resolveError.message : String(resolveError),
          )
        }
      }
    }
    void resolve()
    return () => {
      cancelled = true
    }
  }, [alpha, firstOnly, onResolvedCount, queue, source])

  const addToQueue = () => {
    if (height === '') return
    if (source === 'caserta') {
      if (!selectedWs.length) return
      const next = queue.map((block) => ({ ...block, ws: [...(block.ws ?? [])] }))
      const existing = next.find((block) => block.h === height)
      if (existing) {
        existing.ws = [...new Set([...(existing.ws ?? []), ...selectedWs])].sort(
          (a, b) => a - b,
        )
      } else {
        next.push({ h: height, ws: [...selectedWs].sort((a, b) => a - b) })
      }
      onQueueChange(next.sort((a, b) => a.h - b.h))
      setSelectedWs([])
      return
    }
    if (source === 'crp_stow') {
      const classes = stowClassOptions.filter((item) =>
        selectedStowClasses.includes(stowClassKey(item)),
      )
      if (!classes.length) return
      const next = queue.map((block) => ({
        ...block,
        stow_classes: [...(block.stow_classes ?? [])],
      }))
      const existing = next.find((block) => block.h === height)
      if (existing) {
        const merged = new Map(
          (existing.stow_classes ?? []).map((item) => [stowClassKey(item), item]),
        )
        classes.forEach((item) => merged.set(stowClassKey(item), item))
        existing.stow_classes = [...merged.values()].sort(
          (a, b) => a.vs - b.vs || a.ys - b.ys || a.yt - b.yt,
        )
      } else {
        next.push({
          h: height,
          stow_classes: [...classes].sort(
            (a, b) => a.vs - b.vs || a.ys - b.ys || a.yt - b.yt,
          ),
        })
      }
      onQueueChange(next.sort((a, b) => a.h - b.h))
      setSelectedStowClasses([])
      return
    }
    const pairs = pairOptions.filter((pair) => selectedPairs.includes(pairKey(pair)))
    if (!pairs.length) return
    const next = queue.map((block) => ({
      ...block,
      sn_pairs: [...(block.sn_pairs ?? [])],
    }))
    const existing = next.find((block) => block.h === height)
    if (existing) {
      const merged = new Map(
        (existing.sn_pairs ?? []).map((pair) => [pairKey(pair), pair]),
      )
      pairs.forEach((pair) => merged.set(pairKey(pair), pair))
      existing.sn_pairs = [...merged.values()].sort(
        (a, b) => a.s - b.s || a.n - b.n,
      )
    } else {
      next.push({
        h: height,
        sn_pairs: [...pairs].sort((a, b) => a.s - b.s || a.n - b.n),
      })
    }
    onQueueChange(next.sort((a, b) => a.h - b.h))
    setSelectedPairs([])
  }

  const removeBlock = (index: number) => {
    onQueueChange(queue.filter((_, i) => i !== index))
  }

  return (
    <div className="instance-source">
      <label className="field">
        <span>Instance source</span>
        <select
          value={source}
          disabled={disabled || !catalog}
          onChange={(event) => {
            onSourceChange(event.target.value)
            onQueueChange([])
            setSelectedWs([])
            setSelectedPairs([])
            setSelectedStowClasses([])
          }}
        >
          {(catalog?.sources ?? []).map(
            (item) => (
              <option key={item.id} value={item.id} disabled={!item.available}>
                {item.label}{item.available ? '' : ' (unavailable)'}
              </option>
            ),
          )}
        </select>
      </label>

      {activeSource?.caption && source !== 'random' && (
        <p className="description">{activeSource.caption}</p>
      )}

      {source === 'zhu_dup' && (
        <label className="field">
          <span>Alpha</span>
          <select
            value={alpha}
            disabled={disabled}
            onChange={(event) => {
              onAlphaChange(event.target.value)
              onQueueChange([])
            }}
          >
            {(activeSource?.alphas ?? []).map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
      )}

      {source !== 'random' && (
        <>
          <label className="field">
            <span>
              {source === 'crp_stow' ? 'Minimum vessel height A' : 'Height H'}
            </span>
            <select
              value={height === '' ? '' : String(height)}
              disabled={disabled || !heights.length}
              onChange={(event) => {
                setHeight(Number(event.target.value))
                setSelectedWs([])
                setSelectedPairs([])
                setSelectedStowClasses([])
              }}
            >
              {heights.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>

          {source === 'caserta' ? (
            <fieldset className="multi-select" disabled={disabled}>
              <legend>Stack counts w</legend>
              <div className="chip-grid">
                {wsOptions.map((w) => {
                  const checked = selectedWs.includes(w)
                  return (
                    <label key={w} className={`chip ${checked ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          setSelectedWs((current) =>
                            checked
                              ? current.filter((item) => item !== w)
                              : [...current, w].sort((a, b) => a - b),
                          )
                        }}
                      />
                      w={w}
                    </label>
                  )
                })}
              </div>
            </fieldset>
          ) : source === 'crp_stow' ? (
            <fieldset className="multi-select" disabled={disabled}>
              <legend>BRLP classes (VS, YS, YT)</legend>
              <div className="chip-grid">
                {stowClassOptions.map((item) => {
                  const key = stowClassKey(item)
                  const checked = selectedStowClasses.includes(key)
                  return (
                    <label key={key} className={`chip ${checked ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          setSelectedStowClasses((current) =>
                            checked
                              ? current.filter((value) => value !== key)
                              : [...current, key],
                          )
                        }}
                      />
                      VS={item.vs}, YS={item.ys}, YT={item.yt}
                    </label>
                  )
                })}
              </div>
            </fieldset>
          ) : (
            <fieldset className="multi-select" disabled={disabled}>
              <legend>(S, N) scales</legend>
              <div className="chip-grid">
                {pairOptions.map((pair) => {
                  const key = pairKey(pair)
                  const checked = selectedPairs.includes(key)
                  return (
                    <label key={key} className={`chip ${checked ? 'active' : ''}`}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          setSelectedPairs((current) =>
                            checked
                              ? current.filter((item) => item !== key)
                              : [...current, key],
                          )
                        }}
                      />
                      S={pair.s}, N={pair.n}
                    </label>
                  )
                })}
              </div>
            </fieldset>
          )}

          <div className="run-actions compact">
            <button type="button" disabled={disabled} onClick={addToQueue}>
              Add to instance list
            </button>
            <button
              type="button"
              disabled={disabled || !queue.length}
              onClick={() => onQueueChange([])}
            >
              Clear list
            </button>
          </div>

          <div className="instance-list">
            <div className="card-heading">
              <strong>Instance list</strong>
              <span>{resolvedCount} file{resolvedCount === 1 ? '' : 's'}</span>
            </div>
            {!queue.length && <p className="muted">No blocks yet.</p>}
            <ul>
              {queue.map((block, index) => (
                <li key={`${block.h}-${index}`}>
                  <span>
                    {source === 'crp_stow' ? 'A' : 'H'}={block.h}
                    {block.ws
                      ? ` · w=${block.ws.join(',')}`
                      : block.stow_classes
                        ? ` · ${block.stow_classes
                            .map((item) => (
                              `VS=${item.vs}/YS=${item.ys}/YT=${item.yt}`
                            ))
                            .join(', ')}`
                      : ` · ${(block.sn_pairs ?? [])
                          .map((pair) => `${pair.s}-${pair.n}`)
                          .join(', ')}`}
                  </span>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => removeBlock(index)}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={firstOnly}
              disabled={disabled}
              onChange={(event) => onFirstOnlyChange(event.target.checked)}
            />
            First instance only (inspect one file)
          </label>
        </>
      )}

      {error && <p className="inline-error">{error}</p>}
    </div>
  )
}
