import type {
  BenchmarkQueueBlock,
  BenchmarkSourcesResponse,
  Catalog,
  CompareFolder,
  CompareTable,
  ConfigValues,
  JobSummary,
  RecordsResponse,
  ResolveBenchmarkResponse,
  ResultSummary,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const payload = await response.json()
      message = payload.detail ?? message
    } catch {
      // Keep the HTTP fallback.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  catalog: () => request<Catalog>('/api/catalog'),
  benchmarks: (problem: string) =>
    request<BenchmarkSourcesResponse>(
      `/api/benchmarks?problem=${encodeURIComponent(problem)}`,
    ),
  resolveBenchmark: (payload: {
    source: string
    queue: BenchmarkQueueBlock[]
    alpha?: string
    first_only?: boolean
  }) =>
    request<ResolveBenchmarkResponse>('/api/benchmarks/resolve', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  jobs: () => request<JobSummary[]>('/api/jobs'),
  job: (id: string) => request<JobSummary>(`/api/jobs/${id}`),
  records: (id: string, after: number) =>
    request<RecordsResponse>(`/api/jobs/${id}/records?after=${after}`),
  startJob: (payload: {
    problemName: string
    algorithmName: string
    problemConfig: ConfigValues
    algorithmConfig: ConfigValues
    instanceSource?: string
    benchmarkQueue?: BenchmarkQueueBlock[]
    zhuDupAlpha?: string
    firstOnly?: boolean
  }) =>
    request<JobSummary>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({
        problem_name: payload.problemName,
        algorithm_name: payload.algorithmName,
        problem_config: payload.problemConfig,
        algorithm_config: payload.algorithmConfig,
        instance_source: payload.instanceSource ?? 'random',
        benchmark_queue: payload.benchmarkQueue ?? [],
        zhu_dup_alpha: payload.zhuDupAlpha,
        first_only: payload.firstOnly ?? false,
      }),
    }),
  stopJob: (id: string) =>
    request<JobSummary>(`/api/jobs/${id}/stop`, { method: 'POST' }),
  deleteJob: (id: string, deleteFiles = false) =>
    request<JobSummary>(
      `/api/jobs/${id}?delete_files=${deleteFiles ? 'true' : 'false'}`,
      { method: 'DELETE' },
    ),
  results: () => request<ResultSummary[]>('/api/results'),
  deleteResult: (path: string) =>
    request<{ ok: boolean; path: string }>(
      `/api/results?path=${encodeURIComponent(path)}`,
      { method: 'DELETE' },
    ),
  deleteResultFolder: (problem: string, algorithm: string) =>
    request<{ ok: boolean; removed: number }>(
      `/api/results/folder?problem=${encodeURIComponent(problem)}`
        + `&algorithm=${encodeURIComponent(algorithm)}`,
      { method: 'DELETE' },
    ),
  compareFolders: (problem: string) =>
    request<CompareFolder[]>(
      `/api/compare/folders?problem=${encodeURIComponent(problem)}`,
    ),
  compare: (payload: {
    problem: string
    algorithms: string[]
    metric?: string
    dedup?: boolean
  }) =>
    request<CompareTable>('/api/compare', {
      method: 'POST',
      body: JSON.stringify({
        problem: payload.problem,
        algorithms: payload.algorithms,
        metric: payload.metric ?? 'relocations',
        dedup: payload.dedup ?? true,
      }),
    }),
}
