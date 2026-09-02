export type ConfigValue = string | number | boolean

export interface ConfigField {
  type: 'int' | 'float' | 'bool' | 'str'
  default?: ConfigValue
  min?: number
  max?: number
  step?: number
  label?: string
  help?: string
  options?: ConfigValue[]
  choices?: ConfigValue[]
}

export type ConfigSchema = Record<string, ConfigField>
export type ConfigValues = Record<string, ConfigValue>

export interface ProblemInfo {
  name: string
  description: string
  tags: string[]
  metric_names: string[]
  config_schema: ConfigSchema
}

export interface AlgorithmInfo {
  name: string
  category: string
  method_group: string
  description: string
  compatible_problems: string[]
  requires_solver: boolean
  solver_backend: string | null
  step_label: string
  config_schema: ConfigSchema
}

export interface Catalog {
  problems: ProblemInfo[]
  algorithms: AlgorithmInfo[]
}

export interface YardStack {
  bay: number
  row: number
  containers: number[]
}

export interface YardSnapshot {
  yard: YardStack[]
  metrics?: Record<string, number>
  history?: unknown[]
}

export interface ProgressRecord {
  sequence: number
  step: number
  metric: number | null
  metrics: Record<string, number | null>
  best_metric: number | null
  progress: number
  yard_snapshot: YardSnapshot | null
  extra: Record<string, unknown>
}

export type JobStatus =
  | 'starting'
  | 'running'
  | 'stopping'
  | 'completed'
  | 'stopped'
  | 'failed'

export interface MetricAgg {
  mean: number
  std: number
}

export interface BatchClassSummary {
  source: string
  label: string
  n: number
  h?: number
  w?: number
  s?: number
  metrics: Record<string, MetricAgg>
}

export interface BatchSummary {
  classes: BatchClassSummary[]
  ungrouped: number
  total_runs: number
  error?: string
}

export interface JobSummary {
  id: string
  problem_name: string
  algorithm_name: string
  category: string
  problem_config: ConfigValues
  algorithm_config: ConfigValues
  instance_source?: string
  batch_total?: number | null
  batch_index?: number | null
  completed_count?: number | null
  pending_count?: number | null
  status: JobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  record_count: number
  latest: ProgressRecord | null
  records?: ProgressRecord[]
  error: string | null
  result_file: string | null
  result_files?: string[]
  batch_summary?: BatchSummary | null
}

export interface RecordsResponse {
  job_id: string
  status: JobStatus
  records: ProgressRecord[]
  next_sequence: number
  error: string | null
  result_file: string | null
  result_files?: string[]
  batch_summary?: BatchSummary | null
  batch_total?: number | null
  batch_index?: number | null
  completed_count?: number | null
  pending_count?: number | null
  instance_source?: string
}

export interface SNPair {
  s: number
  n: number
}

export interface BenchmarkQueueBlock {
  h: number
  ws?: number[]
  sn_pairs?: SNPair[]
}

export interface BenchmarkSourceInfo {
  id: string
  label: string
  available: boolean
  caption?: string
  heights?: number[]
  ws_by_height?: Record<string, number[]>
  sn_by_height?: Record<string, SNPair[]>
  alphas?: string[]
  alpha_indexes?: Record<
    string,
    {
      heights: number[]
      sn_by_height: Record<string, SNPair[]>
    }
  >
}

export interface BenchmarkSourcesResponse {
  problem_name: string
  sources: BenchmarkSourceInfo[]
  notes?: string | null
}

export interface ResolveBenchmarkResponse {
  count: number
  sample_paths: string[]
  first_only: boolean
}

export interface ResultSummary {
  file: string
  problem: string
  algorithm: string
  category: string
  seed: number
  timestamp: string
  metrics: Record<string, number>
  n_steps: number
}

export interface CompareFolder {
  problem: string
  algorithm: string
  folder: string
  folder_name: string
  category: string
  n_runs: number
  latest: string
}

export interface CompareCell {
  n: number
  mean: number | null
  std: number | null
  best: boolean
}

export interface CompareRow {
  label: string
  source?: string
  h?: number
  w?: number
  s?: number
  cells: Record<string, CompareCell>
}

export interface CompareTable {
  problem: string
  metric: string
  lower_is_better: boolean
  algorithms: string[]
  available_metrics: string[]
  dedup: boolean
  limit: number | null
  n_files: Record<string, number>
  rows: CompareRow[]
  per_algorithm: Record<
    string,
    { total_runs: number; ungrouped: number; n_classes: number }
  >
}
