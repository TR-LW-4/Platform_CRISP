import { useEffect, useState } from 'react'
import { api } from '../api'
import type { JobSummary } from '../types'

interface JobsProps {
  onOpen: (jobId: string) => void
}

const terminal = new Set(['completed', 'stopped', 'failed'])

export function Jobs({ onOpen }: JobsProps) {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [error, setError] = useState('')
  const [busyId, setBusyId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const latest = await api.jobs()
        if (!cancelled) {
          setJobs(latest)
          setError('')
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : String(loadError))
        }
      }
    }
    void refresh()
    const timer = window.setInterval(refresh, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const removeJob = async (job: JobSummary, deleteFiles: boolean) => {
    const msg = deleteFiles
      ? `Remove job and delete ${job.result_files?.length ?? 0} saved result file(s)?`
      : 'Remove this job from the server list? (Saved result files on disk are kept.)'
    if (!window.confirm(msg)) return
    setBusyId(job.id)
    setError('')
    try {
      await api.deleteJob(job.id, deleteFiles)
      setJobs((current) => current.filter((item) => item.id !== job.id))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusyId(null)
    }
  }

  return (
    <main className="page">
      <header className="page-header">
        <div>
          <span className="eyebrow">Server-side lifecycle</span>
          <h1>Jobs</h1>
          <p>Runs continue when the browser tab is closed. Remove clears the in-memory job.</p>
        </div>
      </header>
      {error && <div className="error-banner">{error}</div>}
      <div className="jobs-scroll">
        <div className="jobs-grid">
          {jobs.map((job) => (
            <article className="job-card card" key={job.id}>
              <div>
                <span className={`status-badge status-${job.status}`}>{job.status}</span>
                <span className="job-time">{job.created_at}</span>
              </div>
              <h2>{job.algorithm_name}</h2>
              <p>{job.problem_name} · {job.category}</p>
              <div className="job-progress">
                <div style={{ width: `${(job.latest?.progress ?? 0) * 100}%` }} />
              </div>
              <footer className="job-footer">
                <span>{job.record_count} updates</span>
                <div className="job-actions">
                  <button onClick={() => onOpen(job.id)}>Open</button>
                  {terminal.has(job.status) || job.status === 'failed' ? (
                    <>
                      <button
                        disabled={busyId === job.id}
                        onClick={() => void removeJob(job, false)}
                      >
                        Remove
                      </button>
                      <button
                        className="danger-ghost"
                        disabled={busyId === job.id}
                        onClick={() => void removeJob(job, true)}
                      >
                        Remove + files
                      </button>
                    </>
                  ) : (
                    <button
                      disabled={busyId === job.id}
                      onClick={() => void removeJob(job, false)}
                    >
                      Stop & remove
                    </button>
                  )}
                </div>
              </footer>
            </article>
          ))}
        </div>
        {!jobs.length && <div className="empty-visual card">No jobs in this server process.</div>}
      </div>
    </main>
  )
}
