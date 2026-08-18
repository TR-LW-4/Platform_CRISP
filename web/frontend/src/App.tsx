import { useCallback, useEffect, useState } from 'react'
import './App.css'
import { api } from './api'
import { Compare } from './pages/Compare'
import { Jobs } from './pages/Jobs'
import { Workbench } from './pages/Workbench'
import type { Catalog } from './types'

type Page = 'workbench' | 'jobs' | 'compare' | 'about'

const navigation: { id: Page; label: string; short: string }[] = [
  { id: 'workbench', label: 'Test workbench', short: 'T' },
  { id: 'jobs', label: 'Jobs', short: 'J' },
  { id: 'compare', label: 'Compare', short: 'C' },
  { id: 'about', label: 'About', short: 'A' },
]

function App() {
  const [page, setPage] = useState<Page>('workbench')
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [catalogError, setCatalogError] = useState('')
  const [resumeJobId, setResumeJobId] = useState<string | null>(null)
  const clearResume = useCallback(() => setResumeJobId(null), [])

  useEffect(() => {
    api.catalog()
      .then(setCatalog)
      .catch((error) => setCatalogError(error instanceof Error ? error.message : String(error)))
  }, [])

  const openJob = (jobId: string) => {
    setResumeJobId(jobId)
    setPage('workbench')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">C</div>
          <div>
            <strong>Platform CRISP</strong>
            <span>Research workbench</span>
          </div>
        </div>
        <nav>
          {navigation.map((item) => (
            <button
              className={page === item.id ? 'active' : ''}
              key={item.id}
              onClick={() => setPage(item.id)}
            >
              <span>{item.short}</span>
              {item.label}
            </button>
          ))}
        </nav>
        <div className="server-status">
          <i className={catalog ? 'online' : 'offline'} />
          {catalog ? 'API connected' : 'Connecting…'}
        </div>
      </aside>

      <section className="content">
        {catalogError && (
          <div className="fatal-state">
            <h1>Unable to load Platform CRISP</h1>
            <p>{catalogError}</p>
          </div>
        )}
        {!catalog && !catalogError && <div className="loading-state">Loading registry…</div>}
        {catalog && (
          <>
            <div className={page === 'workbench' ? '' : 'page-hidden'}>
              <Workbench
                catalog={catalog}
                resumeJobId={resumeJobId}
                onResumeHandled={clearResume}
              />
            </div>
            <div className={page === 'jobs' ? '' : 'page-hidden'}>
              <Jobs onOpen={openJob} />
            </div>
            <div className={page === 'compare' ? '' : 'page-hidden'}>
              <Compare catalog={catalog} active={page === 'compare'} />
            </div>
            <div className={page === 'about' ? '' : 'page-hidden'}>
              <main className="page about-page">
                <span className="eyebrow">Platform architecture</span>
                <h1>Stable by separation</h1>
                <p>
                  React keeps interface state in the browser. FastAPI owns the job
                  lifecycle. Existing CRISP algorithms execute unchanged in isolated
                  Python processes. Saved runs live under <code>results/</code>; use
                  Compare for mean±std tables and folder cleanup.
                </p>
                <div className="architecture card">
                  <span>React interface</span><b>→</b>
                  <span>FastAPI service</span><b>→</b>
                  <span>Algorithm process</span>
                </div>
              </main>
            </div>
          </>
        )}
      </section>
    </div>
  )
}

export default App
