import type { CompareTable } from './types'

function fmt(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

export function downloadText(filename: string, text: string, mime: string) {
  const blob = new Blob([text], { type: mime })
  downloadBlob(filename, blob)
}

export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function exportStem(table: CompareTable) {
  const safe = (s: string) => s.replace(/[^\w.-]+/g, '_')
  return `compare_${safe(table.problem)}_${safe(table.metric)}`
}

function latexEscape(text: string) {
  return text
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/([#$%&_{}])/g, '\\$1')
    .replace(/~/g, '\\textasciitilde{}')
    .replace(/\^/g, '\\textasciicircum{}')
}

function classToLatex(label: string) {
  const match = label.match(/^(\d+)\s*[×xX]\s*(\d+)$/)
  if (match) return `$${match[1]}\\times ${match[2]}$`
  return latexEscape(label)
}

/** Booktabs-free LaTeX tabular for the current Compare table (display only). */
export function toLatex(table: CompareTable): string {
  const algos = table.algorithms
  const colspec = `l${'cc'.repeat(algos.length)}`
  const header1 = [
    'Class',
    ...algos.map((algo) => `\\multicolumn{2}{c}{${latexEscape(algo)}}`),
  ].join(' & ')
  const header2 = [
    '',
    ...algos.flatMap(() => ['mean $\\pm$ std', '$n$']),
  ].join(' & ')
  const body = table.rows.map((row) => {
    const cells = [classToLatex(row.label)]
    for (const algo of algos) {
      const cell = row.cells[algo]
      if (cell?.mean == null) {
        cells.push('--', String(cell?.n ?? 0))
      } else {
        const inner = `${cell.mean.toFixed(3)} \\pm ${
          cell.std == null || Number.isNaN(cell.std) ? '--' : cell.std.toFixed(3)
        }`
        cells.push(cell.best ? `$\\mathbf{${inner}}$` : `$${inner}$`)
        cells.push(String(cell.n))
      }
    }
    return `${cells.join(' & ')} \\\\`
  })
  const caption = `${latexEscape(table.problem)} ${latexEscape(table.metric)} (mean $\\pm$ std by class)`
  const label = `tab:${exportStem(table)}`
  return [
    '\\begin{table}[htbp]',
    '\\centering',
    `\\caption{${caption}}`,
    `\\label{${label}}`,
    `\\begin{tabular}{${colspec}}`,
    '\\hline',
    `${header1} \\\\`,
    `${header2} \\\\`,
    '\\hline',
    ...body,
    '\\hline',
    '\\end{tabular}',
    '\\end{table}',
    '',
  ].join('\n')
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
): string[] {
  const words = text.split(/\s+/).filter(Boolean)
  if (!words.length) return ['']
  const lines: string[] = []
  let current = words[0]
  for (const word of words.slice(1)) {
    const trial = `${current} ${word}`
    if (ctx.measureText(trial).width <= maxWidth) {
      current = trial
    } else {
      lines.push(current)
      current = word
    }
  }
  lines.push(current)
  return lines
}

/** Paint the Compare table onto a PNG (no algorithm data is recomputed). */
export function tableToPngBlob(table: CompareTable): Promise<Blob> {
  const scale = 2
  const pad = 28
  const titleH = 36
  const footerH = 22
  const subHeaderH = 26
  const rowH = 28
  const nColW = 52
  const minMeanW = 132
  const classMinW = 72

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return Promise.reject(new Error('Canvas is not available in this browser.'))
  }

  ctx.font = '13px ui-sans-serif, system-ui, sans-serif'
  const classW = Math.max(
    classMinW,
    ctx.measureText('Class').width,
    ...table.rows.map((row) => ctx.measureText(row.label).width),
  ) + 20

  const meanWidths = table.algorithms.map((algo) => {
    let width = ctx.measureText('mean ± std').width
    for (const row of table.rows) {
      const cell = row.cells[algo]
      const text = cell?.mean == null
        ? '—'
        : `${fmt(cell.mean)} ± ${fmt(cell.std)}`
      width = Math.max(width, ctx.measureText(text).width)
    }
    return Math.max(minMeanW, width + 18)
  })

  ctx.font = 'bold 12px ui-sans-serif, system-ui, sans-serif'
  const wrappedHeaders = table.algorithms.map((algo, index) => {
    const span = meanWidths[index] + nColW - 12
    return wrapText(ctx, algo, span)
  })
  const headerLines = Math.max(1, ...wrappedHeaders.map((lines) => lines.length))
  const headerH = 14 + headerLines * 16

  const tableW = classW
    + meanWidths.reduce((sum, w) => sum + w + nColW, 0)
  const tableH = headerH + subHeaderH + Math.max(table.rows.length, 1) * rowH
  const width = pad * 2 + tableW
  const height = pad * 2 + titleH + tableH + footerH

  canvas.width = Math.ceil(width * scale)
  canvas.height = Math.ceil(height * scale)
  ctx.setTransform(scale, 0, 0, scale, 0, 0)

  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, width, height)

  ctx.fillStyle = '#101828'
  ctx.font = 'bold 16px ui-sans-serif, system-ui, sans-serif'
  ctx.fillText(
    `${table.problem}  ·  ${table.metric}  ·  mean ± std by class`,
    pad,
    pad + 18,
  )
  ctx.font = '12px ui-sans-serif, system-ui, sans-serif'
  ctx.fillStyle = '#667085'
  ctx.fillText(
    table.lower_is_better ? 'Lower mean is better (best cells in green).' : 'Higher mean is better (best cells in green).',
    pad,
    pad + 34,
  )

  const originX = pad
  const originY = pad + titleH
  const colXs: number[] = [originX]
  let x = originX + classW
  colXs.push(x)
  for (let i = 0; i < table.algorithms.length; i += 1) {
    x += meanWidths[i]
    colXs.push(x)
    x += nColW
    colXs.push(x)
  }

  ctx.fillStyle = '#f8fafc'
  ctx.fillRect(originX, originY, tableW, headerH + subHeaderH)

  ctx.strokeStyle = '#e4e7ec'
  ctx.lineWidth = 1
  ctx.strokeRect(originX + 0.5, originY + 0.5, tableW - 1, tableH - 1)

  ctx.fillStyle = '#344054'
  ctx.font = 'bold 12px ui-sans-serif, system-ui, sans-serif'
  ctx.fillText('Class', originX + 10, originY + headerH - 8)

  table.algorithms.forEach((_, index) => {
    const left = colXs[1 + index * 2]
    const span = meanWidths[index] + nColW
    const lines = wrappedHeaders[index]
    ctx.textAlign = 'center'
    lines.forEach((line, lineIndex) => {
      ctx.fillText(
        line,
        left + span / 2,
        originY + 16 + lineIndex * 16,
      )
    })
    ctx.textAlign = 'left'
  })

  const subY = originY + headerH
  ctx.beginPath()
  ctx.moveTo(originX, subY + 0.5)
  ctx.lineTo(originX + tableW, subY + 0.5)
  ctx.stroke()

  ctx.font = '11px ui-sans-serif, system-ui, sans-serif'
  ctx.fillStyle = '#667085'
  table.algorithms.forEach((_, index) => {
    const meanX = colXs[1 + index * 2]
    ctx.fillText('mean ± std', meanX + 8, subY + 18)
    ctx.fillText('n', colXs[2 + index * 2] + 8, subY + 18)
  })

  table.rows.forEach((row, rowIndex) => {
    const y = originY + headerH + subHeaderH + rowIndex * rowH
    if (rowIndex % 2 === 1) {
      ctx.fillStyle = '#fcfcfd'
      ctx.fillRect(originX, y, tableW, rowH)
    }
    ctx.beginPath()
    ctx.strokeStyle = '#e4e7ec'
    ctx.moveTo(originX, y + 0.5)
    ctx.lineTo(originX + tableW, y + 0.5)
    ctx.stroke()

    ctx.font = '13px ui-sans-serif, system-ui, sans-serif'
    ctx.fillStyle = '#101828'
    ctx.fillText(row.label, originX + 10, y + 19)

    table.algorithms.forEach((algo, index) => {
      const cell = row.cells[algo]
      const meanX = colXs[1 + index * 2]
      const nX = colXs[2 + index * 2]
      const text = cell?.mean == null
        ? '—'
        : `${fmt(cell.mean)} ± ${fmt(cell.std)}`
      if (cell?.best) {
        ctx.fillStyle = '#ecfdf3'
        ctx.fillRect(meanX + 1, y + 1, meanWidths[index] - 1, rowH - 1)
        ctx.fillStyle = '#027a48'
        ctx.font = 'bold 13px ui-sans-serif, system-ui, sans-serif'
      } else {
        ctx.fillStyle = '#101828'
        ctx.font = '13px ui-sans-serif, system-ui, sans-serif'
      }
      ctx.fillText(text, meanX + 8, y + 19)
      ctx.fillStyle = '#344054'
      ctx.font = '13px ui-sans-serif, system-ui, sans-serif'
      ctx.fillText(String(cell?.n ?? 0), nX + 8, y + 19)
    })
  })

  ctx.fillStyle = '#98a2b3'
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif'
  ctx.fillText(
    'Platform_CRISP Compare (values unchanged from saved runs)',
    pad,
    originY + tableH + 16,
  )

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) reject(new Error('Could not encode PNG.'))
      else resolve(blob)
    }, 'image/png')
  })
}
