import type { ConfigValues, YardSnapshot } from '../types'

interface YardViewProps {
  snapshot: YardSnapshot | null
  problemConfig: ConfigValues
}

const colours = [
  '#2563eb',
  '#f97316',
  '#dc2626',
  '#0d9488',
  '#16a34a',
  '#ca8a04',
  '#9333ea',
  '#db2777',
  '#7c3aed',
  '#0891b2',
]

export function YardView({ snapshot, problemConfig }: YardViewProps) {
  if (!snapshot?.yard?.length) {
    return <div className="empty-visual">No yard snapshot in this update</div>
  }

  const numBays = Number(problemConfig.num_bays ?? 4)
  const numRows = Number(problemConfig.num_rows ?? 2)
  const maxTiers = Number(problemConfig.max_tiers ?? 3)
  const byPosition = new Map(
    snapshot.yard.map((stack) => [`${stack.bay}-${stack.row}`, stack.containers]),
  )
  const positions = Array.from({ length: numBays * numRows }, (_, index) => ({
    bay: Math.floor(index / numRows) + 1,
    row: (index % numRows) + 1,
  }))

  return (
    <div
      className="yard"
      style={{ gridTemplateColumns: `repeat(${positions.length}, minmax(42px, 1fr))` }}
    >
      {positions.map(({ bay, row }) => {
        const containers = byPosition.get(`${bay}-${row}`) ?? []
        return (
          <div className="stack-column" key={`${bay}-${row}`}>
            <div
              className="stack-cells"
              style={{ gridTemplateRows: `repeat(${maxTiers}, 34px)` }}
            >
              {Array.from({ length: maxTiers }, (_, displayTier) => {
                const tier = maxTiers - displayTier - 1
                const rawGroup = containers[tier]
                const group = rawGroup === undefined ? null : Math.max(0, rawGroup)
                return (
                  <div
                    className={`container-cell ${group === null ? 'empty' : ''}`}
                    key={tier}
                    style={
                      group === null
                        ? undefined
                        : { backgroundColor: colours[group % colours.length] }
                    }
                  >
                    {group}
                  </div>
                )
              })}
            </div>
            <span>B{bay}R{row}</span>
          </div>
        )
      })}
    </div>
  )
}
