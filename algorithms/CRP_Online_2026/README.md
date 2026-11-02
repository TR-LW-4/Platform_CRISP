# CRP-Online (2026 Paper) Algorithms

This module provides the faithful Python implementation of the 10 published manual heuristics
and Genetic Programming rules from **Đurasević, Đumić, & Gil Gala (2026)**:
*Automated design of relocation rules with genetic programming for the online container relocation problem* (ESWA 2026).

## Registered Algorithms

| Algorithm Class | Display Name | Category | Primary Paper |
|---|---|---|---|
| `TLP2026Algorithm` | `TLP / Lowest Position (Lee & Lee 2010 / Zehendner 2017)` | Heuristic | Lee & Lee (2010), Zehendner et al. (2017) |
| `RI2026Algorithm` | `RI (Murty 2005 - 2026 Engine)` | Heuristic | Murty et al. (2005) |
| `MM2026Algorithm` | `Min-Max (Caserta 2012 - 2026 Engine)` | Heuristic | Caserta et al. (2012) |
| `PR42026Algorithm` | `PR4 (Jin 2015 - 2026 Engine)` | Heuristic | Jin et al. (2015) |
| `RDH2026Algorithm` | `RDH (Random Destination - 2026 Engine)` | Heuristic | Đurasević et al. (2026) |
| `ERI2026Algorithm` | `ERI (Expected Reshuffle Index - 2026 Engine)` | Heuristic | Ku & Arthanari (2016), Đurasević (2026) |
| `EM2026Algorithm` | `EM (Expected Min-Max - 2026 Engine)` | Heuristic | Galle et al. (2018), Đurasević (2026) |
| `GPOnline2026Algorithm` | `GP-Online (Durasevic 2026 Paper)` | Evolutionary | Đurasević et al. (2026) |

## Usage via Platform CRISP CLI

```bash
# List all registered problems and algorithms
python main.py list

# Run a single layout with TLP
python main.py layout-run --problem "CRP-Online (2026 Paper)" --algo "TLP / Lowest Position (Lee & Lee 2010 / Zehendner 2017)" --layout benchmark/Caserta_dataset/data3-3-1.dat

# Run a single layout with GP Rule 0
python main.py layout-run --problem "CRP-Online (2026 Paper)" --algo "GP-Online (Durasevic 2026 Paper)" --layout benchmark/Caserta_dataset/data3-3-1.dat

# Run single experiment with synthetic layout
python main.py run --problem "CRP-Online (2026 Paper)" --algo "GP-Online (Durasevic 2026 Paper)"
```

