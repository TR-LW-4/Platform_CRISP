# Platform_CRISP — Online CRP (2026 Paper Extension)

Research platform for container-yard relocation and stowage. This branch integrates the isolated, verified implementation of the **Online Container Relocation Problem (Online CRP)** from:

> **"Automated Design of Relocation Rules with Genetic Programming for the Online Container Relocation Problem"**  
> Francisco Javier Gil Gala, Mateja Đumić, Marko Đurasević (*Expert Systems with Applications, 2026*).

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
pip install pytest   # for verification test suite
```

### 2. Launch the Research Web Workbench
No Node.js or npm build is required — precompiled production assets are served directly:
```bash
python main.py web
```
Open **http://127.0.0.1:8000** in your browser.
1. Select Problem: **`CRP-Online (2026 Paper)`**
2. Configure 2026 Paper Settings:
   - **Relocation Scheme**: `restricted` (Algorithm 1) or `unrestricted` (Algorithm 2 with preventive moves)
   - **Information Structure**: `rolling_horizon` or `container_groups`
   - **Lookahead Horizon ($\alpha$)**: `1.0` (offline), `0.9`, `0.7`, `0.5`, `0.3`, `0.1`, `0.0`
   - **Group Mode**: `normal` (unknown within group) or `relaxed` (known within group)
   - **Group Definition**: by `group_count` ($GC \in \{2, 3, 4, 5\}$) or `group_size` ($GS \in \{2, 3, 4, 5\}$)
3. Select Algorithm: **`GP-Online (Durasevic 2026 Paper)`** or any of the 10 published manual heuristics (`PU1`, `PU2`, `TLP`, `RI`, `Min-Max`, `PR4`, `RDH`, `ERI`, `EM`, `EGH`)
4. Select Instance Source: **`Caserta Benchmark`** (840 instances) or **`Ku & Arthanari Benchmark`** (720 instances)
5. Click **Start run** or **Start batch** to observe live convergence, relocations, and yard layouts.

### 3. Command-Line Execution
Run a single experiment:
```bash
python main.py run --problem "CRP-Online (2026 Paper)" --algo "GP-Online (Durasevic 2026 Paper)" --iterations 100
```

Run on a specific Caserta benchmark layout:
```bash
python main.py layout-run --problem "CRP-Online (2026 Paper)" --algo "GP-Online (Durasevic 2026 Paper)" --layout benchmark/Caserta_dataset/data3-4-1.dat
```

### 4. Verification & Parity Test Suite
Run the 46-scenario integration test suite verifying mathematical parity, information barrier constraints, and edge cases:
```bash
python -m pytest crp_online/tests/test_crisp_integration_scenarios.py -v
```

Run the full 840-instance baseline parity test matching Table 4 & Table 5 of the 2026 paper:
```bash
python -m pytest crp_online/tests/test_all_baselines_840.py -v
```

---

## 2026 Paper Features on this Branch

### Information Structures
1. **Rolling Lookahead Horizon**:
   - As containers are retrieved, the horizon slides: the last container in the considered horizon $[T, T + \alpha N]$ is revealed.
   - $\alpha = 1.0$: all containers are known from the beginning, reducing to the **offline problem**.
   - Unknown container strategies: `"maximum"`  and `"ignore"` (sentinel value).
2. **Container Groups**:
   - **Normal (unknown within group)**: Only the group priority order is known. Exact container IDs inside groups are masked with group proxies.
   - **Relaxed (known within group)**: Container numbers inside the active target group are revealed; subsequent groups have only group proxies.
   - Grouping definitions: **Group Count ($GC \in \{2, 3, 4, 5\}$)** or **Group Size ($GS \in \{2, 3, 4, 5\}$)**.

### Relocation Schemes
1. **Restricted RS (Algorithm 1)**: Relocates only the blocking container above the target.
2. **Unrestricted RS (Algorithm 2)**: Allows preventive relocations from candidate destination stack $D$ to stack $S_1$ if $SC > DC$ and $\min(S_1) > DC$, clearing $D$ before placement.

### Supported Algorithms (All 10 Baselines + GP)
- **GP-Online 2026**: Evolved relocation rules evaluated via Polish prefix notation.
- **Unrestricted Baselines (Zhu et al. 2012)**:
  - `PU1`: Unrestricted Min-Max with preventive relocations.
  - `PU2`: Unrestricted PR4 with preventive relocations.
- **Restricted Baselines**:
  - `EM`: Expected Min-Max
  - `ERI`: Expected Reshuffle Index
  - `EGH`: Expected Group Heuristic (Galle 2018)
  - `RDH`: Random Destination Heuristic
  - `MM`: Min-Max (Caserta 2012)
  - `PR4`: Priority Rank 4 (Jin 2015)
  - `RI`: Reshuffle Index (Murty 2005)
  - `TLP`: Total Leveling Policy / Lowest Position (Lee & Lee 2010 / Zehendner 2017)

### Benchmarks
- **Caserta Benchmark** (`benchmark/Caserta_dataset`): All 840 standard benchmark instances ($H \in \{3, 4, 5, 6, 10\}$).
- **Ku & Arthanari Benchmark** (`benchmark/ku_arthanari`): 24 scenarios (720 instances) with arrival/departure time windows ($S \in \{5\dots 10\}$, $T \in \{3\dots 6\}$).

### Verified Parity Highlights (Table 4, full 840 instances)
| Heuristic | Published 2026 Total | CRISP Engine Total | Parity Status |
| :--- | :---: | :---: | :---: |
| **TLP** | 36,024 | 36,024 | Exact (0 diff) |
| **RI** | 29,524 | 29,524 | Exact (0 diff) |
| **MM** | 25,859 | 25,859 | Exact (0 diff) |
| **PR4** | 25,787 | 25,787 | Exact (0 diff) |
| **RDH** | 26,066* | 26,066 | Exact (0 diff vs C++ oracle) |
| **ERI** | 29,524 | 29,524 | Exact (0 diff) |
| **EM** | 24,612 | 24,612 | Exact (0 diff) |
| **EGH** | 30,990 | 30,990 | Exact (0 diff) |
| **PU1** | 25,049 | 25,049 | Exact (0 diff) |
| **PU2** | 24,962 | 24,962 | Exact (0 diff) |
| **GP-Online** | 24,004 (median) | 24,017 (median) | < 0.05% diff |

*\*Preprint Table 4 has typo 26,866 for RDH; exact C++ code oracle yields 26,066.*
