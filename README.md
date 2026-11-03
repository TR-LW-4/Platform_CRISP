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
2. Select Algorithm: **`GP-Online (Durasevic 2026 Paper)`** (or any of the 8 manual heuristics)
3. Select Instance source: **`Caserta Benchmark`** or **`Ku & Arthanari Benchmark`**
4. Click **Start run** or **Start batch** to observe live convergence, relocations, and yard layouts.

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
Run the 37-scenario integration test suite verifying mathematical parity, information barrier constraints, and edge cases:
```bash
python -m pytest crp_online/tests/test_crisp_integration_scenarios.py -v
```

Run the full 840-instance baseline parity test matching Table 4 & Table 5 of the 2026 paper:
```bash
python -m pytest crp_online/tests/test_all_baselines_840.py -v
```

---

## 2026 Paper Features on this Branch

### Supported Algorithms
- **GP-Online 2026**: Relocation rules evolved via Genetic Programming evaluated using Polish prefix notation. Pre-evolved high-performance rules included by default.
- **8 Manual Baselines**:
  - `EM` (Expected Min-Max)
  - `ERI` (Expected Reshuffle Index)
  - `EGH` (Expected Greedy Heuristic)
  - `RDH` (Reshuffle Decrease Heuristic)
  - `MM` (Min-Max / Blocking Minimization)
  - `PR4` (Priority Rank 4)
  - `RI` (Reshuffle Index)
  - `TLP` (Total Leveling Policy / Lowest Position)

### Benchmarks
- **Caserta Benchmark**: All 840 standard benchmark instances ($H \in \{3, 4, 5, 6, 10\}$).
- **Ku & Arthanari Benchmark**: 24 scenarios (720 instances) with arrival/departure time windows ($S \in \{5\dots 10\}$, $T \in \{3\dots 6\}$).

### Verified Parity Highlights (Table 4, full 840 instances)
| Heuristic | Published 2026 Total | CRISP Engine Total | Parity Status |
| :--- | :---: | :---: | :---: |
| **TLP** | 36,024 | 36,024 | Exact (0 diff) |
| **RI** | 29,524 | 29,524 | Exact (0 diff) |
| **MM** | 25,859 | 25,859 | Exact (0 diff) |
| **PR4** | 25,787 | 25,787 | Exact (0 diff) |
| **ERI** | 29,524 | 29,524 | Exact (0 diff) |
| **EM** | 24,612 | 24,612 | Exact (0 diff) |
| **EGH** | 30,990 | 30,990 | Exact (0 diff) |
| **GP-Online** | 24,004 (median) | 24,017 (median) | < 0.05% diff |

---

## CRISP Platform Problem Families

| Problem | What it is | Primary metric |
|---|---|---|
| **CRP-Online (2026)** | Online CRP with look-ahead horizons & container groups (2026 paper). | relocations |
| **CRP-R** | Fixed retrieval order. Only the target stack may be moved. | relocations |
| **CRP-U** | Fixed retrieval order. Any stack top may be moved. | relocations |
| **CRP-D** | Ordered groups. Any container in the current group may be retrieved. | relocations |
| **CRP-Time** | Same moves as CRP-R. Minimise crane time. | crane_time |
| **CRP-MO** | Same moves as CRP-Time. Minimise relocations and crane time together. | both |
| **CRP-Online** | Retrieval order is revealed gradually. | relocations |
| **CRP-Stow** | Retrieval follows a vessel loading plan. | relocations |
| **CRP-Stoch** | Containers come in batches. Order inside a batch is unknown until the batch starts. | relocations |
