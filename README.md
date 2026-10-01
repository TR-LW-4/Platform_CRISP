# Platform_CRISP

Research platform for container-yard relocation and stowage. Heuristics, exact methods, and experiments share one yard, one registry, and one result store.

## Run

```bash
pip install -r requirements.txt
python main.py list

# smoke-test all problems
python main.py test

# run one experiment
python main.py run --problem "CRP-R" --algo "Caserta (2012) HEUR"
python main.py web
```

## Problems

| Problem | What it is | Primary metric |
|---|---|---|
| **CRP-R** | Fixed retrieval order. Only the target stack may be moved. | relocations |
| **CRP-U** | Fixed retrieval order. Any stack top may be moved. | relocations |
| **CRP-D** | Ordered groups. Any container in the current group may be retrieved. | relocations |
| **CRP-Time** | Same moves as CRP-R. Minimise crane time. | crane_time |
| **CRP-MO** | Same moves as CRP-Time. Minimise relocations and crane time together. | both |
| **CRP-Online** | Retrieval order is revealed gradually. | relocations |
| **CRP-Stow** | Retrieval follows a vessel loading plan. | relocations |
| **CRP-Stoch** | Containers come in batches. Order inside a batch is unknown until the batch starts. | Relocations |
