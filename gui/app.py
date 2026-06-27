"""
CRP Platform – Streamlit GUI

Tabs
----
Test       – run a single algorithm on a single problem, watch in real-time
Experiment – batch comparison of multiple algorithms × problems × seeds
Compare    – load saved experiment results and plot comparison charts
About      – platform info and user guide

Run with:
    streamlit run gui/app.py
"""

from __future__ import annotations

import sys
import os
import time
import threading
import queue as stdlib_queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np

# ── Registry ──────────────────────────────────────────────────────── #
from core.registry import (
    list_problems, list_algorithms,
    get_problem_class, get_algorithm_class,
    get_problem_info, get_algorithm_info,
    compatible_algorithms,
)
from core.base_problem   import ProblemConfig
from core.base_algorithm import AlgorithmConfig, TrainingSession
from core.result_store   import save_run, list_saved_runs, load_runs_for_compare, delete_run
from core.benchmark_keys import LAYOUT_FILE_EXTRA_KEY
from core.layout_trace   import trace_layout
from core.caserta_benchmark import (
    CASERTA_HEIGHT_WHITELIST,
    collect_paths_from_hw_queue,
    index_ws_by_height,
    merge_hw_queue_item,
    parse_caserta_filename,
    problem_config_for_caserta_dat,
)
from core.zhu_benchmark import (
    ZHU_HEIGHT_WHITELIST,
    collect_paths_from_zhu_queue,
    index_sn_pairs_by_height,
    merge_zhu_queue_item,
    parse_zhu_folder_name,
    problem_config_for_zhu_txt,
)
from core.zhu_dup_benchmark import (
    ZHU_DUP_ALPHAS,
    ZHU_DUP_HEIGHT_WHITELIST,
    collect_paths_from_zhu_dup_queue,
    index_sn_pairs_by_height_dup,
    merge_zhu_dup_queue_item,
    parse_zhu_dup_folder_name,
    problem_config_for_zhu_dup_txt,
)

# ── Visualisation ─────────────────────────────────────────────────── #
from visualization.bay_renderer import yard_figure, vessel_figure, metrics_figure

# ================================================================ #
#  Page config                                                       #
# ================================================================ #

st.set_page_config(
    page_title  = "CRP Platform",
    page_icon   = "🚢",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────── #
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        border-left: 4px solid #4E79A7;
    }
    .metric-value { font-size: 1.4em; font-weight: bold; color: #4E79A7; }
    .metric-label { font-size: 0.8em; color: #666; }
    .algo-badge-RL   { background:#4E79A7; color:white; border-radius:4px; padding:2px 8px; }
    .algo-badge-Evolutionary { background:#59A14F; color:white; border-radius:4px; padding:2px 8px; }
    .algo-badge-Heuristic { background:#F28E2B; color:white; border-radius:4px; padding:2px 8px; }
</style>
""", unsafe_allow_html=True)

# ================================================================ #
#  Session state initialisation                                      #
# ================================================================ #

def _init_state():
    defaults = {
        "session":        None,    # TrainingSession
        "progress":       [],      # list of ProgressRecord
        "training":       False,
        "exp_results":    {},      # {(problem, algo, seed): metrics}
        "last_saved_path": None,   # path of most recent auto-saved result
        "train_start":     None,       # time.time() when training started
        "step_label":      "Iteration", # semantic label for one step
        "caserta_instance_queue": [],  # list[{"h": int, "ws": [int, ...]}] for Test tab
        "zhu_instance_queue": [],  # list[{"h": int, "sn_pairs": [[s,n], ...]}]
        "zhu_dup_instance_queue": [],  # list[{"h": int, "sn_pairs": [[s,n], ...]}] for ZhuDup
        "zhu_dup_alpha": ZHU_DUP_ALPHAS[0],  # selected alpha subdirectory
        "bench_pause_waiting": False,
        "bench_pause_payload": None,  # dict: paths, next_idx, run_zhu, run_zhu_dup, problem, algorithm, params…
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Caserta filename scan: height list + w values per height
@st.cache_data(show_spinner=False)
def _caserta_ws_map_cached(root_str: str) -> Tuple[List[int], Dict[int, List[int]]]:
    m = index_ws_by_height(Path(root_str))
    return sorted(m.keys()), m


@st.cache_data(show_spinner=False)
def _zhu_sn_map_cached(root_str: str) -> Tuple[List[int], Dict[int, List[Tuple[int, int]]]]:
    m = index_sn_pairs_by_height(Path(root_str))
    return sorted(m.keys()), m


@st.cache_data(show_spinner=False)
def _zhu_dup_sn_map_cached(
    root_str: str, alpha: str
) -> Tuple[List[int], Dict[int, List[Tuple[int, int]]]]:
    m = index_sn_pairs_by_height_dup(Path(root_str), alpha)
    return sorted(m.keys()), m


def _schema_defaults(prob_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Default problem parameter dict from ``config_schema()`` (no widgets)."""
    d: Dict[str, Any] = {}
    for pname, spec in prob_schema.items():
        t = spec.get("type", "int")
        if t == "int":
            d[pname] = int(spec["default"])
        elif t == "float":
            d[pname] = float(spec["default"])
        elif t == "bool":
            d[pname] = bool(spec["default"])
        else:
            d[pname] = spec["default"]
    return d


BENCH_RANDOM = "Random Layout (schema defaults)"
BENCH_CASERTA = "Caserta Benchmark (select H, pick w values, add to list)"
BENCH_ZHU = "Zhu Benchmark (sub-folder H-S-N: select H, pick sizes, add to list)"
BENCH_ZHU_DUP = "Zhu Dup Benchmark (dup_dataset: select alpha, H, S-N pairs)"

_BASE_ALGO_KEYS = frozenset({
    "max_iterations", "seed", "report_interval", "num_eval_seeds",
    "total_timesteps", "learning_rate", "gamma", "num_envs",
    "num_steps", "batch_size", "hidden_dim",
    "population_size", "crossover_rate", "mutation_rate",
    "tournament_size", "elite_count",
})


def _algorithm_config_from_gui_params(algo_params: Dict[str, Any]) -> AlgorithmConfig:
    base_kwargs = {k: v for k, v in algo_params.items() if k in _BASE_ALGO_KEYS}
    algo_cfg = AlgorithmConfig(**base_kwargs)
    for k, v in algo_params.items():
        algo_cfg.extra[k] = v
    return algo_cfg


def _benchmark_shape_messages(
    prob_cfg: ProblemConfig,
    fpath: Path,
    run_zhu_batch: bool,
    run_zhu_dup_batch: bool = False,
) -> Tuple[str, str]:
    """Return (stderr line, markdown block) describing layout size / shape."""
    nb, nr, mt = prob_cfg.num_bays, prob_cfg.num_rows, prob_cfg.max_tiers
    nc = prob_cfg.num_containers
    is_subfolder_batch = run_zhu_batch or run_zhu_dup_batch
    rel = fpath.parent.name + "/" + fpath.name if is_subfolder_batch else fpath.name
    meta_plain = ""
    meta_md = ""
    if run_zhu_dup_batch:
        # path: dup_dataset/alpha=X/H-S-N/file.txt
        z = parse_zhu_dup_folder_name(fpath.parent.name)
        alpha_tag = fpath.parent.parent.name
        if z:
            h_, s_, n_ = z
            meta_plain = f"{alpha_tag} H-S-N=({h_},{s_},{n_}) "
            meta_md = f"**{alpha_tag}**, Sub-folder **H-S-N** = ({h_}, {s_}, {n_}); "
    elif run_zhu_batch:
        z = parse_zhu_folder_name(fpath.parent.name)
        if z:
            h_, s_, n_ = z
            meta_plain = f"H-S-N folder=({h_},{s_},{n_}) "
            meta_md = f"Sub-folder **H-S-N** = ({h_}, {s_}, {n_}) (size label); "
    else:
        c = parse_caserta_filename(fpath)
        if c:
            h_, w_, id_ = c
            meta_plain = f"filename H,w,id=({h_},{w_},{id_}) "
            meta_md = f"Filename **H, w, id** = ({h_}, {w_}, {id_}); "
    stderr_line = (
        f"[CRP Platform] instance {rel}: {meta_plain}"
        f"bays×rows×tiers={nb}×{nr}×{mt}, containers={nc}"
    )
    md = (
        f"**File** `{rel}`\n\n"
        f"{meta_md}\n"
        f"- **Grid**: {nb} bays × {nr} rows, max height **max_tiers = {mt}**\n"
        f"- **Containers N** = **{nc}**\n"
    )
    return stderr_line, md


def _run_one_benchmark_file(
    *,
    fpath: Path,
    run_zhu_batch: bool,
    run_zhu_dup_batch: bool = False,
    selected_problem: str,
    selected_algo: str,
    algo_params: Dict[str, Any],
    prob_params: Dict[str, Any],
    prob_cls,
    algo_cls,
    algo_cat: str,
    batch_index: int,
    batch_total: int,
    progress_bar,
    status_txt,
    src_tag: str,
    show_completion_banner: bool = True,
    terminal_detail: bool = False,
) -> None:
    """Train on one layout file, append progress, save_run — mirrors batch loop body."""
    import multiprocessing as mp

    is_subfolder = run_zhu_batch or run_zhu_dup_batch
    rel = fpath.parent.name + "/" + fpath.name if is_subfolder else fpath.name
    status_txt.caption(f"Running {batch_index + 1}/{batch_total}: `{rel}` …")
    if terminal_detail:
        print(
            "\n[CRP Platform] ========== First instance only (manual inspection) ==========",
            file=sys.stderr,
            flush=True,
        )
        print(f"  path: {fpath.resolve()}", file=sys.stderr, flush=True)
        print(f"  rel:  {rel}", file=sys.stderr, flush=True)
    trace_layout(
        f"gui benchmark outer loop: instance {batch_index + 1}/{batch_total} "
        f"(each iteration builds **one** ProblemConfig for **one** layout file, "
        f"then starts **one** subprocess train(); list comes from collect_paths, "
        f"not stored — recomputed from your queue each run)"
    )
    trace_layout(f"gui benchmark current file: {fpath.resolve()}")
    if run_zhu_dup_batch:
        prob_cfg = problem_config_for_zhu_dup_txt(fpath, {})
    elif run_zhu_batch:
        prob_cfg = problem_config_for_zhu_txt(fpath, {})
    else:
        prob_cfg = problem_config_for_caserta_dat(fpath, {})

    stderr_line, md_shape = _benchmark_shape_messages(
        prob_cfg, fpath, run_zhu_batch, run_zhu_dup_batch
    )
    print(stderr_line, file=sys.stderr, flush=True)

    algo_cfg = _algorithm_config_from_gui_params(algo_params)
    algo_inst = algo_cls(config=algo_cfg)

    def _pf(_pc=prob_cfg):
        return prob_cls(config=_pc)

    JOIN_T = 900
    q = mp.Queue()
    ev = mp.Event()
    proc = mp.Process(
        target=algo_inst.train,
        args=(_pf, q, ev),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=JOIN_T)
    ev.set()
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)

    all_records = []
    while not q.empty():
        all_records.append(q.get_nowait())

    if all_records:
        final = all_records[-1]
        history = [
            {"step": r.step, "metric": r.metric, "metrics": r.metrics}
            for r in all_records
        ]
        prob_save = {
            **prob_params,
            LAYOUT_FILE_EXTRA_KEY: str(fpath),
            "source": src_tag,
        }
        save_run(
            problem=selected_problem,
            algorithm=selected_algo,
            category=algo_cat,
            seed=abs(hash(fpath.name)) % 2_000_000_000,
            prob_config=prob_save,
            algo_config=algo_params,
            metrics=final.metrics,
            history=history,
        )
        if terminal_detail:
            print(
                "[CRP Platform] ---------- Result (final step) ----------",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"  step={final.step}  metric={final.metric:.6f}  "
                f"best_metric={final.best_metric:.6f}  progress={final.progress:.4f}",
                file=sys.stderr,
                flush=True,
            )
            print(f"  metrics: {final.metrics}", file=sys.stderr, flush=True)
            print(
                "[CRP Platform] ---------- Done (remaining files skipped) ----------\n",
                file=sys.stderr,
                flush=True,
            )
    elif terminal_detail:
        print(
            "[CRP Platform] Warning: no training progress received (subprocess did not write to queue).",
            file=sys.stderr,
            flush=True,
        )

    progress_bar.progress((batch_index + 1) / batch_total)
    if show_completion_banner:
        st.success(
            f"✅ Completed **{batch_index + 1}/{batch_total}** — `{rel}`\n\n{md_shape}"
        )


# ================================================================ #
#  Sidebar                                                           #
# ================================================================ #

with st.sidebar:
    st.title("🚢 CRP Platform")
    st.caption("Container Relocation & Stowage Platform")
    st.markdown("---")

    problems   = list_problems()
    algo_info  = {a["name"]: a for a in get_algorithm_info()}

    if not problems:
        st.error("No problems found. Check `problems/` folder.")
        st.stop()
    if not algo_info:
        st.error("No algorithms found. Check `algorithms/` folder.")
        st.stop()

    # Stable key + drop stale value (e.g. after removing a problem from ``list_problems()``)
    _problem_pick_key = "sidebar_selected_problem"
    if _problem_pick_key in st.session_state and st.session_state[_problem_pick_key] not in problems:
        del st.session_state[_problem_pick_key]

    # ── Step 1: Problem ──────────────────────────────────────────── #
    st.markdown("**① Problem**")
    selected_problem = st.selectbox(
        "Select Problem",
        problems,
        label_visibility="collapsed",
        key=_problem_pick_key,
    )

    prob_tags = get_problem_class(selected_problem).tags if get_problem_class(selected_problem) else []
    if prob_tags:
        st.caption("Tags: " + " · ".join(prob_tags))

    st.markdown("---")

    # ── Step 2: Algorithm Category ───────────────────────────────── #
    st.markdown("**② Algorithm Category**")

    compat       = compatible_algorithms(selected_problem)
    avail_algos  = compat if compat else list(algo_info.keys())

    # Group by category
    CAT_COLOURS  = {
        "RL":           "#4E79A7",
        "Evolutionary": "#59A14F",
        "Heuristic":    "#F28E2B",
        "Exact":        "#9333EA",
    }
    cat_to_algos: Dict[str, List[str]] = {}
    for aname in avail_algos:
        cat = algo_info.get(aname, {}).get("category", "Other")
        cat_to_algos.setdefault(cat, []).append(aname)

    all_categories   = sorted(cat_to_algos.keys())
    category_labels  = {
        "RL":           "🤖 Reinforcement Learning",
        "Evolutionary": "🧬 Evolutionary Algorithm",
        "Heuristic":    "⚡ Heuristic",
        "Exact":        "✓ Exact / Optimal",
    }
    display_cats     = [category_labels.get(c, c) for c in all_categories]
    cat_idx          = st.radio(
        "Category",
        range(len(all_categories)),
        format_func=lambda i: display_cats[i],
        label_visibility="collapsed",
    )
    selected_category = all_categories[cat_idx]

    # Colour badge
    cat_colour = CAT_COLOURS.get(selected_category, "#999")
    st.markdown(
        f'<span style="background:{cat_colour};color:white;border-radius:4px;'
        f'padding:3px 10px;font-size:0.85em">{selected_category}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Solver filter ────────────────────────────────────────────── #
    show_solver_only = st.checkbox(
        "🔑 仅显示求解器算法 (Gurobi)",
        key="solver_filter",
        help="只显示需要 Gurobi 等外部商业/学术求解器才能运行的算法。"
             "请确保已安装 gurobipy 并激活有效 license。",
    )

    # ── Step 3: Specific Algorithm ───────────────────────────────── #
    st.markdown("**③ Algorithm**")
    algos_in_cat = cat_to_algos.get(selected_category, [])

    if show_solver_only:
        algos_in_cat = [
            a for a in algos_in_cat
            if algo_info.get(a, {}).get("requires_solver", False)
        ]

    if not algos_in_cat:
        if show_solver_only:
            st.info(f"当前类别 **{selected_category}** 下没有需要求解器的算法。")
        else:
            st.warning(f"No {selected_category} algorithms available for this problem.")
        selected_algo = list(algo_info.keys())[0]
    else:
        selected_algo = st.selectbox(
            "Select Algorithm",
            algos_in_cat,
            label_visibility="collapsed",
        )

    # Show algorithm description
    desc = algo_info.get(selected_algo, {}).get("description", "")
    if desc:
        st.caption(desc[:120] + ("…" if len(desc) > 120 else ""))

    # ── Solver requirement badge ──────────────────────────────────── #
    if algo_info.get(selected_algo, {}).get("requires_solver", False):
        backend = algo_info.get(selected_algo, {}).get("solver_backend") or "外部求解器"
        st.warning(
            f"⚠️ **此算法需要 {backend.upper()} license**  \n"
            "请确保已完成以下步骤：  \n"
            "① `pip install gurobipy`  \n"
            "② 激活学术 Named-User License（`grbgetkey <key>`）或 WLS License。  \n"
            "未安装或 license 无效时算法将报错退出。"
        )

# ================================================================ #
#  Tabs                                                              #
# ================================================================ #

tab_test, tab_exp, tab_compare, tab_about = st.tabs(
    ["🔬 Test", "📊 Experiment", "📈 Compare", "ℹ️ About"]
)

# ================================================================ #
#  TAB 1: Test                                                       #
# ================================================================ #

with tab_test:
    st.header(f"{selected_problem}  ×  {selected_algo}")

    prob_cls    = get_problem_class(selected_problem)
    prob_schema = prob_cls.config_schema() if prob_cls else {}
    algo_cls    = get_algorithm_class(selected_algo)
    algo_schema = algo_cls.config_schema() if algo_cls else {}
    _step_label = getattr(algo_cls, "step_label", "Iteration") if algo_cls else "Iteration"

    caserta_root = Path(__file__).resolve().parent.parent / "benchmark" / "Caserta_dataset"
    zhu_root     = Path(__file__).resolve().parent.parent / "benchmark" / "Zhu_dataset"
    dup_root     = Path(__file__).resolve().parent.parent / "benchmark" / "dup_dataset"

    # Problems sharing the Caserta / Zhu file format
    _BENCH_PROBLEMS     = {"CRP-R", "CRP-U"}
    # Problems supporting ZhuDup benchmark
    _BENCH_DUP_PROBLEMS = {"CRP-D"}

    ws_by_height: Dict[int, List[int]] = {}
    if caserta_root.is_dir():
        _, ws_by_height = _caserta_ws_map_cached(str(caserta_root))

    zhu_sn_by_height: Dict[int, List[Tuple[int, int]]] = {}
    if zhu_root.is_dir():
        _, zhu_sn_by_height = _zhu_sn_map_cached(str(zhu_root))

    height_options_caserta = [h for h in CASERTA_HEIGHT_WHITELIST if h in ws_by_height]
    height_options_zhu = [h for h in ZHU_HEIGHT_WHITELIST if h in zhu_sn_by_height]

    # ZhuDup alpha / index
    _dup_alpha = st.session_state.get("zhu_dup_alpha", ZHU_DUP_ALPHAS[0])
    _dup_sn_by_height: Dict[int, List[Tuple[int, int]]] = {}
    if dup_root.is_dir() and selected_problem in _BENCH_DUP_PROBLEMS:
        _, _dup_sn_by_height = _zhu_dup_sn_map_cached(str(dup_root), _dup_alpha)
    _height_options_dup = [
        h for h in ZHU_DUP_HEIGHT_WHITELIST if h in _dup_sn_by_height
    ]

    bench_opts: List[str] = [BENCH_RANDOM]
    if ws_by_height and selected_problem in _BENCH_PROBLEMS:
        bench_opts.append(BENCH_CASERTA)
    if height_options_zhu and selected_problem in _BENCH_PROBLEMS:
        bench_opts.append(BENCH_ZHU)
    if _height_options_dup and selected_problem in _BENCH_DUP_PROBLEMS:
        bench_opts.append(BENCH_ZHU_DUP)

    prob_params = _schema_defaults(prob_schema)

    col_left, col_right = st.columns([1, 2])

    bench_source = BENCH_RANDOM
    all_caserta_paths: List[Path] = []
    all_zhu_paths: List[Path] = []
    all_zhu_dup_paths: List[Path] = []

    # ── Right: benchmark + placeholders ───────────────────────────── #
    with col_right:
        st.subheader("Benchmark")
        if selected_problem in _BENCH_PROBLEMS and len(bench_opts) > 1:
            if not os.environ.get("CRISP_TRACE_LAYOUT"):
                st.caption(
                    "To enable layout trace logs in the terminal: run "
                    "`export CRISP_TRACE_LAYOUT=1` in the terminal where Streamlit is started, "
                    "then restart with `streamlit run gui/app.py`. "
                    "Logs appear on **stderr**, not in the browser."
                )
        _bench_eligible = (
            selected_problem in _BENCH_PROBLEMS
            or selected_problem in _BENCH_DUP_PROBLEMS
        )
        if _bench_eligible and len(bench_opts) > 1:
            bench_source = st.radio(
                "Instance Source",
                bench_opts,
                horizontal=False,
                key=f"bench_src_{selected_problem}",
            )
        elif not _bench_eligible:
            st.caption(
                "Caserta / Zhu benchmark is only supported for **CRP-R** and **CRP-U**; "
                "ZhuDup benchmark is only supported for **CRP-D**."
            )
        elif len(bench_opts) == 1:
            st.caption("No benchmark data found for this problem.")

        if selected_problem in _BENCH_PROBLEMS and bench_source == BENCH_CASERTA and ws_by_height:
            st.caption(
                "Filename format: `data[H]-[w]-[id].dat`. "
                "First **select a single height H** (whitelist {3,4,5,6,10} intersected with dataset), "
                "then **select one or more stack counts w** and click 'Add to Instance List'. "
                "**The id segment is not filtered** — all files for the chosen (H, w) pair are included."
            )
            if not height_options_caserta:
                st.warning("No files match standard heights {3,4,5,6,10} in the Caserta dataset.")
            else:
                cur_h = st.selectbox(
                    "Height H (single select)",
                    options=height_options_caserta,
                    format_func=lambda x: f"H = {x}",
                    key=f"cb_H_single_{selected_problem}",
                )
                w_opts = ws_by_height.get(cur_h, [])
                sel_ws_str = st.multiselect(
                    f"Stack count w (H={cur_h}, multi-select)",
                    options=[str(w) for w in w_opts],
                    format_func=lambda x: f"w = {x}",
                    key=f"cb_ws_pick_{selected_problem}_{cur_h}",
                )
                sel_ws = [int(x) for x in sel_ws_str]

                b_add, b_clr = st.columns(2)
                with b_add:
                    add_clicked = st.button("➕ Add to Instance List", key=f"add_hw_{selected_problem}")
                with b_clr:
                    clr_clicked = st.button("🗑 Clear Instance List", key=f"clr_hw_{selected_problem}")

                if clr_clicked:
                    st.session_state.caserta_instance_queue = []
                    st.rerun()
                if add_clicked:
                    if not sel_ws:
                        st.warning("Please select at least one stack count w.")
                    else:
                        merge_hw_queue_item(st.session_state.caserta_instance_queue, cur_h, sel_ws)
                        st.rerun()

            q_c = st.session_state.caserta_instance_queue
            if q_c:
                st.markdown("**Current Instance List (Caserta)**")
                for idx, block in enumerate(q_c):
                    st.write(
                        f"{idx + 1}. **H = {block['h']}**, "
                        f"w ∈ {{{', '.join(str(w) for w in block['ws'])}}}"
                    )

            all_caserta_paths = collect_paths_from_hw_queue(caserta_root, q_c)
            st.info(f"Caserta list contains **{len(all_caserta_paths)}** `.dat` files in total (will run sequentially).")

        elif selected_problem in _BENCH_PROBLEMS and bench_source == BENCH_ZHU and height_options_zhu:
            st.caption(
                "Sub-folder name format: **`H-S-N`** (H = height label, S = stacks, N = containers). "
                "First **select a single H** (Zhu whitelist **3,4,5,6,7,10** intersected with dataset), "
                "then **select one or more (S, N) sizes**. "
                "Each size folder typically contains **100** `.txt` instances — all are included when added."
            )
            cur_hz = st.selectbox(
                "Height H (single select)",
                options=height_options_zhu,
                format_func=lambda x: f"H = {x}",
                key=f"zhu_H_single_{selected_problem}",
            )
            sn_list = zhu_sn_by_height.get(cur_hz, [])
            sn_labels = [f"{s}-{n}" for s, n in sn_list]
            sel_sn_str = st.multiselect(
                f"Scale S-N (H={cur_hz}, S=stacks, N=containers; multi-select)",
                options=sn_labels,
                format_func=lambda lab: f"S={lab.split('-')[0]}, N={lab.split('-')[1]}",
                key=f"zhu_sn_pick_{selected_problem}_{cur_hz}",
            )
            sel_pairs: List[Tuple[int, int]] = []
            for lab in sel_sn_str:
                a, _, b = lab.partition("-")
                if a and b:
                    sel_pairs.append((int(a), int(b)))

            zb_add, zb_clr = st.columns(2)
            with zb_add:
                zhu_add = st.button("➕ Add to Instance List", key=f"add_zhu_hw_{selected_problem}")
            with zb_clr:
                zhu_clr = st.button("🗑 Clear Instance List", key=f"clr_zhu_hw_{selected_problem}")

            if zhu_clr:
                st.session_state.zhu_instance_queue = []
                st.rerun()
            if zhu_add:
                if not sel_pairs:
                    st.warning("Please select at least one scale (S, N).")
                else:
                    merge_zhu_queue_item(st.session_state.zhu_instance_queue, cur_hz, sel_pairs)
                    st.rerun()

            q_z = st.session_state.zhu_instance_queue
            if q_z:
                st.markdown("**Current Instance List (Zhu)**")
                for idx, block in enumerate(q_z):
                    pairs_s = ", ".join(f"({p[0]},{p[1]})" for p in block["sn_pairs"])
                    st.write(f"{idx + 1}. **H = {block['h']}**: {pairs_s}")

            all_zhu_paths = collect_paths_from_zhu_queue(zhu_root, q_z)
            st.info(
                f"Zhu list contains **{len(all_zhu_paths)}** `.txt` files in total "
                f"(will run sequentially; all instances in each selected `H-S-N` folder are included)."
            )

        elif (
            selected_problem in _BENCH_DUP_PROBLEMS
            and bench_source == BENCH_ZHU_DUP
            and _height_options_dup
        ):
            st.caption(
                "Directory layout: `dup_dataset/{alpha}/{H-S-N}/{file}.txt`. "
                "**Step 1**: select an alpha value (duplicate ratio). "
                "**Step 2**: select height H, then one or more (S, N) size pairs. "
                "Each `H-S-N` folder contains **100** instances."
            )

            # Alpha selector
            alpha_choice = st.selectbox(
                "Duplicate ratio α",
                options=list(ZHU_DUP_ALPHAS),
                index=list(ZHU_DUP_ALPHAS).index(_dup_alpha),
                format_func=lambda x: x.replace("alpha=", "α = "),
                key=f"dup_alpha_{selected_problem}",
            )
            if alpha_choice != _dup_alpha:
                st.session_state.zhu_dup_alpha = alpha_choice
                st.session_state.zhu_dup_instance_queue = []
                st.rerun()

            # Re-load index for chosen alpha (may differ from cached _dup_alpha)
            _, _dup_sn_cur = _zhu_dup_sn_map_cached(str(dup_root), alpha_choice)
            _h_opts_cur = [h for h in ZHU_DUP_HEIGHT_WHITELIST if h in _dup_sn_cur]

            cur_hd = st.selectbox(
                "Height H (single select)",
                options=_h_opts_cur,
                format_func=lambda x: f"H = {x}",
                key=f"dup_H_single_{selected_problem}",
            )
            sn_list_d = _dup_sn_cur.get(cur_hd, [])
            sn_labels_d = [f"{s}-{n}" for s, n in sn_list_d]
            sel_sn_d_str = st.multiselect(
                f"Scale S-N (H={cur_hd}, S=stacks, N=containers; multi-select)",
                options=sn_labels_d,
                format_func=lambda lab: f"S={lab.split('-')[0]}, N={lab.split('-')[1]}",
                key=f"dup_sn_pick_{selected_problem}_{cur_hd}",
            )
            sel_pairs_d: List[Tuple[int, int]] = []
            for lab in sel_sn_d_str:
                a, _, b = lab.partition("-")
                if a and b:
                    sel_pairs_d.append((int(a), int(b)))

            db_add, db_clr = st.columns(2)
            with db_add:
                dup_add = st.button(
                    "➕ Add to Instance List", key=f"add_dup_hw_{selected_problem}"
                )
            with db_clr:
                dup_clr = st.button(
                    "🗑 Clear Instance List", key=f"clr_dup_hw_{selected_problem}"
                )

            if dup_clr:
                st.session_state.zhu_dup_instance_queue = []
                st.rerun()
            if dup_add:
                if not sel_pairs_d:
                    st.warning("Please select at least one scale (S, N).")
                else:
                    merge_zhu_dup_queue_item(
                        st.session_state.zhu_dup_instance_queue, cur_hd, sel_pairs_d
                    )
                    st.rerun()

            q_d = st.session_state.zhu_dup_instance_queue
            if q_d:
                st.markdown("**Current Instance List (ZhuDup)**")
                for idx_d, block_d in enumerate(q_d):
                    pairs_s_d = ", ".join(
                        f"({p[0]},{p[1]})" for p in block_d["sn_pairs"]
                    )
                    st.write(f"{idx_d + 1}. **H = {block_d['h']}**: {pairs_s_d}")

            all_zhu_dup_paths = collect_paths_from_zhu_dup_queue(
                dup_root, q_d, alpha_choice
            )
            st.info(
                f"ZhuDup list contains **{len(all_zhu_dup_paths)}** `.txt` files "
                f"(α = {alpha_choice.replace('alpha=', '')}, "
                "all instances in each selected `H-S-N` folder are included)."
            )

        st.markdown("---")
        st.subheader("Yard Visualisation")
        status_placeholder  = st.empty()
        yard_placeholder    = st.empty()
        metric_placeholder  = st.empty()
        chart_placeholder   = st.empty()

    # ── Left: algorithm only (Problem Config form hidden) ───────────── #
    with col_left:
        st.caption("Problem parameters use the **config_schema defaults** (form not shown).")
        st.subheader("Algorithm Config")

        algo_params: Dict[str, Any] = {}
        for aname, spec in algo_schema.items():
            label = spec.get("label", aname)
            help_ = spec.get("help", "")
            t     = spec["type"]
            if t == "int":
                algo_params[aname] = st.number_input(
                    label, value=spec["default"],
                    min_value=spec.get("min"), max_value=spec.get("max"),
                    step=1, key=f"a_{aname}", help=help_,
                )
            elif t == "float":
                algo_params[aname] = st.number_input(
                    label, value=float(spec["default"]),
                    min_value=float(spec.get("min", 0)),
                    max_value=float(spec.get("max", 1e6)),
                    format="%.5f", key=f"a_{aname}", help=help_,
                )
            elif t == "bool":
                algo_params[aname] = st.checkbox(
                    label, value=spec["default"], key=f"a_{aname}", help=help_,
                )
            elif t == "str":
                options = spec.get("options", [spec["default"]])
                default = spec.get("default", options[0])
                idx = options.index(default) if default in options else 0
                algo_params[aname] = st.selectbox(
                    label, options=options, index=idx,
                    key=f"a_{aname}", help=help_,
                )

        show_pause_opt = (
            (selected_problem in _BENCH_PROBLEMS
             and (len(all_caserta_paths) > 0 or len(all_zhu_paths) > 0))
            or (selected_problem in _BENCH_DUP_PROBLEMS
                and len(all_zhu_dup_paths) > 0)
        )
        first_instance_only = False
        pause_each_file = False
        if show_pause_opt:
            first_instance_only = st.checkbox(
                "Run **first instance only**: print size info and result to **terminal (stderr)** then stop",
                key=f"bench_first_only_{selected_problem}",
                help="Useful for manually inspecting the first instance. Overrides the 'pause after each file' option.",
            )
            pause_each_file = st.checkbox(
                "Pause after each layout file and display instance info (filename / H-S-N, grid, height, containers)",
                key=f"bench_pause_each_{selected_problem}",
                help="Only applies when running a Caserta/Zhu batch via Start. Overridden by 'first instance only'.",
                disabled=first_instance_only,
            )

        # ── Control buttons ───────────────────────────────────────── #
        btn_col1, btn_col2 = st.columns(2)
        start_btn = btn_col1.button("▶ Start", type="primary",  use_container_width=True)
        stop_btn  = btn_col2.button("⏹ Stop",  type="secondary", use_container_width=True)

    bench_pause_cont = False
    bench_pause_cancel = False
    if st.session_state.get("bench_pause_waiting"):
        pay_b = st.session_state.get("bench_pause_payload") or {}
        paths_bn = len(pay_b.get("paths", []))
        next_i = int(pay_b.get("next_idx", 0))
        done_n = min(next_i, paths_bn)
        st.info(
            f"**Step-by-step batch**: completed **{done_n} / {paths_bn}** layout files; "
            f"next up is item **{next_i + 1}** in the list (index {next_i})."
        )
        bp1, bp2 = st.columns(2)
        with bp1:
            bench_pause_cont = st.button(
                "▶ Continue to Next Layout File", key="bench_pause_continue", type="primary"
            )
        with bp2:
            bench_pause_cancel = st.button("Cancel Step-by-Step Batch", key="bench_pause_cancel")

    # ── Button handlers ───────────────────────────────────────────── #
    prob_cfg = None   # defined below when Start is clicked

    if bench_pause_cancel:
        st.session_state.bench_pause_waiting = False
        st.session_state.bench_pause_payload = None
        st.warning("Step-by-step batch cancelled (instance queue preserved; click Start to restart).")
        st.rerun()

    if bench_pause_cont and st.session_state.get("bench_pause_payload"):
        pay = st.session_state["bench_pause_payload"]
        paths_bp = [Path(p) for p in pay["paths"]]
        idx = int(pay["next_idx"])
        if idx >= len(paths_bp):
            st.session_state.bench_pause_waiting = False
            st.session_state.bench_pause_payload = None
            st.rerun()
        sp = pay["problem"]
        sa = pay["algorithm"]
        prob_cls_p = get_problem_class(sp)
        algo_cls_p = get_algorithm_class(sa)
        algo_params_p: Dict[str, Any] = pay["algo_params"]
        prob_params_p: Dict[str, Any] = pay["prob_params"]
        run_zhu_p     = bool(pay.get("run_zhu", False))
        run_zhu_dup_p = bool(pay.get("run_zhu_dup", False))
        algo_cat_p = algo_info.get(sa, {}).get("category", "")
        if run_zhu_dup_p:
            src_tag_p = "zhu_dup_batch"
            batch_label_p = "ZhuDup"
        elif run_zhu_p:
            src_tag_p = "zhu_batch"
            batch_label_p = "Zhu"
        else:
            src_tag_p = "caserta_batch"
            batch_label_p = "Caserta"
        progress_bar_p = st.progress(idx / max(len(paths_bp), 1))
        status_txt_p = st.empty()
        _run_one_benchmark_file(
            fpath=paths_bp[idx],
            run_zhu_batch=run_zhu_p,
            run_zhu_dup_batch=run_zhu_dup_p,
            selected_problem=sp,
            selected_algo=sa,
            algo_params=algo_params_p,
            prob_params=prob_params_p,
            prob_cls=prob_cls_p,
            algo_cls=algo_cls_p,
            algo_cat=algo_cat_p,
            batch_index=idx,
            batch_total=len(paths_bp),
            progress_bar=progress_bar_p,
            status_txt=status_txt_p,
            src_tag=src_tag_p,
            show_completion_banner=True,
        )
        pay["next_idx"] = idx + 1
        if pay["next_idx"] >= len(paths_bp):
            st.session_state.bench_pause_waiting = False
            st.session_state.bench_pause_payload = None
            status_txt_p.empty()
            progress_bar_p.empty()
            st.success(
                f"✅ {batch_label_p} batch complete: **{len(paths_bp)}** instances processed; results saved to `results/`."
            )
        else:
            st.session_state.bench_pause_waiting = True
            st.session_state.bench_pause_payload = pay
        st.rerun()

    if start_btn:
        if st.session_state.session and st.session_state.training:
            st.session_state.session.stop()

        run_caserta_batch = (
            bench_source == BENCH_CASERTA
            and selected_problem in _BENCH_PROBLEMS
            and len(all_caserta_paths) > 0
        )
        run_zhu_batch = (
            bench_source == BENCH_ZHU
            and selected_problem in _BENCH_PROBLEMS
            and len(all_zhu_paths) > 0
        )
        run_zhu_dup_batch = (
            bench_source == BENCH_ZHU_DUP
            and selected_problem in _BENCH_DUP_PROBLEMS
            and len(all_zhu_dup_paths) > 0
        )

        if bench_source == BENCH_CASERTA and selected_problem not in _BENCH_PROBLEMS:
            st.error("Caserta benchmark requires **CRP-R** or **CRP-U**.")
            st.stop()
        if bench_source == BENCH_ZHU and selected_problem not in _BENCH_PROBLEMS:
            st.error("Zhu benchmark requires **CRP-R** or **CRP-U**.")
            st.stop()
        if bench_source == BENCH_ZHU_DUP and selected_problem not in _BENCH_DUP_PROBLEMS:
            st.error("ZhuDup benchmark requires **CRP-D**.")
            st.stop()
        if bench_source == BENCH_CASERTA and selected_problem in _BENCH_PROBLEMS and not all_caserta_paths:
            st.error("Please add instances first: **select height H + select stack count w** (click 'Add to Instance List').")
            st.stop()
        if bench_source == BENCH_ZHU and selected_problem in _BENCH_PROBLEMS and not all_zhu_paths:
            st.error(
                "Please add instances first: **select height H + select scale (S, N)** (click 'Add to Instance List')."
            )
            st.stop()
        if bench_source == BENCH_ZHU_DUP and selected_problem in _BENCH_DUP_PROBLEMS and not all_zhu_dup_paths:
            st.error(
                "Please add instances first: **select α, height H + select scale (S, N)** (click 'Add to Instance List')."
            )
            st.stop()

        algo_cfg = _algorithm_config_from_gui_params(algo_params)

        if run_caserta_batch or run_zhu_batch or run_zhu_dup_batch:
            if not pause_each_file:
                st.session_state.bench_pause_waiting = False
                st.session_state.bench_pause_payload = None

            if run_zhu_dup_batch:
                batch_paths = all_zhu_dup_paths
                src_tag = "zhu_dup_batch"
                batch_label = "ZhuDup"
            elif run_caserta_batch:
                batch_paths = all_caserta_paths
                src_tag = "caserta_batch"
                batch_label = "Caserta"
            else:
                batch_paths = all_zhu_paths
                src_tag = "zhu_batch"
                batch_label = "Zhu"

            algo_cat = algo_info.get(selected_algo, {}).get("category", "")

            if first_instance_only:
                st.session_state.bench_pause_waiting = False
                st.session_state.bench_pause_payload = None
                first_path = batch_paths[0]
                print(
                    "\n[CRP Platform] mode=first_instance_only — "
                    f"queue has {len(batch_paths)} file(s); running only the first, skipping the rest.\n",
                    file=sys.stderr,
                    flush=True,
                )
                progress_bar_f = st.progress(0)
                status_txt_f = st.empty()
                _run_one_benchmark_file(
                    fpath=first_path,
                    run_zhu_batch=run_zhu_batch,
                    run_zhu_dup_batch=run_zhu_dup_batch,
                    selected_problem=selected_problem,
                    selected_algo=selected_algo,
                    algo_params=algo_params,
                    prob_params=prob_params,
                    prob_cls=prob_cls,
                    algo_cls=algo_cls,
                    algo_cat=algo_cat,
                    batch_index=0,
                    batch_total=1,
                    progress_bar=progress_bar_f,
                    status_txt=status_txt_f,
                    src_tag=src_tag,
                    show_completion_banner=False,
                    terminal_detail=True,
                )
                status_txt_f.empty()
                progress_bar_f.empty()
                st.info(
                    "Results printed to the **terminal** (stderr): instance path, size info, final metrics. "
                    "The remaining instances in the queue were skipped."
                )
                st.rerun()

            print(
                f"[CRP Platform] benchmark batch started: {len(batch_paths)} instance(s). "
                "Per-file layout trace (problem_config / reset / yard load) requires starting "
                "the app with: CRISP_TRACE_LAYOUT=1 streamlit run gui/app.py — watch this terminal.",
                file=sys.stderr,
                flush=True,
            )

            if pause_each_file:
                pay_new: Dict[str, Any] = {
                    "paths": [str(p) for p in batch_paths],
                    "next_idx": 0,
                    "run_zhu": run_zhu_batch,
                    "run_zhu_dup": run_zhu_dup_batch,
                    "problem": selected_problem,
                    "algorithm": selected_algo,
                    "algo_params": dict(algo_params),
                    "prob_params": dict(prob_params),
                }
                progress_bar_s = st.progress(0)
                status_txt_s = st.empty()
                _run_one_benchmark_file(
                    fpath=batch_paths[0],
                    run_zhu_batch=run_zhu_batch,
                    run_zhu_dup_batch=run_zhu_dup_batch,
                    selected_problem=selected_problem,
                    selected_algo=selected_algo,
                    algo_params=algo_params,
                    prob_params=prob_params,
                    prob_cls=prob_cls,
                    algo_cls=algo_cls,
                    algo_cat=algo_cat,
                    batch_index=0,
                    batch_total=len(batch_paths),
                    progress_bar=progress_bar_s,
                    status_txt=status_txt_s,
                    src_tag=src_tag,
                    show_completion_banner=True,
                )
                pay_new["next_idx"] = 1
                if len(batch_paths) <= 1:
                    st.session_state.bench_pause_waiting = False
                    st.session_state.bench_pause_payload = None
                    status_txt_s.empty()
                    progress_bar_s.empty()
                else:
                    st.session_state.bench_pause_waiting = True
                    st.session_state.bench_pause_payload = pay_new
                st.rerun()

            else:
                progress_bar = st.progress(0)
                status_txt = st.empty()
                for i, fpath in enumerate(batch_paths):
                    _run_one_benchmark_file(
                        fpath=fpath,
                        run_zhu_batch=run_zhu_batch,
                        run_zhu_dup_batch=run_zhu_dup_batch,
                        selected_problem=selected_problem,
                        selected_algo=selected_algo,
                        algo_params=algo_params,
                        prob_params=prob_params,
                        prob_cls=prob_cls,
                        algo_cls=algo_cls,
                        algo_cat=algo_cat,
                        batch_index=i,
                        batch_total=len(batch_paths),
                        progress_bar=progress_bar,
                        status_txt=status_txt,
                        src_tag=src_tag,
                        show_completion_banner=False,
                    )

                status_txt.empty()
                progress_bar.empty()
                st.success(
                    f"✅ {batch_label} batch complete: **{len(batch_paths)}** instances processed; results saved to `results/`."
                )
                st.rerun()

        # ── Random layout: single live TrainingSession ───────────────── #
        prob_cfg = ProblemConfig(**{
            k: v for k, v in prob_params.items()
            if hasattr(ProblemConfig, k) or k in ("num_bays", "num_rows", "max_tiers",
                "num_containers", "num_groups", "num_cranes", "crane_speed",
                "vessel_bays", "vessel_rows", "vessel_tiers",
                "enable_weight", "enable_size", "enable_type", "seed")
        })
        for k in prob_params:
            try:
                getattr(prob_cfg, k)
            except AttributeError:
                prob_cfg.extra[k] = prob_params[k]

        def problem_factory(_cfg=prob_cfg, _cls=prob_cls):
            return _cls(config=_cfg)

        algo_inst = algo_cls(config=algo_cfg)
        session = TrainingSession(algo_inst, problem_factory)

        st.session_state.test_prob_effective = dict(prob_params)

        session.start()

        st.session_state.session = session
        st.session_state.progress = []
        st.session_state.training = True
        st.session_state.train_start = time.time()
        st.session_state.step_label = _step_label
        st.rerun()

    if stop_btn and st.session_state.session:
        st.session_state.session.stop()
        st.session_state.training = False

    # ── Live update loop ──────────────────────────────────────────── #
    if st.session_state.training and st.session_state.session:
        session = st.session_state.session
        records = session.drain()
        if records:
            st.session_state.progress.extend(records)

        progress_list = st.session_state.progress
        if progress_list:
            latest = progress_list[-1]

            # ── Step counter + elapsed time + progress bar ────────── #
            slabel      = st.session_state.get("step_label", "Iteration")
            # Infer max from the schema's first "iterations/generations/episodes/seeds" key
            _max_keys   = ("num_episodes", "num_updates", "max_generations",
                           "num_eval_seeds", "max_iterations")
            max_iter    = next(
                (int(algo_params[k]) for k in _max_keys if k in algo_params), 500
            )
            cur_step    = latest.step
            pct         = min(latest.progress, 1.0)
            elapsed_sec = time.time() - (st.session_state.get("train_start") or time.time())
            elapsed_str = f"{int(elapsed_sec // 60)}m {int(elapsed_sec % 60)}s"
            eta_sec     = (elapsed_sec / pct - elapsed_sec) if pct > 0.02 else 0
            eta_str     = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"

            with status_placeholder.container():
                pcols = st.columns(4)
                pcols[0].metric(slabel,     f"{cur_step} / {max_iter}")
                pcols[1].metric("Progress", f"{pct*100:.1f}%")
                pcols[2].metric("Elapsed",  elapsed_str)
                pcols[3].metric("ETA",      eta_str if pct < 0.99 else "✅ Done")
                st.progress(pct)

            # Metric cards
            m = latest.metrics
            cols = metric_placeholder.columns(min(4, len(m)))
            for i, (key, val) in enumerate(list(m.items())[:4]):
                cols[i].metric(key.replace("_", " ").title(), f"{val:.3f}")

            # Convergence chart
            metric_history = [r.metric for r in progress_list]
            chart_fig = metrics_figure(
                metric_history,
                label=prob_cls.metric_names[0] if prob_cls else "metric",
                title=f"{selected_algo} – Training Progress",
            )
            chart_placeholder.pyplot(chart_fig)

            # Yard visualisation
            snap = latest.yard_snapshot
            if snap and "yard" in snap:
                yard_snap  = snap["yard"]
                _eff = st.session_state.get("test_prob_effective") or prob_params
                n_bays  = int(_eff.get("num_bays",  4))
                n_rows  = int(_eff.get("num_rows",  2))
                n_tiers = int(_eff.get("max_tiers", 3))
                yard_fig = yard_figure(
                    yard_snap, n_bays, n_rows, n_tiers,
                    title=f"Step {latest.step} | Best: {latest.best_metric:.2f}",
                )
                yard_placeholder.pyplot(yard_fig)

        if not session.is_running():
            st.session_state.training = False

            # ── Auto-save result ─────────────────────────────────── #
            if progress_list:
                final   = progress_list[-1]
                history = [
                    {"step": r.step, "metric": r.metric, "metrics": r.metrics}
                    for r in progress_list
                ]
                algo_cat = algo_info.get(selected_algo, {}).get("category", "")
                saved_path = save_run(
                    problem     = selected_problem,
                    algorithm   = selected_algo,
                    category    = algo_cat,
                    seed        = int(prob_params.get("seed", 0)),
                    prob_config = prob_params,
                    algo_config = algo_params,
                    metrics     = final.metrics,
                    history     = history,
                )
                st.session_state.last_saved_path = str(saved_path)
                st.success(f"✅ Training complete!  Results saved → `{saved_path.name}`")
            else:
                st.success("✅ Training complete!")
        else:
            time.sleep(1.0)
            st.rerun()

# ================================================================ #
#  TAB 2: Experiment                                                 #
# ================================================================ #

with tab_exp:
    st.header("Batch Experiment")
    st.caption("Run multiple algorithms on multiple problems across seeds and compare results.")

    ecol1, ecol2 = st.columns(2)
    with ecol1:
        exp_problems  = st.multiselect("Problems",  list_problems(),  default=list_problems()[:2])
        exp_algos     = st.multiselect("Algorithms", list_algorithms(), default=list_algorithms()[:2])
    with ecol2:
        exp_seeds    = st.slider("Seeds", min_value=1, max_value=20, value=3)
        exp_iters    = st.number_input("Max iterations / seed", value=100, min_value=10, step=10)

    if st.button("🚀 Run Experiment", type="primary"):
        import multiprocessing as mp
        results      = {}
        saved_files  = []
        progress_bar = st.progress(0)
        status_text  = st.empty()
        total = len(exp_problems) * len(exp_algos) * exp_seeds
        done  = 0

        for pname in exp_problems:
            for aname in exp_algos:
                for seed in range(exp_seeds):
                    pcls = get_problem_class(pname)
                    acls = get_algorithm_class(aname)
                    if pcls is None or acls is None:
                        continue
                    status_text.caption(f"Running {aname} on {pname} (seed={seed})…")
                    cfg_p = ProblemConfig(seed=seed)
                    cfg_a = AlgorithmConfig(
                        max_iterations=int(exp_iters), seed=seed,
                        report_interval=max(1, int(exp_iters) // 20),
                    )
                    algo = acls(cfg_a)
                    q    = mp.Queue()
                    ev   = mp.Event()

                    def _factory(_p=pcls, _c=cfg_p):
                        return _p(config=_c)

                    proc = mp.Process(target=algo.train, args=(_factory, q, ev), daemon=True)
                    proc.start()
                    proc.join(timeout=120)
                    ev.set()

                    all_records = []
                    while not q.empty():
                        all_records.append(q.get_nowait())

                    if all_records:
                        final = all_records[-1]
                        results[(pname, aname, seed)] = final.metrics
                        # ── Auto-save each run ────────────────────── #
                        history = [
                            {"step": r.step, "metric": r.metric, "metrics": r.metrics}
                            for r in all_records
                        ]
                        cat = algo_info.get(aname, {}).get("category", "")
                        sp  = save_run(
                            problem=pname, algorithm=aname, category=cat,
                            seed=seed,
                            prob_config=cfg_p.to_dict(),
                            algo_config=cfg_a.to_dict(),
                            metrics=final.metrics,
                            history=history,
                        )
                        saved_files.append(sp.name)

                    done += 1
                    progress_bar.progress(done / total)

        st.session_state.exp_results = results
        status_text.empty()
        st.success(f"Done! {len(results)} runs completed. {len(saved_files)} files saved to `results/`.")

    if st.session_state.exp_results:
        import pandas as pd
        rows = []
        for (pname, aname, seed), metrics in st.session_state.exp_results.items():
            row = {"Problem": pname, "Algorithm": aname, "Seed": seed}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

# ================================================================ #
#  TAB 3: Compare                                                    #
# ================================================================ #

with tab_compare:
    st.header("📈 Algorithm Comparison")
    st.caption("Load saved results from any previous run and overlay them on the same chart.")

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load saved runs from disk ─────────────────────────────────── #
    saved_runs = list_saved_runs()

    if not saved_runs:
        st.info("No saved results yet. Run Test or Experiment first – results are saved automatically.")
    else:
        # Build a display label for each saved run
        run_labels = [
            f"{r['problem']}  ×  {r['algorithm']}  [seed={r['seed']}]  {r['timestamp'][:16]}"
            for r in saved_runs
        ]
        selected_labels = st.multiselect(
            "Select runs to compare (pick 2 or more):",
            options=run_labels,
            default=run_labels[:min(4, len(run_labels))],
        )

        if not selected_labels:
            st.info("Select at least one run above.")
        else:
            selected_files = [
                saved_runs[run_labels.index(lbl)]["file"]
                for lbl in selected_labels
            ]
            loaded = load_runs_for_compare(selected_files)

            # ── Final metrics table ───────────────────────────────── #
            st.subheader("Final Metrics")
            rows = []
            for r in loaded:
                row = {
                    "Problem":   r["problem"],
                    "Algorithm": r["algorithm"],
                    "Seed":      r["seed"],
                }
                row.update(r.get("metrics", {}))
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # ── Bar chart: final metric per algorithm ─────────────── #
            metric_cols = [c for c in df.columns if c not in ("Problem", "Algorithm", "Seed")]
            if metric_cols:
                chosen_metric = st.selectbox("Metric for bar chart:", metric_cols)
                agg = (
                    df.groupby(["Problem", "Algorithm"])[chosen_metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                problems_u = agg["Problem"].unique()
                colours    = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
                fig_bar, axes = plt.subplots(
                    1, len(problems_u),
                    figsize=(5 * len(problems_u), 4),
                    squeeze=False,
                )
                for pi, pname in enumerate(problems_u):
                    ax  = axes[0][pi]
                    sub = agg[agg["Problem"] == pname]
                    ax.bar(
                        sub["Algorithm"], sub["mean"],
                        yerr=sub["std"], color=colours[:len(sub)], capsize=4,
                    )
                    ax.set_title(pname, fontsize=10)
                    ax.set_ylabel(chosen_metric)
                    ax.tick_params(axis="x", rotation=20)
                    ax.grid(axis="y", alpha=0.3)
                fig_bar.suptitle(f"Mean {chosen_metric} (lower is better)", fontsize=11)
                fig_bar.tight_layout()
                st.pyplot(fig_bar)

            # ── Convergence curves overlay ────────────────────────── #
            st.subheader("Convergence Curves")
            curve_metric = st.selectbox("Metric for curves:", metric_cols, key="curve_metric") if metric_cols else None
            if curve_metric:
                fig_curve, ax_c = plt.subplots(figsize=(9, 4))
                colours = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
                for i, r in enumerate(loaded):
                    history = r.get("history", [])
                    if not history:
                        continue
                    steps   = [h["step"] for h in history]
                    vals    = [h["metrics"].get(curve_metric, h["metric"]) for h in history]
                    label   = f"{r['algorithm']} / {r['problem']} s{r['seed']}"
                    ax_c.plot(steps, vals, label=label,
                              color=colours[i % len(colours)], linewidth=1.8)
                ax_c.set_xlabel("Step")
                ax_c.set_ylabel(curve_metric)
                ax_c.set_title(f"Convergence: {curve_metric}")
                ax_c.legend(fontsize=8, loc="upper right")
                ax_c.grid(True, alpha=0.3)
                fig_curve.tight_layout()
                st.pyplot(fig_curve)

            # ── Delete runs ───────────────────────────────────────── #
            with st.expander("🗑 Delete selected runs"):
                if st.button("Delete selected runs", type="secondary"):
                    for fp in selected_files:
                        delete_run(fp)
                    st.success("Deleted. Refresh the page.")
                    st.rerun()

# ================================================================ #
#  TAB 4: About                                                      #
# ================================================================ #

with tab_about:
    st.header("CRP Platform")
    st.markdown("""
    **Container Relocation & Stowage Platform**

    A research platform for solving container yard management problems,
    inspired by PlatEMO but designed for logistics optimisation.

    ---

    ### Problems
    | Name | Description |
    |---|---|
    | **CRP-R** | Block Relocation Problem – fixed retrieval order |
    | **CRP-Prem** | Pre-marshalling: rearrange yard before ship arrival (sorting) |
    | **CRP-Stow** | Container stowage planning – yard → vessel loading |
    | **CRP-Stoch** | Stochastic CRP (extends CRP-Time scaffold) |
    | **CRP-U** | Unrestricted BRP: fixed retrieval order 1→…→N, relocations may take the **top** of **any** stack to any non-full stack (uBRP; Jovanović, Tuba & Voß 2019, EJOR) |
    | **CRP-D** | Duplicate-group stowage (extends CRP-Stow scaffold) |

    ### Algorithms
    | Name | Category | Description |
    |---|---|---|
    | **Genetic Algorithm** | Evolutionary | Chromosome = action sequence, uniform crossover |
    | **Caserta (2012) HEUR** | Heuristic | Caserta et al. 2012 fixed-order destination-stack rule |

    ### Debugging (Caserta / Zhu layout call chain)
    Set environment variable **`CRISP_TRACE_LAYOUT=1`** before launching (`streamlit run gui/app.py`).
    Each instance prints to the terminal: `problem_config_for_caserta_dat` → `CRP_R._build_episode` → `apply_caserta_file_to_yard` (layout path is stored as **`layout_file_path`** in ``ProblemConfig.extra``).
    See `core/layout_trace.py`.

    ### Extending the Platform
    **Add a new problem:**
    1. Create `problems/my_problem.py`
    2. Subclass `BaseProblem` and implement `_setup_spaces`, `_build_episode`, `reset`, `step`, `evaluate`, `get_metrics`
    3. Set class attributes `name`, `description`, `tags`
    4. The platform auto-discovers it – no registration needed

    **Add a new algorithm:**
    1. Create `algorithms/category/my_algo.py`
    2. Subclass `BaseAlgorithm` and implement `train`, `get_best_solution`
    3. Set `name`, `category`, `description`
    4. Auto-discovered on next launch
    """)

    st.subheader("Registered Components")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.write("**Problems**")
        for info in get_problem_info():
            st.markdown(f"- **{info['name']}**: {info['description'][:60]}…")
    with ci2:
        st.write("**Algorithms**")
        for info in get_algorithm_info():
            c = info["category"]
            colour = {"RL":"#4E79A7","Evolutionary":"#59A14F","Heuristic":"#F28E2B"}.get(c,"#999")
            st.markdown(
                f"- <span style='background:{colour};color:white;border-radius:3px;"
                f"padding:1px 6px;font-size:0.75em'>{c}</span> **{info['name']}**",
                unsafe_allow_html=True,
            )
