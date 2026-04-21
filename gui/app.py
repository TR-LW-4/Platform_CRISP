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
from typing import Any, Dict, List, Optional

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

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

    # ── Step 1: Problem ──────────────────────────────────────────── #
    st.markdown("**① Problem**")
    selected_problem = st.selectbox("Select Problem", problems, label_visibility="collapsed")

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

    # ── Step 3: Specific Algorithm ───────────────────────────────── #
    st.markdown("**③ Algorithm**")
    algos_in_cat = cat_to_algos.get(selected_category, [])
    if not algos_in_cat:
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

    col_left, col_right = st.columns([1, 2])

    # ── Left: parameter config ────────────────────────────────────── #
    with col_left:
        st.subheader("Problem Config")
        prob_cls    = get_problem_class(selected_problem)
        prob_schema = prob_cls.config_schema() if prob_cls else {}

        prob_params: Dict[str, Any] = {}
        for pname, spec in prob_schema.items():
            label = spec.get("label", pname)
            help_ = spec.get("help", "")
            t     = spec["type"]
            if t == "int":
                prob_params[pname] = st.number_input(
                    label, value=spec["default"],
                    min_value=spec.get("min"), max_value=spec.get("max"),
                    step=1, key=f"p_{pname}", help=help_,
                )
            elif t == "float":
                prob_params[pname] = st.number_input(
                    label, value=float(spec["default"]),
                    min_value=float(spec.get("min", 0)),
                    max_value=float(spec.get("max", 1e6)),
                    format="%.4f", key=f"p_{pname}", help=help_,
                )
            elif t == "bool":
                prob_params[pname] = st.checkbox(
                    label, value=spec["default"], key=f"p_{pname}", help=help_,
                )

        st.subheader("Algorithm Config")
        algo_cls    = get_algorithm_class(selected_algo)
        algo_schema = algo_cls.config_schema() if algo_cls else {}
        # Step label: semantic name for one "iteration" (Episode / Generation / Seed / Update)
        _step_label = getattr(algo_cls, "step_label", "Iteration") if algo_cls else "Iteration"

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

        # ── Control buttons ───────────────────────────────────────── #
        btn_col1, btn_col2 = st.columns(2)
        start_btn = btn_col1.button("▶ Start", type="primary",  use_container_width=True)
        stop_btn  = btn_col2.button("⏹ Stop",  type="secondary", use_container_width=True)

    # ── Right: live visualisation ─────────────────────────────────── #
    with col_right:
        st.subheader("Yard Visualisation")
        status_placeholder  = st.empty()   # step / time / progress bar
        yard_placeholder    = st.empty()
        metric_placeholder  = st.empty()
        chart_placeholder   = st.empty()

    # ── Button handlers ───────────────────────────────────────────── #
    prob_cfg = None   # defined below when Start is clicked
    if start_btn:
        if st.session_state.session and st.session_state.training:
            st.session_state.session.stop()

        # Build problem factory
        prob_cfg = ProblemConfig(**{
            k: v for k, v in prob_params.items()
            if hasattr(ProblemConfig, k) or k in ("num_bays", "num_rows", "max_tiers",
                "num_containers", "num_groups", "num_cranes", "crane_speed",
                "vessel_bays", "vessel_rows", "vessel_tiers",
                "enable_weight", "enable_size", "enable_type", "seed")
        })
        # Extra keys
        extra_keys = set(prob_params) - {f.name for f in ProblemConfig.__dataclass_fields__.values()} \
                     if hasattr(ProblemConfig, '__dataclass_fields__') else set()
        for k in prob_params:
            try:
                getattr(prob_cfg, k)
            except AttributeError:
                prob_cfg.extra[k] = prob_params[k]

        def problem_factory(_cfg=prob_cfg, _cls=prob_cls):
            return _cls(config=_cfg)

        # Build algorithm – put ALL schema keys into extra so train() can read them
        _BASE_KEYS = {
            "max_iterations", "seed", "report_interval", "num_eval_seeds",
            "total_timesteps", "learning_rate", "gamma", "num_envs",
            "num_steps", "batch_size", "hidden_dim",
            "population_size", "crossover_rate", "mutation_rate",
            "tournament_size", "elite_count",
        }
        base_kwargs = {k: v for k, v in algo_params.items() if k in _BASE_KEYS}
        algo_cfg    = AlgorithmConfig(**base_kwargs)
        # All schema-specific keys (num_episodes, max_generations, …) → extra dict
        for k, v in algo_params.items():
            algo_cfg.extra[k] = v

        algo_inst = algo_cls(config=algo_cfg)
        session   = TrainingSession(algo_inst, problem_factory)
        session.start()

        st.session_state.session     = session
        st.session_state.progress    = []
        st.session_state.training    = True
        st.session_state.train_start = time.time()
        st.session_state.step_label  = _step_label   # ← store for progress display
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
                n_bays  = int(prob_params.get("num_bays",  4))
                n_rows  = int(prob_params.get("num_rows",  2))
                n_tiers = int(prob_params.get("max_tiers", 3))
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
    | **BRP-Fixed** | Block Relocation Problem – fixed retrieval order |
    | **BRP-NonFixed** | BRP – free retrieval order optimised by agent |
    | **Pre-Marshalling** | Rearrange yard before ship arrival (sorting) |
    | **CSPP** | Container Stowage Planning – yard → vessel loading |
    | **CSPP-Constrained** | CSPP + weight limits, reefer slots, hazmat segregation |

    ### Algorithms
    | Name | Category | Description |
    |---|---|---|
    | **REINFORCE** | RL | Policy gradient with greedy baseline |
    | **PPO** | RL | Proximal Policy Optimization (Actor-Critic) |
    | **Genetic Algorithm** | Evolutionary | Chromosome = action sequence, uniform crossover |
    | **Greedy Heuristic** | Heuristic | Depth-1 look-ahead baseline |

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
