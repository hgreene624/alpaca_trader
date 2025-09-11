# pages/2_Ticker_Selector_and_Tuning.py
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import json
from src.models.atr_breakout import backtest_single
from src.tuning.evolve import evolve_params, Bounds
from src.utils.plotting import equity_chart
from src.storage import (
    list_portfolios, create_portfolio, add_item,
    save_strategy, set_default_strategy,
    list_param_bounds, save_param_bounds, get_param_bounds, get_default_param_bounds
)
from src.tuning.auto_bounds import suggest_bounds_for, apply_bounds_to_streamlit_state
import src.tuning.evolve as ev

@st.cache_data
def load_universe_index() -> dict:
    """
    Reads data/indexes.json and returns a dict:
    {
      "All (combined)": [tickers...],
      "S&P 500": [...],
      "Nasdaq-100": [...],
      "Dow Jones Industrial Average": [...]
    }
    (tickers are de-duplicated and sorted)
    """
    p = Path(__file__).resolve().parents[1] / "data" / "indexes.json"
    if not p.exists():
        return {"All (combined)": []}

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    indexes = payload.get("indexes", {})
    # dedupe while preserving order, then sort
    def _dedupe_sorted(seq):
        return sorted(dict.fromkeys(seq))

    combined = []
    for lst in indexes.values():
        combined.extend(lst)
    combined = _dedupe_sorted(combined)

    cleaned = {name: _dedupe_sorted(lst) for name, lst in indexes.items()}
    return {"All (combined)": combined, **cleaned}

# ---- Apply queued profile BEFORE widgets are created ----
if "_apply_profile_payload" in st.session_state:
    pr = st.session_state.pop("_apply_profile_payload")  # consume the payload

    # Dates / equity / EA settings
    from datetime import date  # (if not already imported at top)
    st.session_state["ev_start"] = date.fromisoformat(pr["start"])
    st.session_state["ev_end"] = date.fromisoformat(pr["end"])
    st.session_state["ev_eq"] = float(pr.get("starting_equity", 10000))
    st.session_state["ev_pop"] = int(pr.get("pop_size", 40))
    st.session_state["ev_gen"] = int(pr.get("generations", 20))
    st.session_state["ev_cr"] = float(pr.get("crossover_rate", 0.7))
    st.session_state["ev_mr"] = float(pr.get("mutation_rate", 0.35))

    # Bounds (use .get with sane defaults)
    b = pr.get("bounds", {})
    st.session_state["b_breakout_min"] = int(b.get("breakout_min", 20))
    st.session_state["b_breakout_max"] = int(b.get("breakout_max", 120))
    st.session_state["b_exit_min"] = int(b.get("exit_min", 10))
    st.session_state["b_exit_max"] = int(b.get("exit_max", 60))
    st.session_state["b_atr_min"] = int(b.get("atr_min", 7))
    st.session_state["b_atr_max"] = int(b.get("atr_max", 30))
    st.session_state["b_atrmult_min"] = float(b.get("atr_multiple_min", 1.5))
    st.session_state["b_atrmult_max"] = float(b.get("atr_multiple_max", 5.0))
    st.session_state["b_rpt_min"] = float(b.get("risk_per_trade_min", 0.002)) * 100.0
    st.session_state["b_rpt_max"] = float(b.get("risk_per_trade_max", 0.02)) * 100.0
    st.session_state["b_tp_min"] = float(b.get("tp_multiple_min", 0.0))
    st.session_state["b_tp_max"] = float(b.get("tp_multiple_max", 6.0))
    st.session_state["b_allow_trend"] = bool(b.get("allow_trend_filter", True))
    st.session_state["b_sma_fast_min"] = int(b.get("sma_fast_min", 10))
    st.session_state["b_sma_fast_max"] = int(b.get("sma_fast_max", 50))
    st.session_state["b_sma_slow_min"] = int(b.get("sma_slow_min", 40))
    st.session_state["b_sma_slow_max"] = int(b.get("sma_slow_max", 100))
    st.session_state["b_sma_long_min"] = int(b.get("sma_long_min", 100))
    st.session_state["b_sma_long_max"] = int(b.get("sma_long_max", 300))
    st.session_state["b_slope_min"] = int(b.get("long_slope_len_min", 10))
    st.session_state["b_slope_max"] = int(b.get("long_slope_len_max", 50))
    st.session_state["b_hold_min"] = int(b.get("holding_period_min", 0))
    st.session_state["b_hold_max"] = int(b.get("holding_period_max", 120))
    st.session_state["b_cost_min"] = float(b.get("cost_bps_min", 0.0))
    st.session_state["b_cost_max"] = float(b.get("cost_bps_max", 10.0))

# ---- Apply queued auto-bounds BEFORE widgets are created ----
if "_apply_bounds_payload" in st.session_state:
    _ab = st.session_state.pop("_apply_bounds_payload")
    try:
        apply_bounds_to_streamlit_state(_ab)
    except Exception:
        pass

st.set_page_config(page_title="Evolutionary Tuning", page_icon="🧬")
st.title("🧬 Evolutionary Tuning (ATR Breakout)")

# ---------- Ticker & Dates ----------
c1, c2, c3 = st.columns(3)
with c1:
    # --- Universe & Ticker dropdowns ---
    universe_map = load_universe_index()
    universes = list(universe_map.keys()) or ["All (combined)"]

    # remember last universe (optional)
    default_universe = st.session_state.get("ev_universe", universes[0])
    if default_universe not in universes:
        default_universe = universes[0]

    colU, colT = st.columns(2)
    with colU:
        universe = st.selectbox("Universe", universes,
                                index=universes.index(default_universe),
                                key="ev_universe")

    tickers = universe_map.get(universe, [])
    # Seed ticker dropdown with last chosen symbol if it’s still in list
    seed_symbol = st.session_state.get("ev_symbol", tickers[0] if tickers else "AAPL")
    if seed_symbol not in tickers and tickers:
        seed_symbol = tickers[0]

    with colT:
        if tickers:
            symbol = st.selectbox("Ticker", tickers,
                                  index=tickers.index(seed_symbol),
                                  key="ev_symbol_select")
        else:
            # Fallback if file missing/empty
            symbol = st.text_input("Ticker", value=seed_symbol)

    # keep the same key your code expects elsewhere
    st.session_state["ev_symbol"] = symbol.upper().strip()
    symbol = st.session_state["ev_symbol"]
with c2:
    end = st.date_input("End Date", value=date.today(), key="ev_end")
with c3:
    start = st.date_input("Start Date", value=end - timedelta(days=365*3), key="ev_start")

# ---------- Parameter Profiles (bounds + run settings) ----------
st.markdown("**Parameter Profile (bounds + run settings)**")
profiles = list_param_bounds(symbol=symbol, model="atr_breakout")
prof_labels = ["Default (system)"]
prof_ids = ["__DEFAULT__"]
default_bounds = get_default_param_bounds(symbol, "atr_breakout")
if default_bounds:
    prof_labels.append(f"★ Default for {symbol}  ({default_bounds['id'][:8]})")
    prof_ids.append(default_bounds["id"])
for pb in profiles:
    if default_bounds and pb["id"] == default_bounds["id"]:
        continue
    prof_labels.append(f"{pb['name']}  ({pb['id'][:8]})")
    prof_ids.append(pb["id"])

sel_prof_idx = st.selectbox("Profiles", list(range(len(prof_labels))),
                            format_func=lambda i: prof_labels[i], index=0, key="ev_prof_idx")

def _apply_profile_to_state(rec):
    pr = rec["profile"]
    # Dates / equity / EA settings
    st.session_state["ev_start"] = date.fromisoformat(pr["start"])
    st.session_state["ev_end"] = date.fromisoformat(pr["end"])
    st.session_state["ev_eq"] = float(pr.get("starting_equity", 10000))
    st.session_state["ev_pop"] = int(pr.get("pop_size", 40))
    st.session_state["ev_gen"] = int(pr.get("generations", 20))
    st.session_state["ev_cr"] = float(pr.get("crossover_rate", 0.7))
    st.session_state["ev_mr"] = float(pr.get("mutation_rate", 0.35))
    # Bounds (fill with defaults if missing)
    b = pr.get("bounds", {})
    def g(k, d): return b.get(k, d)
    st.session_state["b_breakout_min"] = int(g("breakout_min", 20))
    st.session_state["b_breakout_max"] = int(g("breakout_max", 120))
    st.session_state["b_exit_min"] = int(g("exit_min", 10))
    st.session_state["b_exit_max"] = int(g("exit_max", 60))
    st.session_state["b_atr_min"] = int(g("atr_min", 7))
    st.session_state["b_atr_max"] = int(g("atr_max", 30))
    st.session_state["b_atrmult_min"] = float(g("atr_multiple_min", 1.5))
    st.session_state["b_atrmult_max"] = float(g("atr_multiple_max", 5.0))
    st.session_state["b_rpt_min"] = float(g("risk_per_trade_min", 0.002)) * 100.0
    st.session_state["b_rpt_max"] = float(g("risk_per_trade_max", 0.02)) * 100.0
    st.session_state["b_tp_min"] = float(g("tp_multiple_min", 0.0))
    st.session_state["b_tp_max"] = float(g("tp_multiple_max", 6.0))
    st.session_state["b_allow_trend"] = bool(g("allow_trend_filter", True))
    st.session_state["b_sma_fast_min"] = int(g("sma_fast_min", 10))
    st.session_state["b_sma_fast_max"] = int(g("sma_fast_max", 50))
    st.session_state["b_sma_slow_min"] = int(g("sma_slow_min", 40))
    st.session_state["b_sma_slow_max"] = int(g("sma_slow_max", 100))
    st.session_state["b_sma_long_min"] = int(g("sma_long_min", 100))
    st.session_state["b_sma_long_max"] = int(g("sma_long_max", 300))
    st.session_state["b_slope_min"] = int(g("long_slope_len_min", 10))
    st.session_state["b_slope_max"] = int(g("long_slope_len_max", 50))
    st.session_state["b_hold_min"] = int(g("holding_period_min", 0))
    st.session_state["b_hold_max"] = int(g("holding_period_max", 120))
    st.session_state["b_cost_min"] = float(g("cost_bps_min", 0.0))
    st.session_state["b_cost_max"] = float(g("cost_bps_max", 10.0))

if st.button("Load Profile", key="btn_load_prof"):
    chosen_id = prof_ids[sel_prof_idx]
    if chosen_id != "__DEFAULT__":
        rec = get_param_bounds(chosen_id)
        if rec:
            # Queue the profile payload; it will be applied before widgets on next run
            st.session_state["_apply_profile_payload"] = rec["profile"]
            st.success(f"Loaded profile: {rec['name']}")
            st.rerun()
        else:
            st.warning("Profile not found.")
    else:
        st.info("Loaded default system values.")

# ---------- Validation constraints (anti-cherry-picking) ----------
with st.expander("Validation trade constraints", expanded=False):
    min_valid_trades = st.slider(
        "Min validation trades",
        0, 30,
        int(getattr(ev, "MIN_VALIDATION_TRADES", 8)),
        1,
        help="Individuals with fewer trades than this on the validation slice are rejected outright."
    )
    target_valid_trades = st.slider(
        "Target validation trades (scale)",
        1, 60,
        int(getattr(ev, "TARGET_VALIDATION_TRADES", 12)),
        1,
        help="Soft target: fitness scales up with trade count up to this value (prevents tiny-sample flukes)."
    )

# Apply to evolve module for this run
ev.MIN_VALIDATION_TRADES = int(min_valid_trades)
ev.TARGET_VALIDATION_TRADES = int(target_valid_trades)

# ---------- EA settings ----------
# --- Self-adaptive GA/ES (advanced) ---
with st.expander("Self-adaptive GA/ES (advanced)", expanded=False):
    sa_p_on   = st.slider(
        "Mask on (p_on)", 0.0, 0.2, float(getattr(ev, "SA_P_ON", 0.05)), 0.005,
        help="Chance to activate an inactive gene at each mutation."
    )
    sa_p_off  = st.slider(
        "Mask off (p_off)", 0.0, 0.2, float(getattr(ev, "SA_P_OFF", 0.02)), 0.005,
        help="Chance to deactivate an active gene at each mutation."
    )

    col1, col2 = st.columns(2)
    with col1:
        sa_step_min = st.slider(
            "Step min frac (ints/add floats)", 0.0, 0.2,
            float(getattr(ev, "SA_STEP_MIN_FRAC", 0.02)), 0.005,
            help="Lower bound for per-gene step size as a fraction of range."
        )
        sa_log_min  = st.slider(
            "Log-step min (mult floats)", 0.01, 0.5,
            float(getattr(ev, "SA_LOGSTEP_MIN", 0.03)), 0.01,
            help="Min std-dev in log domain for multiplicative genes (atr_multiple, risk_per_trade)."
        )
    with col2:
        sa_step_max = st.slider(
            "Step max frac (ints/add floats)", 0.05, 0.6,
            float(getattr(ev, "SA_STEP_MAX_FRAC", 0.35)), 0.01,
            help="Upper bound for per-gene step size as a fraction of range."
        )
        sa_log_max  = st.slider(
            "Log-step max (mult floats)", 0.05, 0.8,
            float(getattr(ev, "SA_LOGSTEP_MAX", 0.35)), 0.01,
            help="Max std-dev in log domain for multiplicative genes."
        )

    sa_sigma = st.slider(
        "Step-size noise σ", 0.0, 0.5, float(getattr(ev, "SA_SIGMA_NOISE", 0.15)), 0.01,
        help="Global noise applied to step sizes each mutation (self-adaptation).\nHigher = more exploration."
    )

# Sanity: enforce min<=max where applicable
if sa_step_max < sa_step_min:
    sa_step_max = sa_step_min
if sa_log_max < sa_log_min:
    sa_log_max = sa_log_min

# Apply overrides to evolve module globals so the current run uses them
ev.SA_P_ON = sa_p_on
ev.SA_P_OFF = sa_p_off
ev.SA_STEP_MIN_FRAC = sa_step_min
ev.SA_STEP_MAX_FRAC = sa_step_max
ev.SA_LOGSTEP_MIN = sa_log_min
ev.SA_LOGSTEP_MAX = sa_log_max
ev.SA_SIGMA_NOISE = sa_sigma

c4, c5, c6 = st.columns(3)
with c4:
    starting_equity = st.number_input("Starting Equity ($)", min_value=1000, value=10_000, step=500, key="ev_eq")
with c5:
    pop_size = st.slider("Population size", 10, 200, 40, 5, key="ev_pop")
    generations = st.slider("Generations", 5, 100, 20, 1, key="ev_gen")
with c6:
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.7, 0.05, key="ev_cr")
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.35, 0.05, key="ev_mr")

# ---------- Bounds ----------
# --- Auto-bounds from recent data ---
with st.container():
    c1, c2, c3 = st.columns([1,1,3])
    with c1:
        if st.button("Auto-bounds", help="Analyze recent data and set sensible bounds"):
            # Resolve symbol robustly from session state or the local selectbox value
            sym = (
                st.session_state.get("ev_symbol")
                or st.session_state.get("ev_symbol_select")
                or st.session_state.get("ticker")
                or st.session_state.get("symbol")
                or (symbol if 'symbol' in locals() else None)
            )
            sym = (sym or "").strip().upper()
            start_dt = st.session_state.get("ev_start")
            end_dt   = st.session_state.get("ev_end")
            if not sym:
                st.warning("Select a ticker first.")
            else:
                rec = suggest_bounds_for(sym, start_dt, end_dt)
                st.session_state["_apply_bounds_payload"] = rec
                st.session_state["auto_bounds_notes"] = rec.get("notes", "Auto-bounds computed.")
                st.toast(st.session_state["auto_bounds_notes"], icon="✅")
                st.rerun()
    with c2:
        txt = st.session_state.get("auto_bounds_notes")
        if txt:
            st.caption(txt)

st.subheader("Parameter Bounds")
# Core
cA1, cA2, cA3 = st.columns(3)
with cA1:
    breakout_min = st.number_input("Breakout min", 10, 400, 20, 1, key="b_breakout_min")
    exit_min = st.number_input("Exit min", 5, 300, 10, 1, key="b_exit_min")
    atr_min = st.number_input("ATR min", 5, 150, 7, 1, key="b_atr_min")
with cA2:
    breakout_max = st.number_input("Breakout max", breakout_min + 1, 500, 120, 1, key="b_breakout_max")
    exit_max = st.number_input("Exit max", exit_min + 1, 300, 60, 1, key="b_exit_max")
    atr_max = st.number_input("ATR max", atr_min + 1, 200, 30, 1, key="b_atr_max")
with cA3:
    atr_multiple_min = st.number_input("ATR multiple min", 0.5, 10.0, 1.5, 0.1, key="b_atrmult_min")
    atr_multiple_max = st.number_input("ATR multiple max", atr_multiple_min + 0.1, 15.0, 5.0, 0.1, key="b_atrmult_max")
    risk_per_trade_min = st.number_input("Risk/trade min (%)", 0.05, 5.0, 0.2, 0.05, key="b_rpt_min")
    risk_per_trade_max = st.number_input("Risk/trade max (%)", risk_per_trade_min + 0.05, 10.0, 2.0, 0.05, key="b_rpt_max")

# Risk/TP/Costs
cB1, cB2, cB3 = st.columns(3)
with cB1:
    tp_multiple_min = st.number_input("TP multiple min", 0.0, 10.0, 0.0, 0.1, key="b_tp_min")
    tp_multiple_max = st.number_input("TP multiple max", tp_multiple_min + 0.1, 10.0, 6.0, 0.1, key="b_tp_max")
with cB2:
    cost_bps_min = st.number_input("Cost bps min", 0.0, 50.0, 0.0, 0.1, key="b_cost_min")
    cost_bps_max = st.number_input("Cost bps max", cost_bps_min + 0.1, 50.0, 10.0, 0.1, key="b_cost_max")
with cB3:
    holding_min = st.number_input("Hold limit min (bars, 0=off)", 0, 252, 0, 1, key="b_hold_min")
    holding_max = st.number_input("Hold limit max (bars)", holding_min + 1, 252, 120, 1, key="b_hold_max")

# Trend
st.checkbox("Allow tuner to use Trend Filter (fast>slow & long SMA slope up)", value=True, key="b_allow_trend")
cC1, cC2, cC3 = st.columns(3)
with cC1:
    sma_fast_min = st.number_input("SMA fast min", 5, 100, 10, 1, key="b_sma_fast_min")
    sma_slow_min = st.number_input("SMA slow min", 10, 200, 40, 1, key="b_sma_slow_min")
with cC2:
    sma_fast_max = st.number_input("SMA fast max", sma_fast_min + 1, 150, 50, 1, key="b_sma_fast_max")
    sma_slow_max = st.number_input("SMA slow max", sma_slow_min + 1, 250, 100, 1, key="b_sma_slow_max")
with cC3:
    sma_long_min = st.number_input("SMA long min", 50, 400, 100, 1, key="b_sma_long_min")
    sma_long_max = st.number_input("SMA long max", sma_long_min + 1, 500, 300, 1, key="b_sma_long_max")
    long_slope_min = st.number_input("Long slope len min", 5, 100, 10, 1, key="b_slope_min")
    long_slope_max = st.number_input("Long slope len max", long_slope_min + 1, 100, 50, 1, key="b_slope_max")

# ---------- Run evolution ----------
prog = st.progress(0, text="Idle")

if st.button("Run Evolutionary Tuning", type="primary", key="btn_ev"):
    b = Bounds(
        breakout_min=int(breakout_min), breakout_max=int(breakout_max),
        exit_min=int(exit_min), exit_max=int(exit_max),
        atr_min=int(atr_min), atr_max=int(atr_max),
        atr_multiple_min=float(atr_multiple_min), atr_multiple_max=float(atr_multiple_max),
        risk_per_trade_min=float(risk_per_trade_min)/100.0, risk_per_trade_max=float(risk_per_trade_max)/100.0,
        tp_multiple_min=float(tp_multiple_min), tp_multiple_max=float(tp_multiple_max),
        allow_trend_filter=bool(st.session_state["b_allow_trend"]),
        sma_fast_min=int(sma_fast_min), sma_fast_max=int(sma_fast_max),
        sma_slow_min=int(sma_slow_min), sma_slow_max=int(sma_slow_max),
        sma_long_min=int(sma_long_min), sma_long_max=int(sma_long_max),
        long_slope_len_min=int(long_slope_min), long_slope_len_max=int(long_slope_max),
        holding_period_min=int(holding_min), holding_period_max=int(holding_max),
        cost_bps_min=float(cost_bps_min), cost_bps_max=float(cost_bps_max),
        pop_size=int(pop_size), generations=int(generations),
        crossover_rate=float(crossover_rate), mutation_rate=float(mutation_rate),
    )

    def _cb(done: int, total: int, best_fit: float):
        prog.progress(min(int(100 * done / total), 100), text=f"Generation {done}/{total} — best fit: {best_fit:.3f}")

    with st.spinner("Evolving…"):
        try:
            best_params, _best_metrics_fit, _hist = evolve_params(
                symbol=symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                starting_equity=float(starting_equity),
                bounds=b,
                pop_size=int(pop_size),
                generations=int(generations),
                crossover_rate=float(crossover_rate),
                mutation_rate=float(mutation_rate),
                random_seed=42,
                progress_cb=_cb,
            )
            prog.progress(100, text="Done")

            # Run one final backtest for consistent metrics & equity
            res_best = backtest_single(
                symbol,
                start.isoformat(),
                end.isoformat(),
                best_params["breakout_n"],
                best_params["exit_n"],
                best_params["atr_n"],
                float(starting_equity),
                atr_multiple=best_params["atr_multiple"],
                risk_per_trade=best_params["risk_per_trade"],
                allow_fractional=True,
            )
            # The engine already applies TP/Trend/Holding/Costs when called via evolve.
            # backtest_single wrapper doesn't expose those; we keep equity/metrics from evolution result instead:
            # recompute via engine directly using best_params to ensure parity:
            from src.backtest.engine import ATRParams, backtest_atr_breakout
            final_params = ATRParams(
                breakout_n=int(best_params["breakout_n"]),
                exit_n=int(best_params["exit_n"]),
                atr_n=int(best_params["atr_n"]),
                atr_multiple=float(best_params["atr_multiple"]),
                risk_per_trade=float(best_params["risk_per_trade"]),
                allow_fractional=True,
                slippage_bp=5.0,
                cost_bps=float(best_params.get("cost_bps", 1.0)),
                fee_per_trade=0.0,
                tp_multiple=(None if best_params.get("tp_multiple", 0.0) <= 0 else float(best_params["tp_multiple"])),
                use_trend_filter=bool(best_params.get("use_trend_filter", False)),
                sma_fast=int(best_params.get("sma_fast", 30)),
                sma_slow=int(best_params.get("sma_slow", 50)),
                sma_long=int(best_params.get("sma_long", 150)),
                long_slope_len=int(best_params.get("long_slope_len", 15)),
                holding_period_limit=(None if int(best_params.get("holding_period_limit", 0)) <= 0 else int(best_params["holding_period_limit"])),
            )
            final = backtest_atr_breakout(symbol, start.isoformat(), end.isoformat(), float(starting_equity), final_params)

            st.session_state["ev_best_params"] = dict(best_params)
            st.session_state["ev_best_metrics"] = final["metrics"]
            st.session_state["ev_best_equity"] = final["equity"]

            st.success("Evolution complete.")
        except Exception as e:
            st.error(f"Error during evolution: {e}")

# ---------- Results (persistent) ----------
def _render_results():
    if "ev_best_params" not in st.session_state or "ev_best_metrics" not in st.session_state:
        return
    st.subheader(f"Best Parameters — {st.session_state.get('ev_symbol', 'TICKER')}")
    st.write(pd.DataFrame([st.session_state["ev_best_metrics"]]))
    st.plotly_chart(
        equity_chart(st.session_state["ev_best_equity"], title=f"Equity — {st.session_state.get('ev_symbol','TICKER')} (Best)"),
        use_container_width=True,
    )

    # Save Parameter Bounds Profile (FORM)
    with st.form("save_bounds_form", clear_on_submit=False):
        st.markdown("**Save Parameter Bounds Profile**")
        auto_name = (
            f"{symbol}_{start}_{end}"
            f"_pop{st.session_state['ev_pop']}_gen{st.session_state['ev_gen']}"
            f"_b{int(st.session_state['b_breakout_min'])}-{int(st.session_state['b_breakout_max'])}"
            f"_e{int(st.session_state['b_exit_min'])}-{int(st.session_state['b_exit_max'])}"
            f"_atr{int(st.session_state['b_atr_min'])}-{int(st.session_state['b_atr_max'])}"
            f"_am{st.session_state['b_atrmult_min']:.1f}-{st.session_state['b_atrmult_max']:.1f}"
            f"_rpt{st.session_state['b_rpt_min']:.2f}-{st.session_state['b_rpt_max']:.2f}"
            f"_tp{st.session_state['b_tp_min']:.1f}-{st.session_state['b_tp_max']:.1f}"
            f"_cost{st.session_state['b_cost_min']:.1f}-{st.session_state['b_cost_max']:.1f}"
        )
        prof_name = st.text_input("Profile name", value=auto_name, key="ev_prof_name")
        save_bounds = st.form_submit_button("Save Parameter Bounds")
        if save_bounds:
            profile = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "starting_equity": float(st.session_state["ev_eq"]),
                "pop_size": int(st.session_state["ev_pop"]),
                "generations": int(st.session_state["ev_gen"]),
                "crossover_rate": float(st.session_state["ev_cr"]),
                "mutation_rate": float(st.session_state["ev_mr"]),
                "bounds": {
                    "breakout_min": int(st.session_state["b_breakout_min"]),
                    "breakout_max": int(st.session_state["b_breakout_max"]),
                    "exit_min": int(st.session_state["b_exit_min"]),
                    "exit_max": int(st.session_state["b_exit_max"]),
                    "atr_min": int(st.session_state["b_atr_min"]),
                    "atr_max": int(st.session_state["b_atr_max"]),
                    "atr_multiple_min": float(st.session_state["b_atrmult_min"]),
                    "atr_multiple_max": float(st.session_state["b_atrmult_max"]),
                    "risk_per_trade_min": float(st.session_state["b_rpt_min"]) / 100.0,
                    "risk_per_trade_max": float(st.session_state["b_rpt_max"]) / 100.0,
                    "tp_multiple_min": float(st.session_state["b_tp_min"]),
                    "tp_multiple_max": float(st.session_state["b_tp_max"]),
                    "allow_trend_filter": bool(st.session_state["b_allow_trend"]),
                    "sma_fast_min": int(st.session_state["b_sma_fast_min"]),
                    "sma_fast_max": int(st.session_state["b_sma_fast_max"]),
                    "sma_slow_min": int(st.session_state["b_sma_slow_min"]),
                    "sma_slow_max": int(st.session_state["b_sma_slow_max"]),
                    "sma_long_min": int(st.session_state["b_sma_long_min"]),
                    "sma_long_max": int(st.session_state["b_sma_long_max"]),
                    "long_slope_len_min": int(st.session_state["b_slope_min"]),
                    "long_slope_len_max": int(st.session_state["b_slope_max"]),
                    "holding_period_min": int(st.session_state["b_hold_min"]),
                    "holding_period_max": int(st.session_state["b_hold_max"]),
                    "cost_bps_min": float(st.session_state["b_cost_min"]),
                    "cost_bps_max": float(st.session_state["b_cost_max"]),
                },
            }
            rec = save_param_bounds(symbol, "atr_breakout", profile, name=prof_name, is_default=False)
            st.success(f"Saved profile: {rec['name']}  ({rec['id'][:8]})")

    # Save Best Strategy (FORM)
    with st.form("save_best_form", clear_on_submit=False):
        st.markdown("**Save Best Model Params**")
        default_best_name = f"{symbol}-Best"
        best_name = st.text_input("Strategy name", value=default_best_name, key="ev_best_name")
        set_default_model = st.checkbox("Set as default model params for this ticker", value=False, key="ev_best_default")
        save_best = st.form_submit_button("Save Best Strategy")
        if save_best:
            srec = save_strategy(symbol, "atr_breakout", st.session_state["ev_best_params"], name=best_name, is_default=set_default_model)
            if set_default_model:
                set_default_strategy(symbol, "atr_breakout", srec["id"])
            st.success(f"Saved best params as: {srec['name']}  ({srec['id'][:8]})")

    # For portfolio save block
    st.session_state["last_result"] = {"symbol": symbol, "params": st.session_state["ev_best_params"]}

_render_results()

# ---------- Save to Portfolio ----------
st.divider()
st.subheader("Save this configuration to a Portfolio")

portfolios = list_portfolios()
names = ["— Create new —"] + [p["name"] for p in portfolios]
choice = st.selectbox("Target Portfolio", names, index=1 if len(portfolios) > 0 else 0)
new_name = ""
if choice == "— Create new —":
    new_name = st.text_input("New Portfolio Name", value="My Portfolio")

if st.button("Save to Portfolio", key="btn_save"):
    last = st.session_state.get("last_result")
    if not last:
        st.warning("Run tuning first.")
    else:
        if choice == "— Create new —":
            if not new_name.strip():
                st.warning("Please provide a name for the new portfolio.")
            else:
                p = create_portfolio(new_name.strip())
                add_item(p["id"], last["symbol"], "atr_breakout", last["params"])
                st.success(f"Saved {last['symbol']} to portfolio '{p['name']}'.")
        else:
            pid = next((p["id"] for p in portfolios if p["name"] == choice), None)
            if pid:
                add_item(pid, last["symbol"], "atr_breakout", last["params"])
                st.success(f"Saved {last['symbol']} to portfolio '{choice}'.")
            else:
                st.error("Portfolio not found.")