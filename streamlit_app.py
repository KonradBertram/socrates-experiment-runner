from __future__ import annotations

import hashlib
import os
import random
import re
import secrets
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from socrates_core import (
    APP_NAME,
    DEFAULT_DEMOGRAPHIC_DIMENSIONS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOP_K,
    DEMOGRAPHIC_DEFINITIONS,
    ESTIMATED_COMPLETION_TOKENS_PER_RESPONSE,
    MAX_APP_WORKERS,
    MAX_NEW_TOKENS,
    MAX_RESPONSE_OPTIONS,
    MAX_SIMULATIONS,
    MAX_SLOT_ATTEMPTS,
    MODEL_ID,
    OPTION_CODES,
    RESPONSE_STRUCTURE_CATEGORICAL,
    RESPONSE_STRUCTURE_ORDERED,
    RESPONSE_STRUCTURE_OPTIONS,
    SYSTEM_PROMPT,
    US_PRESET_NOTE,
    US_PRESET_SOURCES,
    build_simulation_plan,
    build_user_prompt,
    config_fingerprint,
    create_excel_export,
    distributions_from_preset,
    estimate_run,
    generate_profiles,
    get_runtime_info,
    largest_remainder_counts,
    option_orders_for_structure,
    overall_result_rows,
    parse_answer_options,
    profile_balance_rows,
    run_one,
    segment_result_rows,
    serialize_messages_for_tokenizer,
    heuristic_token_count,
    validate_distribution,
    build_run_timestamp,
)


st.set_page_config(page_title=APP_NAME, layout="wide")

st.markdown(
    """
    <style>
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #000000 !important;
    }
    .small-note {
        color: #AAB2C0;
        font-size: 0.88rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    try:
        value = st.secrets.get("FEATHERLESS_API_KEY")
        if value:
            return str(value)
    except Exception:
        pass
    value = os.getenv("FEATHERLESS_API_KEY")
    return value if value else None


def equal_allocations(number_variants: int) -> list[int]:
    base = 100 // number_variants
    allocations = [base] * number_variants
    for index in range(100 - sum(allocations)):
        allocations[index] += 1
    return allocations


def format_percentage(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}%"


def format_pp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    return f"{value:+.1f} pp"


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_")
    return cleaned[:80] or "socrates_experiment"


def render_advanced_settings() -> dict[str, Any]:
    state_defaults = {
        "socrates_temperature": DEFAULT_TEMPERATURE,
        "socrates_top_p": DEFAULT_TOP_P,
        "socrates_use_top_k": DEFAULT_USE_TOP_K,
        "socrates_top_k": DEFAULT_TOP_K,
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.expander("Advanced model settings"):
        st.caption(
            "The Socrates paper used temperature 0.6 and top-p 0.9 for open-source "
            "model inference. Keep these settings fixed across variants."
        )
        if st.button("Reset to paper-aligned defaults", key="reset_socrates_settings"):
            for key, value in state_defaults.items():
                st.session_state[key] = value
            st.rerun()

        left, right = st.columns(2)
        with left:
            temperature = float(
                st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    key="socrates_temperature",
                    help="Controls sampling diversity. The Socrates paper used 0.6.",
                )
            )
        with right:
            top_p = float(
                st.number_input(
                    "Top-p",
                    min_value=0.01,
                    max_value=1.0,
                    step=0.05,
                    key="socrates_top_p",
                    help="Nucleus-sampling threshold. The Socrates paper used 0.9.",
                )
            )

        use_top_k = st.checkbox(
            "Enable Top-k",
            key="socrates_use_top_k",
            help="Optional additional sampling restriction. Disabled by default.",
        )
        top_k = int(
            st.number_input(
                "Top-k",
                min_value=1,
                max_value=500,
                step=1,
                key="socrates_top_k",
                disabled=not use_top_k,
            )
        )

    return {
        "temperature": temperature,
        "top_p": top_p,
        "use_top_k": use_top_k,
        "top_k": top_k if use_top_k else None,
    }


def render_demographic_controls(
    dimensions: Sequence[str],
    use_us_preset: bool,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    distributions: dict[str, dict[str, float]] = {}
    errors: list[str] = []

    for dimension in dimensions:
        definition = DEMOGRAPHIC_DEFINITIONS[dimension]
        categories = list(definition["categories"])
        presets = list(definition["preset"])

        with st.expander(dimension, expanded=dimension in {"Age", "Gender"}):
            if use_us_preset:
                dataframe = pd.DataFrame(
                    {"Category": categories, "Share (%)": presets}
                )
                st.dataframe(dataframe, hide_index=True, use_container_width=True)
                distribution = dict(zip(categories, [float(value) for value in presets]))
            else:
                st.caption("Enter percentages. Values must total exactly 100%.")
                distribution = {}
                columns = st.columns(2)
                for index, (category, preset) in enumerate(zip(categories, presets)):
                    key = f"custom_demo::{dimension}::{index}"
                    if key not in st.session_state:
                        st.session_state[key] = float(preset)
                    with columns[index % 2]:
                        distribution[category] = float(
                            st.number_input(
                                category,
                                min_value=0.0,
                                max_value=100.0,
                                step=1.0,
                                key=key,
                            )
                        )
                total = sum(distribution.values())
                st.metric("Total", f"{total:.1f}%")
                if not validate_distribution(distribution):
                    errors.append(f"{dimension} percentages total {total:.1f}%; they must total 100%.")

            distributions[dimension] = distribution

    return distributions, errors


def validate_inputs(
    *,
    experiment_name: str,
    line_of_business: str,
    customer_journey_type: str,
    context: str,
    variants: Sequence[Mapping[str, Any]],
    allocation_total: float,
    allocation_counts: Sequence[int],
    outcome_question: str,
    answer_options: Sequence[str],
    response_option_structure: str,
    target_behavior: str | None,
    demographic_dimensions: Sequence[str],
    demographic_errors: Sequence[str],
    total_simulations: int,
    control_variant: str | None,
) -> list[str]:
    errors = list(demographic_errors)

    if not experiment_name.strip():
        errors.append("Add an experiment name.")
    if not line_of_business.strip():
        errors.append("Add the line of business.")
    if not customer_journey_type.strip():
        errors.append("Add the customer journey type.")
    if not context.strip():
        errors.append("Add the customer and experiment context.")

    names = [str(variant["name"]).strip() for variant in variants]
    if any(not name for name in names):
        errors.append("Give every variant a name.")
    if len({name.casefold() for name in names if name}) != len(names):
        errors.append("Variant names must be unique.")
    if any(not str(variant["text"]).strip() for variant in variants):
        errors.append("Add content for every variant.")
    if abs(allocation_total - 100.0) > 0.01:
        errors.append(f"Variant allocations total {allocation_total:.1f}%; they must total 100%.")
    for index, count in enumerate(allocation_counts):
        if count <= 0:
            errors.append(
                f"{names[index] or f'Variant {index + 1}'} receives zero simulations. "
                "Increase its allocation or the total N."
            )
    if total_simulations < len(variants):
        errors.append("Total valid simulations must be at least the number of variants.")

    if not outcome_question.strip():
        errors.append("Add the question asked after the customer sees the variant.")

    if response_option_structure not in RESPONSE_STRUCTURE_OPTIONS:
        errors.append("Select a valid response option structure.")

    if len(answer_options) < 2:
        errors.append("Add at least two answer options, one per line.")
    if len(answer_options) > MAX_RESPONSE_OPTIONS:
        errors.append(f"Use no more than {MAX_RESPONSE_OPTIONS} answer options.")
    if len({option.casefold() for option in answer_options}) != len(answer_options):
        errors.append("Answer options must be unique.")
    if not target_behavior or target_behavior not in answer_options:
        errors.append("Select a target behavior from the answer options.")
    if not control_variant or control_variant not in names:
        errors.append("Select a valid control variant.")
    if not demographic_dimensions:
        errors.append("Select at least one supported demographic dimension.")

    return errors


def build_config(
    *,
    experiment_name: str,
    line_of_business: str,
    customer_journey_type: str,
    context: str,
    variants: Sequence[Mapping[str, Any]],
    allocation_counts: Sequence[int],
    outcome_question: str,
    answer_options: Sequence[str],
    response_option_structure: str,
    target_behavior: str,
    total_simulations: int,
    control_variant: str,
    use_us_preset: bool,
    demographic_dimensions: Sequence[str],
    demographic_distributions: Mapping[str, Mapping[str, float]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    built_variants = []
    for variant, count in zip(variants, allocation_counts):
        built_variants.append(
            {
                "name": str(variant["name"]).strip(),
                "text": str(variant["text"]).strip(),
                "allocation": float(variant["allocation"]),
                "count": int(count),
            }
        )

    return {
        "experiment_name": experiment_name.strip(),
        "line_of_business": line_of_business.strip(),
        "customer_journey_type": customer_journey_type.strip(),
        "context": context.strip(),
        "variants": built_variants,
        "outcome_question": outcome_question.strip(),
        "answer_options": list(answer_options),
        "response_option_structure": response_option_structure,
        "target_behavior": target_behavior,
        "total_simulations": int(total_simulations),
        "control_variant": control_variant,
        "use_us_preset": bool(use_us_preset),
        "demographic_dimensions": list(demographic_dimensions),
        "demographic_distributions": {
            dimension: {
                category: float(percentage)
                for category, percentage in demographic_distributions[dimension].items()
            }
            for dimension in demographic_dimensions
        },
        "settings": dict(settings),
    }


def planned_segment_warning(config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> str | None:
    minimum = None
    minimum_description = None
    for dimension in config["demographic_dimensions"]:
        categories = config["demographic_distributions"][dimension].keys()
        for category in categories:
            for variant in config["variants"]:
                count = sum(
                    1
                    for job in jobs
                    if job["variant"] == variant["name"]
                    and job["profile_segments"].get(dimension) == category
                )
                if minimum is None or count < minimum:
                    minimum = count
                    minimum_description = f"{dimension}: {category} × {variant['name']}"
    if minimum is not None and minimum < 10:
        return (
            f"Some segment-by-variant cells are very small. The smallest planned cell is "
            f"{minimum_description} with N={minimum}. Segment effects will be unstable; "
            "increase total N or simplify the demographic segmentation."
        )
    return None


def execute_experiment(
    config: Mapping[str, Any],
    planned_jobs: Sequence[Mapping[str, Any]],
    api_key: str,
    runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    total_target = len(planned_jobs)
    workers = max(1, min(int(runtime_info.get("workers", 1) or 1), MAX_APP_WORKERS))

    all_results: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    accepted_slots: set[str] = set()

    input_tokens = 0
    output_tokens = 0
    actual_cost = 0.0
    cost_known = bool(runtime_info.get("pricing_available"))
    invalid_outputs = 0
    api_failed_attempts = 0
    retry_events = 0
    http_requests = 0
    next_attempt_id = 1

    st.subheader("Running experiment")
    progress_bar = st.progress(0.0)
    metric_columns = st.columns(5)
    valid_box = metric_columns[0].empty()
    invalid_box = metric_columns[1].empty()
    api_box = metric_columns[2].empty()
    token_box = metric_columns[3].empty()
    cost_box = metric_columns[4].empty()
    status_box = st.empty()

    valid_box.metric("Valid responses", f"0 / {total_target:,}")
    invalid_box.metric("Invalid outputs retried", "0")
    api_box.metric("API failures retried", "0")
    token_box.metric("Tokens used", "0")
    cost_box.metric("Cost so far", "$0.0000" if cost_known else "Unavailable")

    last_ui_update = 0.0

    def with_attempt_id(job: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal next_attempt_id
        prepared = dict(job)
        prepared["attempt_id"] = next_attempt_id
        next_attempt_id += 1
        return prepared

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Any, dict[str, Any]] = {}
        for base_job in planned_jobs:
            job = with_attempt_id(base_job)
            future = executor.submit(
                run_one,
                job,
                api_key,
                config["settings"],
                runtime_info,
            )
            pending[future] = job

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **job,
                        "raw_completion": "",
                        "selected_code": None,
                        "selected_option": None,
                        "selected_option_id": None,
                        "status": "api_error",
                        "error": str(exc),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": None,
                        "http_attempts": 0,
                        "retry_events": 0,
                        "finish_reason": None,
                    }

                all_results.append(result)
                input_tokens += int(result.get("prompt_tokens", 0) or 0)
                output_tokens += int(result.get("completion_tokens", 0) or 0)
                retry_events += int(result.get("retry_events", 0) or 0)
                http_requests += int(result.get("http_attempts", 0) or 0)
                if result.get("cost") is not None:
                    actual_cost += float(result["cost"])

                slot_id = str(result["slot_id"])
                if result["status"] == "valid" and slot_id not in accepted_slots:
                    accepted_slots.add(slot_id)
                    accepted_rows.append(result)
                else:
                    if result["status"] == "invalid_output":
                        invalid_outputs += 1
                    elif result["status"] == "api_error":
                        api_failed_attempts += 1

                    if (
                        slot_id not in accepted_slots
                        and int(result["slot_attempt"]) < MAX_SLOT_ATTEMPTS
                    ):
                        replacement = dict(job)
                        replacement["slot_attempt"] = int(result["slot_attempt"]) + 1
                        replacement = with_attempt_id(replacement)
                        replacement_future = executor.submit(
                            run_one,
                            replacement,
                            api_key,
                            config["settings"],
                            runtime_info,
                        )
                        pending[replacement_future] = replacement

                valid_total = len(accepted_rows)
                now = time.time()
                if now - last_ui_update >= 0.25 or valid_total == total_target:
                    progress_bar.progress(min(1.0, valid_total / total_target))
                    valid_box.metric("Valid responses", f"{valid_total:,} / {total_target:,}")
                    invalid_box.metric("Invalid outputs retried", f"{invalid_outputs:,}")
                    api_box.metric("API failures retried", f"{api_failed_attempts:,}")
                    token_box.metric("Tokens used", f"{input_tokens + output_tokens:,}")
                    if cost_known:
                        cost_box.metric("Cost so far", f"${actual_cost:,.4f}")
                    status_box.caption(
                        f"Up to {workers} requests in parallel · "
                        f"{len(all_results):,} completed attempts · "
                        f"{retry_events:,} temporary API retry event(s)"
                    )
                    last_ui_update = now

    valid_total = len(accepted_rows)
    progress_bar.progress(min(1.0, valid_total / total_target))
    status_box.empty()
    all_results.sort(key=lambda item: int(item["attempt_id"]))
    accepted_rows.sort(key=lambda item: item["slot_id"])

    return {
        "rows": all_results,
        "accepted_rows": accepted_rows,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_cost": actual_cost if cost_known else None,
        "invalid_outputs": invalid_outputs,
        "api_failed_attempts": api_failed_attempts,
        "retry_events": retry_events,
        "http_requests": http_requests,
        "simulation_attempts": len(all_results),
        "valid_total": valid_total,
        "total_target": total_target,
        "workers": workers,
        "run_timestamp": build_run_timestamp(),
    }


def render_results(
    config: Mapping[str, Any],
    planned_jobs: Sequence[Mapping[str, Any]],
    run_data: Mapping[str, Any],
    excel_bytes: bytes | None,
) -> None:
    if run_data["valid_total"] == run_data["total_target"]:
        st.success(
            f"Experiment complete — collected all {run_data['total_target']:,} "
            "requested valid Socrates responses."
        )
    else:
        st.warning(
            "The three-attempt safety ceiling was reached for one or more respondent "
            f"slots. Collected {run_data['valid_total']:,} of "
            f"{run_data['total_target']:,} requested valid responses."
        )

    overall_rows = overall_result_rows(config, run_data["accepted_rows"])
    overall_df = pd.DataFrame(overall_rows)

    st.subheader("Target behavior")
    target_chart = overall_df[["Variant", "Target rate (%)"]].rename(
        columns={"Target rate (%)": "Target behavior rate (%)"}
    )
    st.bar_chart(
        target_chart,
        x="Variant",
        y="Target behavior rate (%)",
        height=390,
    )

    headline_df = overall_df[
        [
            "Variant",
            "Control",
            "Target count",
            "Valid N",
            "Target rate (%)",
            "Effect vs control (pp)",
        ]
    ].copy()
    headline_df["Target rate"] = headline_df["Target rate (%)"].map(format_percentage)
    headline_df["Effect vs control"] = headline_df["Effect vs control (pp)"].map(format_pp)
    headline_df = headline_df[
        ["Variant", "Control", "Target count", "Valid N", "Target rate", "Effect vs control"]
    ]
    st.dataframe(headline_df, hide_index=True, use_container_width=True)
    st.caption(
        f"Target behavior: **{config['target_behavior']}**. Effects are percentage-point "
        f"differences relative to **{config['control_variant']}**."
    )

    st.subheader("Full response distribution")
    chart_rows: list[dict[str, Any]] = []
    for row in overall_rows:
        for option in config["answer_options"]:
            chart_rows.append(
                {
                    "Variant": row["Variant"],
                    "Response": option,
                    "Share (%)": row[f"{option} rate (%)"] or 0.0,
                }
            )
    st.bar_chart(
        pd.DataFrame(chart_rows),
        x="Variant",
        y="Share (%)",
        color="Response",
        stack=False,
        height=430,
    )
    st.caption("Response shares are calculated within each variant and sum to 100%.")

    st.subheader("Effects by demographic segment")
    segment_rows = segment_result_rows(config, run_data["accepted_rows"])
    segment_df = pd.DataFrame(segment_rows)
    tabs = st.tabs(config["demographic_dimensions"])
    for tab, dimension in zip(tabs, config["demographic_dimensions"]):
        with tab:
            dimension_df = segment_df[segment_df["Demographic"] == dimension].copy()
            effect_df = dimension_df[dimension_df["Variant"] != config["control_variant"]][
                ["Segment", "Variant", "Effect vs control (pp)"]
            ].dropna()
            if not effect_df.empty:
                st.bar_chart(
                    effect_df,
                    x="Segment",
                    y="Effect vs control (pp)",
                    color="Variant",
                    stack=False,
                    height=380,
                )
            else:
                st.info("No segment effect can be calculated because a control cell has no valid responses.")

            table = dimension_df[
                ["Segment", "Variant", "Control", "Valid N", "Target count", "Target rate (%)", "Effect vs control (pp)"]
            ].copy()
            table["Target rate"] = table["Target rate (%)"].map(format_percentage)
            table["Effect vs control"] = table["Effect vs control (pp)"].map(format_pp)
            table = table[
                ["Segment", "Variant", "Control", "Valid N", "Target count", "Target rate", "Effect vs control"]
            ]
            st.dataframe(table, hide_index=True, use_container_width=True)
            if (dimension_df["Valid N"] < 10).any():
                st.warning(
                    "At least one segment-by-variant cell has fewer than 10 valid responses. "
                    "Treat those differences as highly exploratory."
                )

    with st.expander("Check demographic balance across variants"):
        balance_df = pd.DataFrame(
            profile_balance_rows(config, planned_jobs, run_data["accepted_rows"])
        )
        st.dataframe(balance_df, hide_index=True, use_container_width=True)
        st.caption(
            "The plan reproduces each selected marginal demographic distribution separately "
            "inside every variant. Demographic dimensions are sampled independently."
        )

    st.subheader("Usage & run quality")
    usage_columns = st.columns(4)
    usage_columns[0].metric("Input tokens", f"{run_data['input_tokens']:,}")
    usage_columns[1].metric("Output tokens", f"{run_data['output_tokens']:,}")
    usage_columns[2].metric(
        "Total tokens",
        f"{run_data['input_tokens'] + run_data['output_tokens']:,}",
    )
    usage_columns[3].metric(
        "Calculated API cost",
        f"${run_data['actual_cost']:,.4f}" if run_data["actual_cost"] is not None else "Unavailable",
    )

    quality_columns = st.columns(4)
    quality_columns[0].metric(
        "Invalid model outputs",
        f"{run_data['invalid_outputs']:,}",
        help="Successful completions that did not map to an allowed response code or option.",
    )
    quality_columns[1].metric(
        "API-failed attempts",
        f"{run_data['api_failed_attempts']:,}",
        help="Slot attempts that still failed after internal HTTP retries.",
    )
    quality_columns[2].metric(
        "Temporary API retries",
        f"{run_data['retry_events']:,}",
    )
    quality_columns[3].metric(
        "Simulation attempts",
        f"{run_data['simulation_attempts']:,}",
        help="Valid responses plus invalid and API-failed slot attempts.",
    )

    invalid_rows = [
        row for row in run_data.get("rows", []) if row.get("status") == "invalid_output"
    ]
    if invalid_rows:
        with st.expander("Inspect invalid model outputs"):
            diagnostic_rows = []
            for row in invalid_rows[:25]:
                diagnostic_rows.append(
                    {
                        "Attempt": row.get("attempt_id"),
                        "Variant": row.get("variant"),
                        "Raw completion": row.get("raw_completion", ""),
                        "Finish reason": row.get("finish_reason"),
                    }
                )
            st.dataframe(pd.DataFrame(diagnostic_rows), hide_index=True, use_container_width=True)
            st.caption(
                "These are the first invalid completions returned by Featherless. "
                "They are shown for debugging only and are never counted as valid responses."
            )

    with st.expander("Technical run details"):
        st.write(f"Model: `{MODEL_ID}`")
        st.write(f"Run timestamp: `{run_data['run_timestamp']}`")
        st.write(f"Plan seed: `{config['plan_seed']}`")
        st.write(f"Parallel requests used: **up to {run_data['workers']}**")
        st.write(f"Underlying HTTP requests: **{run_data['http_requests']:,}**")
        st.write(f"Temperature: `{config['settings']['temperature']}`")
        st.write(f"Top-p: `{config['settings']['top_p']}`")
        st.write(
            f"Top-k: `{config['settings']['top_k']}`"
            if config["settings"]["use_top_k"]
            else "Top-k: **disabled / provider default**"
        )
        st.write(f"Response option structure: `{config['response_option_structure']}`")
        st.write(f"Maximum generated tokens per response: `{MAX_NEW_TOKENS}`")
        st.write(f"Maximum attempts per respondent slot: `{MAX_SLOT_ATTEMPTS}`")

    st.subheader("Export")
    if excel_bytes:
        filename = f"{safe_filename(config['experiment_name'])}_socrates_results.xlsx"
        st.download_button(
            "Download Excel results",
            data=excel_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )
    else:
        st.warning("The run completed, but the Excel workbook could not be generated.")

    st.caption(
        "These are stochastic Socrates model simulations for hypothesis screening and "
        "experiment design. They are not observations from human participants and should "
        "not be treated as a substitute for a real-world experiment."
    )


# Header and API key ---------------------------------------------------------
logo_path = Path("logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=130)

st.caption("Behavioral intervention simulation with Socrates Qwen2.5 14B SFT")
st.title(APP_NAME)
st.write(
    "Compare up to five customer-touchpoint variants, simulate individual respondents "
    "from supported demographic profiles, and inspect overall and segment-level "
    "behavioral signals before field testing."
)

api_key = get_api_key()
if not api_key:
    st.error(
        "The Featherless API key is not available in this environment. Add "
        "`FEATHERLESS_API_KEY` to this Streamlit app's Secrets settings."
    )
    st.stop()

st.divider()


# 1. Experiment context -----------------------------------------------------
st.header("1. Experiment context")
experiment_name = st.text_input(
    "Experiment name",
    placeholder="e.g. Motor renewal email test",
)
context_left, context_right = st.columns(2)
with context_left:
    line_of_business = st.text_input(
        "Line of business (LOB)",
        placeholder="e.g. Motor insurance",
    )
with context_right:
    customer_journey_type = st.text_input(
        "Customer journey type",
        placeholder="e.g. Renewal, cancellation prevention, quote-to-bind",
    )
context = st.text_area(
    "Customer and experiment context",
    placeholder="""Describe the customer situation and the point in the journey.

Example: The customer has held a motor policy for four years. Renewal is due in 30 days and the premium will increase by 9%. The customer has not made a claim in the last year. Each variant below is the complete renewal email they receive.""",
    height=170,
)


# 2. Intervention variants -------------------------------------------------
st.header("2. Intervention variants")
variant_setup_left, variant_setup_right = st.columns(2)
with variant_setup_left:
    number_variants = int(
        st.selectbox("Number of variants", [1, 2, 3, 4, 5], index=1)
    )
with variant_setup_right:
    total_simulations = int(
        st.number_input(
            "Total valid simulated respondents",
            min_value=1,
            max_value=MAX_SIMULATIONS,
            value=100,
            step=1,
            help="The app attempts to collect this many valid individual Socrates responses.",
        )
    )

default_names = ["Control", "Variant B", "Variant C", "Variant D", "Variant E"]
default_allocations = equal_allocations(number_variants)
variants_input: list[dict[str, Any]] = []
for index in range(number_variants):
    st.markdown(f"#### Variant {index + 1}")
    content_column, allocation_column = st.columns([4, 1])
    with content_column:
        name = st.text_input(
            f"Variant {index + 1} name",
            value=default_names[index],
            key=f"variant_name::{number_variants}::{index}",
        )
        text = st.text_area(
            f"Variant {index + 1} content",
            placeholder="Paste the complete email, letter, SMS, message, choice architecture, or other touchpoint shown to the customer.",
            height=180,
            key=f"variant_text::{number_variants}::{index}",
        )
    with allocation_column:
        allocation = float(
            st.number_input(
                "Allocation %",
                min_value=0.0,
                max_value=100.0,
                value=float(default_allocations[index]),
                step=1.0,
                key=f"variant_allocation::{number_variants}::{index}",
            )
        )
    variants_input.append(
        {"name": name.strip(), "text": text.strip(), "allocation": allocation}
    )

allocation_total = sum(variant["allocation"] for variant in variants_input)
allocation_counts = (
    largest_remainder_counts(
        total_simulations,
        [variant["allocation"] for variant in variants_input],
    )
    if abs(allocation_total - 100.0) <= 0.01
    else [0] * number_variants
)
allocation_columns = st.columns(number_variants)
for index, variant in enumerate(variants_input):
    with allocation_columns[index]:
        st.metric(
            variant["name"] or f"Variant {index + 1}",
            f"{variant['allocation']:.0f}%",
            f"{allocation_counts[index]:,} respondents" if abs(allocation_total - 100.0) <= 0.01 else None,
        )
if abs(allocation_total - 100.0) > 0.01:
    st.error(f"Variant allocations currently total {allocation_total:.1f}%. They must total 100%.")

variant_names = [variant["name"] for variant in variants_input if variant["name"]]
control_variant = st.selectbox(
    "Control / baseline variant",
    options=variant_names if variant_names else ["Control"],
    index=0,
    help="All overall and segment-level treatment effects are calculated relative to this variant.",
)


# 3. Behavioral outcome -----------------------------------------------------
st.header("3. Behavioral outcome")
outcome_question = st.text_area(
    "Question asked after the customer sees the assigned variant",
    placeholder="e.g. What would you be most likely to do next?",
    height=100,
)

response_option_structure = st.radio(
    "Response option structure",
    options=list(RESPONSE_STRUCTURE_OPTIONS),
    index=0,
    horizontal=True,
    help=(
        "Categorical options may be counterbalanced across prompt-facing response codes, "
        "but every answer keeps a fixed canonical internal identity for analysis. "
        "Ordered / Likert options always remain in the exact order entered below and "
        "use the same code-to-option mapping in every respondent prompt."
    ),
)

answer_options_text = st.text_area(
    "Answer options — one option per line",
    value="Follow the call to action now\nFollow the call to action later\nContact the insurer\nCompare alternatives\nCancel the policy\nTake no action",
    height=170,
    help=(
        "Enter one response option per line and do not add response-code labels. "
        "The entered list defines the fixed canonical answer identities used in analysis. "
        "Categorical mode may vary the prompt-facing code assignment; ordered / Likert "
        "mode preserves this exact order and mapping in every prompt."
    ),
)
answer_options = parse_answer_options(answer_options_text)

# The target-behavior selector must always reflect the CURRENT answer options.
# Streamlit can preserve widget state across reruns, so a fixed selectbox key can
# otherwise leave an old value visible after the answer-option list is edited.
# We therefore give the widget a key derived from the current option list while
# separately remembering the researcher's last valid selection.
if answer_options:
    options_signature = hashlib.sha256(
        "\n".join(answer_options).encode("utf-8")
    ).hexdigest()[:12]
    target_widget_key = f"target_behavior::{options_signature}"

    previous_widget_key = st.session_state.get("_target_behavior_widget_key")
    if previous_widget_key and previous_widget_key != target_widget_key:
        st.session_state.pop(previous_widget_key, None)
    st.session_state["_target_behavior_widget_key"] = target_widget_key

    previous_target = st.session_state.get("_target_behavior_value")
    target_index = (
        answer_options.index(previous_target)
        if previous_target in answer_options
        else 0
    )

    target_behavior = st.selectbox(
        "Target behavior",
        options=answer_options,
        index=target_index,
        key=target_widget_key,
        help=(
            "Choose the current answer option whose rate and effect versus control "
            "should be highlighted. This list updates automatically when you edit "
            "the answer options above."
        ),
    )
    st.session_state["_target_behavior_value"] = target_behavior
else:
    target_behavior = None
    st.selectbox(
        "Target behavior",
        options=["Add answer options above"],
        index=0,
        disabled=True,
        key="target_behavior_empty",
        help="Add at least two answer options above before selecting a target behavior.",
    )

settings = render_advanced_settings()


# 4. Demographic simulation ------------------------------------------------
st.header("4. Demographic simulation")
st.caption(
    "Only demographic fields explicitly used in the Socrates model prompt/paper are "
    "available here. Location is intentionally excluded."
)
use_us_preset = st.toggle(
    "Use representative U.S. sample preset (approximate marginals)",
    value=True,
    help=US_PRESET_NOTE,
)
selected_dimensions = st.multiselect(
    "Demographic dimensions included in each respondent prompt",
    options=list(DEMOGRAPHIC_DEFINITIONS.keys()),
    default=list(DEFAULT_DEMOGRAPHIC_DIMENSIONS),
)
if use_us_preset:
    st.info(US_PRESET_NOTE)
    with st.expander("U.S. preset source notes"):
        st.write(
            "The rounded defaults are informed by recent U.S. Census Bureau ACS and "
            "BLS public estimates and mapped to Socrates-compatible categories."
        )
        for source in US_PRESET_SOURCES:
            st.code(source, language=None)
else:
    st.warning(
        "Custom percentages are interpreted as independent marginal distributions. "
        "They do not define a joint demographic population."
    )

demographic_distributions, demographic_errors = render_demographic_controls(
    selected_dimensions,
    use_us_preset,
)


# Build current configuration and validation -------------------------------
errors = validate_inputs(
    experiment_name=experiment_name,
    line_of_business=line_of_business,
    customer_journey_type=customer_journey_type,
    context=context,
    variants=variants_input,
    allocation_total=allocation_total,
    allocation_counts=allocation_counts,
    outcome_question=outcome_question,
    answer_options=answer_options,
    response_option_structure=response_option_structure,
    target_behavior=target_behavior if target_behavior in answer_options else None,
    demographic_dimensions=selected_dimensions,
    demographic_errors=demographic_errors,
    total_simulations=total_simulations,
    control_variant=control_variant,
)

current_config = None
current_fingerprint = None
if not errors:
    current_config = build_config(
        experiment_name=experiment_name,
        line_of_business=line_of_business,
        customer_journey_type=customer_journey_type,
        context=context,
        variants=variants_input,
        allocation_counts=allocation_counts,
        outcome_question=outcome_question,
        answer_options=answer_options,
        response_option_structure=response_option_structure,
        target_behavior=target_behavior,
        total_simulations=total_simulations,
        control_variant=control_variant,
        use_us_preset=use_us_preset,
        demographic_dimensions=selected_dimensions,
        demographic_distributions=demographic_distributions,
        settings=settings,
    )
    current_fingerprint = config_fingerprint(current_config)


# Lightweight prompt template preview before review ------------------------
with st.expander("Preview Socrates prompt template"):
    if current_config:
        preview_rng = random.Random(20260817)

        preview_profile = generate_profiles(
            1,
            current_config["demographic_dimensions"],
            current_config["demographic_distributions"],
            preview_rng,
        )[0]

        preview_tabs = st.tabs(
            [variant["name"] for variant in current_config["variants"]]
        )

        for tab, variant in zip(
            preview_tabs,
            current_config["variants"],
        ):
            with tab:
                preview_option_order = option_orders_for_structure(
                    current_config["answer_options"],
                    1,
                    preview_rng,
                    current_config["response_option_structure"],
                )[0]

                user_prompt, mapping = build_user_prompt(
                    current_config,
                    variant,
                    preview_profile["values"],
                    preview_option_order,
                )

                st.markdown("**System message**")
                st.code(SYSTEM_PROMPT, language=None)

                st.markdown("**Example user message**")
                st.code(user_prompt, language=None)

                if (
                    current_config["response_option_structure"]
                    == RESPONSE_STRUCTURE_ORDERED
                ):
                    st.caption(
                        "Ordered mode: the paid run preserves this exact answer order and "
                        "the same response-code mapping for every respondent. Internal "
                        "canonical answer identities are also fixed."
                    )
                else:
                    st.caption(
                        "Categorical mode: prompt-facing response-code assignments may vary "
                        "by respondent for counterbalancing. Each returned code is decoded "
                        "through that respondent's stored mapping to a fixed canonical answer "
                        "identity before any result is calculated."
                    )

    else:
        st.caption(
            "Complete the required fields to preview the prompt template."
        )
# Review and cost estimate --------------------------------------------------
review_clicked = st.button(
    "Review experiment, prompts & estimate cost",
    type="primary",
    use_container_width=True,
)

if review_clicked:
    if errors:
        for error in errors:
            st.error(error)
    else:
        assert current_config is not None
        assert current_fingerprint is not None
        plan_seed = secrets.randbits(63)
        prepared_config = dict(current_config)
        prepared_config["plan_seed"] = plan_seed
        with st.spinner("Building the respondent plan and estimating Featherless usage..."):
            planned_jobs = build_simulation_plan(prepared_config, plan_seed)
            runtime_info = get_runtime_info(api_key)
            estimate = estimate_run(api_key, planned_jobs, runtime_info)

        context_limit_error = None
        effective_context = runtime_info.get("effective_context_length")
        if effective_context:
            largest_estimate = max(
                heuristic_token_count(
                    serialize_messages_for_tokenizer(job["system_prompt"], job["user_prompt"])
                )
                for job in planned_jobs
            )
            if largest_estimate + MAX_NEW_TOKENS > int(effective_context):
                context_limit_error = (
                    f"At least one prompt is estimated at {largest_estimate:,} tokens, "
                    f"which may exceed the effective Featherless context limit of "
                    f"{int(effective_context):,} tokens. Shorten the context or variants."
                )

        st.session_state["prepared_run"] = {
            "fingerprint": current_fingerprint,
            "config": prepared_config,
            "planned_jobs": planned_jobs,
            "runtime_info": runtime_info,
            "estimate": estimate,
            "segment_warning": planned_segment_warning(prepared_config, planned_jobs),
            "context_limit_error": context_limit_error,
        }

prepared = st.session_state.get("prepared_run")
if prepared and current_fingerprint != prepared["fingerprint"]:
    st.info(
        "The experiment has changed since the last review. Review the experiment and "
        "cost again before running."
    )

if prepared and current_fingerprint == prepared["fingerprint"]:
    prepared_config = prepared["config"]
    planned_jobs = prepared["planned_jobs"]
    runtime_info = prepared["runtime_info"]
    estimate = prepared["estimate"]

    st.header("Reviewed run plan")
    plan_columns = st.columns(4)
    plan_columns[0].metric(
        "Target valid respondents",
        f"{prepared_config['total_simulations']:,}",
    )
    plan_columns[1].metric("Estimated input tokens", f"{estimate['prompt_tokens']:,}")
    plan_columns[2].metric("Estimated output tokens", f"{estimate['completion_tokens']:,}")
    plan_columns[3].metric(
        "Base estimated API cost",
        f"~${estimate['base_cost']:,.4f}" if estimate["base_cost"] is not None else "Unavailable",
    )

    if estimate["base_cost"] is not None:
        st.caption(
            "The base estimate assumes one valid completion per planned respondent and "
            f"approximately {ESTIMATED_COMPLETION_TOKENS_PER_RESPONSE} output tokens. "
            f"The conservative {MAX_SLOT_ATTEMPTS}× safety-ceiling estimate is "
            f"${estimate['safety_ceiling_cost']:,.4f}."
        )
    if estimate["used_fallback"]:
        st.warning(
            "At least one estimate used a character-based token fallback because the "
            "Featherless tokenizer was unavailable or returned an unusable result."
        )
    if runtime_info.get("error"):
        st.warning(
            "Live Featherless model/plan details could not be retrieved. The app can still "
            "attempt the run sequentially, but cost and availability checks may be unavailable."
        )
    if runtime_info.get("available_on_current_plan") is False:
        st.error(
            "Featherless reports that this Socrates model is not available on the current API plan."
        )
    if prepared.get("context_limit_error"):
        st.error(prepared["context_limit_error"])
    if prepared.get("segment_warning"):
        st.warning(prepared["segment_warning"])

    with st.expander("Token estimate by variant"):
        estimate_df = pd.DataFrame(estimate["condition_estimates"]).rename(
            columns={
                "variant": "Variant",
                "sampled_prompts": "Prompts sampled",
                "estimated_tokens_per_prompt": "Estimated tokens / prompt",
                "target_valid_n": "Target valid N",
                "estimated_input_tokens": "Estimated input tokens",
                "token_source": "Token source",
            }
        )
        st.dataframe(estimate_df, hide_index=True, use_container_width=True)

    with st.expander("Preview exact planned prompts", expanded=True):
        if prepared_config["response_option_structure"] == RESPONSE_STRUCTURE_ORDERED:
            st.success(
                "Ordered mapping integrity check passed: every planned respondent uses the "
                "entered answer order and the identical code-to-option mapping."
            )
        else:
            st.info(
                "Categorical mode: prompt-facing code assignments are counterbalanced by design. "
                "The canonical option ID shown below is fixed and is what the results logic uses."
            )
        st.caption(
            "The examples below are actual respondent slots from this prepared run. The "
            "same plan, profiles, assignments and answer mappings will be used when you click Run."
        )
        prompt_tabs = st.tabs([variant["name"] for variant in prepared_config["variants"]])
        for tab, variant in zip(prompt_tabs, prepared_config["variants"]):
            with tab:
                example_job = next(job for job in planned_jobs if job["variant"] == variant["name"])
                profile_df = pd.DataFrame(
                    [
                        {
                            "Demographic": dimension,
                            "Segment": example_job["profile_segments"][dimension],
                            "Prompt value": example_job["profile_values"][dimension],
                        }
                        for dimension in prepared_config["demographic_dimensions"]
                    ]
                )
                st.dataframe(profile_df, hide_index=True, use_container_width=True)
                mapping_df = pd.DataFrame(
                    [
                        {
                            "Prompt response code": code,
                            "Canonical option ID": example_job["code_to_canonical_id"][code],
                            "Answer option": option,
                        }
                        for code, option in example_job["code_to_option"].items()
                    ]
                )
                st.dataframe(mapping_df, hide_index=True, use_container_width=True)
                st.markdown("**System message**")
                st.code(example_job["system_prompt"], language=None)
                st.markdown("**User message**")
                st.code(example_job["user_prompt"], language=None)

    run_disabled = bool(
        runtime_info.get("available_on_current_plan") is False
        or prepared.get("context_limit_error")
    )
    run_clicked = st.button(
        "Run experiment",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    )

    if run_clicked:
        run_data = execute_experiment(
            prepared_config,
            planned_jobs,
            api_key,
            runtime_info,
        )
        excel_bytes = None
        try:
            excel_bytes = create_excel_export(prepared_config, planned_jobs, run_data)
        except Exception as exc:
            st.warning(f"The simulation completed, but Excel export generation failed: {exc}")

        st.session_state["last_run"] = {
            "fingerprint": prepared["fingerprint"],
            "config": prepared_config,
            "planned_jobs": planned_jobs,
            "run_data": run_data,
            "excel_bytes": excel_bytes,
        }

last_run = st.session_state.get("last_run")
if last_run:
    st.divider()
    st.header("Results")
    render_results(
        last_run["config"],
        last_run["planned_jobs"],
        last_run["run_data"],
        last_run.get("excel_bytes"),
    )
