"""Socrates Experiment Runner: multi-question study UI and saved history."""
from __future__ import annotations

import hashlib
import os
import random
import re
import secrets
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from socrates_core import (
    APP_NAME, DEFAULT_DEMOGRAPHIC_DIMENSIONS, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_USE_TOP_K, DEMOGRAPHIC_DEFINITIONS,
    MAX_APP_WORKERS, MAX_NEW_TOKENS, MAX_QUESTIONS, MAX_RESPONSE_OPTIONS,
    MAX_SIMULATIONS, MAX_SLOT_ATTEMPTS, MAX_VARIANTS, MODEL_ID,
    RESPONSE_STRUCTURE_CATEGORICAL, RESPONSE_STRUCTURE_OPTIONS,
    RESPONSE_STRUCTURE_ORDERED, SYSTEM_PROMPT, US_PRESET_NOTE, US_PRESET_SOURCES,
    build_run_timestamp, build_simulation_plan, build_user_prompt,
    compact_attempt, config_fingerprint, create_excel_export,
    distributions_from_preset, generate_profiles, get_runtime_info,
    largest_remainder_counts, option_orders_for_structure, overall_result_rows,
    parse_answer_options, profile_balance_rows, run_one, segment_result_rows,
    validate_distribution,
)
from socrates_history import HistoryError, list_runs, load_run, save_run

st.set_page_config(page_title=APP_NAME, layout='wide')
st.markdown('''
<style>
button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * {
  color: #000000 !important;
}
</style>''', unsafe_allow_html=True)


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.getenv(name) or '').strip()


def get_history_settings() -> tuple[str, str]:
    return get_secret('SOCRATES_HISTORY_REPO'), get_secret('SOCRATES_HISTORY_TOKEN')


def require_team_password_if_history_is_enabled() -> None:
    repo, token = get_history_settings()
    if not (repo or token):
        return
    password = get_secret('SOCRATES_APP_PASSWORD')
    if not repo or not token or not password:
        st.error(
            'GitHub history is partly configured. For shared history, set all three '
            'Streamlit secrets: SOCRATES_HISTORY_REPO, SOCRATES_HISTORY_TOKEN and '
            'SOCRATES_APP_PASSWORD. Until then, shared history is disabled.'
        )
        st.stop()
    if st.session_state.get('_team_authenticated'):
        return
    st.title('Team access')
    entered = st.text_input('Shared app password', type='password')
    if st.button('Sign in', type='primary'):
        if secrets.compare_digest(entered, password):
            st.session_state['_team_authenticated'] = True
            st.rerun()
        else:
            st.error('Incorrect password.')
    st.stop()


def equal_allocations(number_variants: int) -> list[int]:
    result = [100 // number_variants] * number_variants
    for index in range(100 - sum(result)):
        result[index] += 1
    return result


def safe_filename(value: str) -> str:
    return (re.sub(r'[^A-Za-z0-9_-]+', '_', value).strip('_')[:65] or 'socrates')


def fmt_pct(value: Any) -> str:
    return f'{float(value):.1f}%' if value is not None and not pd.isna(value) else 'N/A'


def fmt_pp(value: Any) -> str:
    return f'{float(value):+.1f} pp' if value is not None and not pd.isna(value) else 'N/A'


def format_date(utc_iso: str) -> str:
    try:
        parsed = datetime.fromisoformat(utc_iso.replace('Z', '+00:00'))
        return parsed.astimezone(ZoneInfo('Europe/Zurich')).strftime('%d %b %Y, %H:%M %Z')
    except (ValueError, KeyError):
        return utc_iso


def render_advanced_settings() -> dict[str, Any]:
    defaults = {
        'socrates_temperature': DEFAULT_TEMPERATURE,
        'socrates_top_p': DEFAULT_TOP_P,
        'socrates_use_top_k': DEFAULT_USE_TOP_K,
        'socrates_top_k': DEFAULT_TOP_K,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    with st.expander('Advanced model settings'):
        st.caption('Keep model sampling settings fixed across all survey questions.')
        if st.button('Reset to paper-aligned defaults', key='reset_socrates_settings'):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()
        left, right = st.columns(2)
        with left:
            temperature = float(st.number_input(
                'Temperature', min_value=0.0, max_value=2.0,
                step=0.1, key='socrates_temperature',
                help='The model paper used 0.6.',
            ))
        with right:
            top_p = float(st.number_input(
                'Top-p', min_value=0.01, max_value=1.0,
                step=0.05, key='socrates_top_p',
                help='The model paper used 0.9.',
            ))
        use_top_k = st.checkbox('Enable top-k', key='socrates_use_top_k')
        top_k = int(st.number_input(
            'Top-k', min_value=1, max_value=500, step=1,
            disabled=not use_top_k, key='socrates_top_k',
        ))
    return {
        'temperature': temperature, 'top_p': top_p,
        'use_top_k': use_top_k, 'top_k': top_k if use_top_k else None,
    }


def render_demographics(dimensions: Sequence[str], preset: bool) -> tuple[dict[str, dict[str, float]], list[str]]:
    distributions: dict[str, dict[str, float]] = {}
    errors: list[str] = []
    for dimension in dimensions:
        definition = DEMOGRAPHIC_DEFINITIONS[dimension]
        categories = list(definition['categories'])
        shares = list(definition['preset'])
        with st.expander(dimension, expanded=dimension in ('Age', 'Gender')):
            if preset:
                st.dataframe(pd.DataFrame({'Category': categories, 'Share (%)': shares}),
                             hide_index=True, use_container_width=True)
                distribution = dict(zip(categories, map(float, shares)))
            else:
                st.caption('Percentages must total 100%.')
                distribution = {}
                columns = st.columns(2)
                for index, (category, share) in enumerate(zip(categories, shares)):
                    key = f'demo::{dimension}::{index}'
                    st.session_state.setdefault(key, float(share))
                    with columns[index % 2]:
                        distribution[category] = float(st.number_input(
                            category, min_value=0.0, max_value=100.0,
                            step=1.0, key=key,
                        ))
                total = sum(distribution.values())
                st.caption(f'Total: {total:.1f}%')
                if not validate_distribution(distribution):
                    errors.append(f'{dimension} percentages total {total:.1f}%, not 100%.')
            distributions[dimension] = distribution
    return distributions, errors


DEFAULT_CATEGORICAL_ANSWERS = (
    'Follow the call to action now\nFollow the call to action later\n'
    'Contact the insurer\nCompare alternatives\nTake no action'
)
DEFAULT_LIKERT_ANSWERS = (
    'Strongly disagree\nDisagree\nNeither agree nor disagree\n'
    'Agree\nStrongly agree'
)


def on_response_structure_changed(qid: str) -> None:
    structure = st.session_state.get(f'{qid}::structure')
    value = st.session_state.get(f'{qid}::answers', '')
    if structure == RESPONSE_STRUCTURE_ORDERED and value == DEFAULT_CATEGORICAL_ANSWERS:
        st.session_state[f'{qid}::answers'] = DEFAULT_LIKERT_ANSWERS
    elif structure == RESPONSE_STRUCTURE_CATEGORICAL and value == DEFAULT_LIKERT_ANSWERS:
        st.session_state[f'{qid}::answers'] = DEFAULT_CATEGORICAL_ANSWERS


def optional_selector(label: str, values: Sequence[str], key: str, help_text: str) -> str | None:
    """Keep state coherent when answer options or variant names are edited."""
    signature = hashlib.sha256('\n'.join(values).encode('utf-8')).hexdigest()[:16]
    state_key = f'{key}::{signature}'
    last_key = f'{key}::_last_key'
    previous_key = st.session_state.get(last_key)
    if previous_key and previous_key != state_key:
        st.session_state.pop(previous_key, None)
    st.session_state[last_key] = state_key
    previous_value = st.session_state.get(f'{key}::_value')
    if previous_value not in values:
        previous_value = None
    items: list[str | None] = [None, *values]
    selected = st.selectbox(
        label, options=items, index=items.index(previous_value),
        format_func=lambda item: 'None / not selected' if item is None else item,
        key=state_key, help=help_text,
    )
    st.session_state[f'{key}::_value'] = selected
    return selected


def render_question(question_index: int, total_n: int) -> dict[str, Any]:
    qid = f'Q{question_index + 1}'
    st.markdown(f'#### {qid} - Question & intervention setup')
    name = st.text_input(
        'Question name', value=f'Question {question_index + 1}',
        key=f'{qid}::name', max_chars=100,
    ).strip()
    has_intervention = st.checkbox(
        'Include an intervention / stimulus for this question',
        value=True, key=f'{qid}::has_intervention',
        help='Turn off for a standalone survey question with no message or touchpoint.',
    )
    variants: list[dict[str, Any]] = []
    if has_intervention:
        number_variants = int(st.selectbox(
            'Number of intervention variants',
            options=list(range(1, MAX_VARIANTS + 1)),
            index=1, key=f'{qid}::number_variants',
        ))
        allocations = equal_allocations(number_variants)
        for index in range(number_variants):
            with st.expander(f'Variant {index + 1}', expanded=index == 0):
                text_col, allocation_col = st.columns([4, 1])
                with text_col:
                    variant_name = st.text_input(
                        'Variant name', value='Control' if index == 0 else f'Variant {chr(65 + index)}',
                        key=f'{qid}::v{number_variants}::{index}::name',
                    ).strip()
                    variant_text = st.text_area(
                        'Intervention content', height=140,
                        placeholder='Complete message, email, stimulus, or other material shown.',
                        key=f'{qid}::v{number_variants}::{index}::text',
                    ).strip()
                with allocation_col:
                    allocation = float(st.number_input(
                        'Allocation (%)', min_value=0.0, max_value=100.0,
                        value=float(allocations[index]), step=1.0,
                        key=f'{qid}::v{number_variants}::{index}::allocation',
                    ))
                variants.append({
                    'name': variant_name, 'text': variant_text, 'allocation': allocation,
                })
    else:
        st.caption('All respondents see the same context and question. No intervention is added to the prompt.')
        variants = [{'name': 'No intervention', 'text': '', 'allocation': 100.0}]

    allocation_sum = sum(variant['allocation'] for variant in variants)
    counts = largest_remainder_counts(total_n, [v['allocation'] for v in variants]) if abs(allocation_sum-100) <= .01 else [0] * len(variants)
    for variant, count in zip(variants, counts):
        variant['count'] = count
    if has_intervention:
        st.caption(f'Allocation: {allocation_sum:.1f}% | ' +
                   ' | '.join(f"{v['name'] or 'Unnamed'}: {v['count']} respondents" for v in variants))

    st.markdown('#### Behavioral outcome')
    outcome_question = st.text_area(
        'Survey question / behavioral outcome', height=105,
        placeholder='What would you be most likely to do, think or choose?',
        key=f'{qid}::outcome_question',
    ).strip()
    structure = st.radio(
        'Response option structure', options=list(RESPONSE_STRUCTURE_OPTIONS),
        horizontal=True, key=f'{qid}::structure',
        on_change=on_response_structure_changed, args=(qid,),
        help='Ordered/Likert: fixed ordering and codes. Categorical: counterbalanced prompt codes, stable analysis identities.',
    )
    answer_text = st.text_area(
        'Answer options (one per line)',
        value=DEFAULT_CATEGORICAL_ANSWERS,
        height=145,
        key=f'{qid}::answers',
        help='The order entered here defines permanent canonical identities; it is never altered for ordered questions.',
    )
    options = parse_answer_options(answer_text)
    selector_left, selector_right = st.columns(2)
    with selector_left:
        target = optional_selector(
            'Target behavior (optional)', options, f'{qid}::target',
            'Leave blank to analyse the full response distribution without a focal response.',
        ) if options else None
    with selector_right:
        control = optional_selector(
            'Control / baseline (optional)', [v['name'] for v in variants if v['name']],
            f'{qid}::control',
            'Only used for percentage-point effects when a target behavior is also selected.',
        ) if len(variants) > 1 else None
    return {
        'id': qid, 'name': name, 'has_intervention': has_intervention,
        'variants': variants, 'outcome_question': outcome_question,
        'answer_options': options, 'response_option_structure': structure,
        'target_behavior': target, 'control_variant': control,
        '_allocation_sum': allocation_sum,
    }


def validate_study(
    experiment_name: str, context: str, total_n: int, questions: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str], errors: Sequence[str],
) -> list[str]:
    problems = list(errors)
    if not experiment_name.strip():
        problems.append('Give the experiment a name for its history and exports.')
    if not context.strip():
        problems.append('Add the experiment and customer context.')
    if not dimensions:
        problems.append('Select at least one demographic dimension.')
    if not 1 <= total_n <= MAX_SIMULATIONS:
        problems.append('Number of respondents per question is out of range.')
    if len({q['name'].casefold() for q in questions if q['name']}) != len(questions):
        problems.append('Give all survey questions unique names.')
    for question in questions:
        tag = question['id']
        if not question['name']:
            problems.append(f'{tag}: give the question a name.')
        if not question['outcome_question']:
            problems.append(f'{tag}: add the survey question.')
        options = question['answer_options']
        if not 2 <= len(options) <= MAX_RESPONSE_OPTIONS:
            problems.append(f'{tag}: provide 2 to {MAX_RESPONSE_OPTIONS} answer options.')
        if len({option.casefold() for option in options}) != len(options):
            problems.append(f'{tag}: answer options must be unique.')
        if question['response_option_structure'] not in RESPONSE_STRUCTURE_OPTIONS:
            problems.append(f'{tag}: invalid response structure.')
        if question.get('target_behavior') and question['target_behavior'] not in options:
            problems.append(f'{tag}: select a valid target behavior.')
        variants = question['variants']
        names = [variant['name'] for variant in variants]
        if any(not name for name in names) or len({name.casefold() for name in names}) != len(names):
            problems.append(f'{tag}: variant names must be present and unique.')
        if question['has_intervention'] and any(not variant['text'] for variant in variants):
            problems.append(f'{tag}: add intervention content to every enabled variant.')
        if abs(question['_allocation_sum'] - 100) > .01:
            problems.append(f'{tag}: variant allocations must total 100%.')
        elif any(variant['count'] < 1 for variant in variants):
            problems.append(f'{tag}: each variant must receive at least one respondent.')
        if question.get('control_variant') and question['control_variant'] not in names:
            problems.append(f'{tag}: invalid control variant.')
    return problems


def make_config(
    experiment_name: str, context: str, total_n: int,
    questions: Sequence[Mapping[str, Any]], us_preset: bool,
    dimensions: Sequence[str], distributions: Mapping[str, Mapping[str, float]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        'experiment_name': experiment_name.strip(),
        'context': context.strip(),
        'total_simulations': int(total_n),
        'questions': [{key: value for key, value in q.items() if not key.startswith('_')}
                      for q in questions],
        'use_us_preset': bool(us_preset),
        'demographic_dimensions': list(dimensions),
        'demographic_distributions': {
            dimension: {category: float(share) for category, share in distributions[dimension].items()}
            for dimension in dimensions
        },
        'settings': dict(settings),
    }


def planned_small_cell_warning(config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> str | None:
    sizes = Counter(
        (job['question_id'], job['variant'], dim, job['profile_segments'][dim])
        for job in jobs for dim in config['demographic_dimensions']
    )
    if not sizes:
        return None
    smallest = min(sizes.values())
    if smallest < 10:
        return f'Some variant-by-demographic cells are small (minimum planned N={smallest}). Treat segment patterns as exploratory.'
    return None


def render_prompt_review(config: Mapping[str, Any], planned_jobs: Sequence[Mapping[str, Any]]) -> None:
    st.subheader('Exact planned prompts')
    st.caption('These are real slots from the prepared plan. The same slots and mappings will be used in the paid run.')
    lookup = {}
    for job in planned_jobs:
        lookup.setdefault((job['question_id'], job['variant']), job)
    tabs = st.tabs([f"{q['id']} - {q['name']}" for q in config['questions']])
    for tab, question in zip(tabs, config['questions']):
        with tab:
            variants = [variant['name'] for variant in question['variants']]
            selected = st.selectbox('Variant to preview', variants,
                                    key=f'preview_variant::{question["id"]}')
            job = lookup[(question['id'], selected)]
            mapping_rows = [
                {
                    'Response code': code,
                    'Canonical ID': job['code_to_canonical_id'][code],
                    'Answer option': option,
                }
                for code, option in job['code_to_option'].items()
            ]
            st.dataframe(pd.DataFrame(mapping_rows), hide_index=True, use_container_width=True)
            if question['response_option_structure'] == RESPONSE_STRUCTURE_ORDERED:
                st.caption('Ordered/Likert option order and codes are fixed for every respondent.')
            else:
                st.caption('Categorical codes may vary between respondents; canonical answer IDs stay fixed.')
            st.markdown('**System message**')
            st.code(job['system_prompt'], language=None)
            st.markdown('**User message**')
            st.code(job['user_prompt'], language=None)


def execute_experiment(
    config: Mapping[str, Any], planned_jobs: Sequence[Mapping[str, Any]],
    api_key: str, runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(planned_jobs)
    workers = max(1, min(MAX_APP_WORKERS, int(runtime_info.get('workers') or 1)))
    accepted_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    accepted_slots: set[str] = set()
    input_tokens = output_tokens = retry_events = http_requests = 0
    invalid_outputs = api_failed_attempts = 0
    actual_cost = 0.0
    cost_known = bool(runtime_info.get('pricing_available'))
    next_attempt_id = 1
    valid_by_question = Counter()

    st.subheader('Running experiment')
    bar = st.progress(0.0)
    columns = st.columns(4)
    good = columns[0].empty()
    invalid = columns[1].empty()
    api_fail = columns[2].empty()
    tokens = columns[3].empty()
    status = st.empty()
    last_update = 0.0

    def numbered(base_job: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal next_attempt_id
        job = dict(base_job)
        job['attempt_id'] = next_attempt_id
        next_attempt_id += 1
        return job

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        for base in planned_jobs:
            job = numbered(base)
            future = executor.submit(run_one, job, api_key, config['settings'], runtime_info)
            pending[future] = job
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **job, 'raw_completion': '', 'selected_code': None,
                        'selected_option': None, 'selected_option_id': None,
                        'status': 'api_error', 'error': str(exc),
                        'prompt_tokens': 0, 'completion_tokens': 0,
                        'cost': None, 'http_attempts': 0, 'retry_events': 0,
                        'finish_reason': None,
                    }
                compact = compact_attempt(result)
                all_results.append(compact)
                input_tokens += int(compact.get('prompt_tokens') or 0)
                output_tokens += int(compact.get('completion_tokens') or 0)
                retry_events += int(compact.get('retry_events') or 0)
                http_requests += int(compact.get('http_attempts') or 0)
                if compact.get('cost') is not None:
                    actual_cost += float(compact['cost'])
                slot_id = str(compact['slot_id'])
                if compact['status'] == 'valid' and slot_id not in accepted_slots:
                    accepted_slots.add(slot_id)
                    accepted_rows.append(compact)
                    valid_by_question[compact['question_id']] += 1
                else:
                    if compact['status'] == 'invalid_output':
                        invalid_outputs += 1
                    elif compact['status'] == 'api_error':
                        api_failed_attempts += 1
                    if slot_id not in accepted_slots and int(compact['slot_attempt']) < MAX_SLOT_ATTEMPTS:
                        replacement = dict(job)
                        replacement['slot_attempt'] = int(job['slot_attempt']) + 1
                        replacement = numbered(replacement)
                        replacement_future = executor.submit(
                            run_one, replacement, api_key, config['settings'], runtime_info
                        )
                        pending[replacement_future] = replacement
                now = time.time()
                if now - last_update >= .3 or len(accepted_rows) == total:
                    bar.progress(min(1.0, len(accepted_rows) / total))
                    good.metric('Valid responses', f'{len(accepted_rows):,} / {total:,}')
                    invalid.metric('Invalid outputs', f'{invalid_outputs:,}')
                    api_fail.metric('API failed attempts', f'{api_failed_attempts:,}')
                    tokens.metric('Tokens used', f'{input_tokens + output_tokens:,}')
                    qsummary = ' | '.join(
                        f"{q['id']}: {valid_by_question[q['id']]:,}/{config['total_simulations']:,}"
                        for q in config['questions']
                    )
                    status.caption(
                        f'{qsummary} | Up to {workers} parallel API requests | '
                        f'{retry_events} temporary API retry events'
                    )
                    last_update = now
    bar.progress(min(1.0, len(accepted_rows) / total))
    status.empty()
    all_results.sort(key=lambda row: int(row['attempt_id']))
    accepted_rows.sort(key=lambda row: (row['question_id'], row['respondent_id']))
    balanced = [
        row for question in config['questions']
        for row in profile_balance_rows(config, question, planned_jobs, accepted_rows)
    ]
    return {
        'rows': all_results, 'accepted_rows': accepted_rows,
        'balance_rows': balanced,
        'input_tokens': input_tokens, 'output_tokens': output_tokens,
        'actual_cost': actual_cost if cost_known else None,
        'invalid_outputs': invalid_outputs,
        'api_failed_attempts': api_failed_attempts,
        'retry_events': retry_events, 'http_requests': http_requests,
        'simulation_attempts': len(all_results),
        'valid_total': len(accepted_rows), 'total_target': total,
        'valid_by_question': dict(valid_by_question),
        'workers': workers, 'run_timestamp': build_run_timestamp(),
    }


def labelled_percent_chart(
    chart_rows: Sequence[Mapping[str, Any]],
    group_order: Sequence[str], variant_order: Sequence[str],
    group_label: str = 'Response',
) -> None:
    """Responsive grouped bars with percent values written at the end."""
    if not chart_rows:
        st.info('No valid responses to chart.')
        return
    df = pd.DataFrame(chart_rows)
    df['Label'] = df['Share (%)'].map(lambda value: f'{float(value):.1f}%')
    group_count, variant_count = len(group_order), len(variant_order)
    height = max(185, min(920, group_count * (variant_count * 27 + 13)))
    base = alt.Chart(df).encode(
        y=alt.Y(
            f'{group_label}:N', sort=list(group_order),
            axis=alt.Axis(title=None, labelLimit=270),
        ),
        x=alt.X(
            'Share (%):Q', scale=alt.Scale(domain=[0, 110]),
            axis=alt.Axis(title='Share of valid responses (%)', values=[0, 25, 50, 75, 100]),
        ),
        color=alt.Color('Variant:N', sort=list(variant_order),
                        legend=alt.Legend(title='Variant') if variant_count > 1 else None),
        tooltip=[
            alt.Tooltip(f'{group_label}:N', title=group_label),
            alt.Tooltip('Variant:N'),
            alt.Tooltip('Share (%):Q', format='.1f', title='Share (%)'),
        ],
    )
    if variant_count > 1:
        base = base.encode(yOffset=alt.YOffset('Variant:N', sort=list(variant_order)))
    bars = base.mark_bar(cornerRadiusEnd=3)
    labels = base.mark_text(align='left', baseline='middle', dx=5, fontSize=11).encode(
        text=alt.Text('Label:N'), color=alt.value('#8A97AD'),
    )
    chart = alt.layer(bars, labels).properties(height=height)
    st.altair_chart(chart, use_container_width=True)


def _distribution_long_rows(
    question: Mapping[str, Any], summary_rows: Sequence[Mapping[str, Any]],
    variant_filter: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {'Response': option, 'Variant': row['Variant'], 'Share (%)': row[f'{option} rate (%)']}
        for row in summary_rows
        if row['Valid N'] > 0 and (variant_filter is None or row['Variant'] == variant_filter)
        for option in question['answer_options']
    ]


def render_distribution_chart(
    question: Mapping[str, Any], summary_rows: Sequence[Mapping[str, Any]],
    key: str, segmented: bool = False,
) -> None:
    variant_names = [v['name'] for v in question['variants']]
    selected_variant = None
    if len(variant_names) > 2 or len(question['answer_options']) > 7:
        choice = st.selectbox(
            'Chart view', ['Compare all variants', *variant_names], key=key,
        )
        if choice != 'Compare all variants':
            selected_variant = choice
    visible_variants = [selected_variant] if selected_variant else variant_names
    rows = _distribution_long_rows(question, summary_rows, selected_variant)
    labelled_percent_chart(rows, question['answer_options'], visible_variants)
    st.caption(
        'Response percentages use valid answers within the selected segment and variant.'
        if segmented else 'Response percentages use valid answers within each variant.'
    )


def render_question_results(
    config: Mapping[str, Any], question: Mapping[str, Any],
    run_data: Mapping[str, Any],
) -> None:
    accepted = run_data['accepted_rows']
    overall = overall_result_rows(question, accepted)
    total_n = sum(row['Valid N'] for row in overall)
    target_n = sum(row['Target N'] for row in overall)
    st.markdown(f"**Survey question:** {question['outcome_question']}")
    st.caption(f'{total_n:,} of {target_n:,} valid simulated answers | {len(question["variants"])} variant(s)')
    if total_n == 0:
        st.warning('No valid responses were recorded for this question. Distributions are unavailable.')
        return
    target = question.get('target_behavior')
    control = question.get('control_variant')
    if target:
        st.subheader('Selected target response')
        st.caption(f'Target option: {target}')
        target_rows = [
            {'Response': row['Variant'], 'Variant': 'Target response',
             'Share (%)': row['Target rate (%)'] or 0.0}
            for row in overall if row['Valid N'] > 0
        ]
        labelled_percent_chart(
            target_rows,
            [v['name'] for v in question['variants']],
            ['Target response'],
        )
        focal = pd.DataFrame([{
            'Variant': row['Variant'], 'Valid N': row['Valid N'],
            'Target rate': fmt_pct(row['Target rate (%)']),
            **({'Effect vs control': fmt_pp(row['Effect vs control (pp)'])} if control else {}),
        } for row in overall])
        st.dataframe(focal, hide_index=True, use_container_width=True)
    st.subheader('Full response distribution')
    render_distribution_chart(question, overall, f'distribution::{question["id"]}')
    result_table = []
    for row in overall:
        result = {'Variant': row['Variant'], 'Valid N': row['Valid N']}
        for option in question['answer_options']:
            share = row[f'{option} rate (%)']
            count = row[f'{option} count']
            result[option] = f'{share:.1f}% ({count:,})' if share is not None else 'N/A'
        result_table.append(result)
    st.dataframe(pd.DataFrame(result_table), hide_index=True, use_container_width=True)
    if question['response_option_structure'] == RESPONSE_STRUCTURE_ORDERED:
        st.subheader('Ordered response summary')
        st.caption('Means and medians use ordinal option positions (1 = first answer option). Interpret as descriptive scores, not interval-scale effects.')
        ordinal = pd.DataFrame([{
            'Variant': row['Variant'], 'Valid N': row['Valid N'],
            f'Mean (1-{len(question["answer_options"])})': (
                round(row['Mean ordinal score'], 2) if row['Mean ordinal score'] is not None else None
            ),
            'Median': row['Median ordinal score'],
        } for row in overall])
        st.dataframe(ordinal, hide_index=True, use_container_width=True)

    st.subheader('Results by demographic segment')
    segments = segment_result_rows(config, question, accepted)
    tabs = st.tabs(list(config['demographic_dimensions']))
    for tab, dimension in zip(tabs, config['demographic_dimensions']):
        with tab:
            dimension_rows = [row for row in segments if row['Demographic'] == dimension]
            if any(row['Valid N'] < 10 for row in dimension_rows):
                st.caption('Some segment-by-variant cells have fewer than 10 valid responses. Differences are highly exploratory.')
            fields = ['Segment', 'Variant', 'Valid N']
            tabular_rows = []
            for row in dimension_rows:
                entry = {field: row[field] for field in fields}
                if target:
                    entry['Target rate'] = fmt_pct(row['Target rate (%)'])
                    if control:
                        entry['Effect vs control'] = fmt_pp(row['Effect vs control (pp)'])
                if question['response_option_structure'] == RESPONSE_STRUCTURE_ORDERED:
                    mean = row['Mean ordinal score']
                    entry['Mean rank'] = round(mean, 2) if mean is not None else None
                    entry['Median rank'] = row['Median ordinal score']
                for option in question['answer_options']:
                    entry[f'{option} (%)'] = fmt_pct(row[f'{option} rate (%)'])
                tabular_rows.append(entry)
            st.dataframe(pd.DataFrame(tabular_rows), hide_index=True, use_container_width=True)
            categories = list(config['demographic_distributions'][dimension])
            if target and len(categories) * len(question['variants']) <= 24:
                target_data = [{
                    'Response': row['Segment'], 'Variant': row['Variant'],
                    'Share (%)': row['Target rate (%)'] or 0.0,
                } for row in dimension_rows if row['Valid N'] > 0]
                st.caption('Target response share within each segment and intervention variant:')
                labelled_percent_chart(target_data, categories,
                                       [v['name'] for v in question['variants']])
            else:
                segment = st.selectbox(
                    'Inspect a segment', categories,
                    key=f'segment::{question["id"]}::{dimension}',
                )
                selected = [row for row in dimension_rows if row['Segment'] == segment]
                render_distribution_chart(
                    question, selected,
                    f'segment_distribution::{question["id"]}::{dimension}',
                    segmented=True,
                )
    with st.expander('Demographic balance across variants'):
        balance = [row for row in run_data.get('balance_rows', [])
                   if row['Question ID'] == question['id']]
        if balance:
            st.dataframe(pd.DataFrame(balance), hide_index=True, use_container_width=True)
        else:
            st.caption('Balance details are unavailable for this saved run.')
        st.caption('The same demographic profiles are reused across questions. Variant assignment is independently randomised and approximately balanced within each question.')


def render_usage_and_export(
    config: Mapping[str, Any], run_data: Mapping[str, Any],
    run_id: str, planned_jobs: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    with st.expander('Technical details and actual API cost'):
        left, right, third, fourth = st.columns(4)
        left.metric('Input tokens', f'{run_data["input_tokens"]:,}')
        right.metric('Output tokens', f'{run_data["output_tokens"]:,}')
        third.metric('API calls', f'{run_data["http_requests"]:,}')
        fourth.metric('Calculated actual cost',
                      f'${run_data["actual_cost"]:,.4f}' if run_data.get('actual_cost') is not None else 'Unavailable')
        st.caption('Cost is calculated from API usage and provider pricing when available, not a billing statement.')
        st.write(f'Model: `{MODEL_ID}` | Parallel workers: {run_data["workers"]} | Maximum attempts per slot: {MAX_SLOT_ATTEMPTS}')
        st.write(f'Temperature: {config["settings"]["temperature"]} | Top-p: {config["settings"]["top_p"]} | Top-k: {config["settings"]["top_k"] if config["settings"]["use_top_k"] else "disabled"}')
        st.write(f'Invalid outputs: {run_data["invalid_outputs"]:,} | API-failed attempts: {run_data["api_failed_attempts"]:,} | Temporary retries: {run_data["retry_events"]:,}')
        st.write(f'Run timestamp (UTC): `{run_data["run_timestamp"]}` | Plan seed: `{config["plan_seed"]}`')
        invalid_rows = [row for row in run_data['rows'] if row.get('status') == 'invalid_output']
        if invalid_rows:
            st.caption('First invalid model outputs (never counted as valid):')
            st.dataframe(pd.DataFrame([{
                'Question': row['question_id'], 'Variant': row['variant'],
                'Raw completion': row.get('raw_completion'),
            } for row in invalid_rows[:25]]), hide_index=True, use_container_width=True)

    st.subheader('Export')
    key = f'excel::{run_id}'
    if st.button('Generate Excel workbook', key=f'generate_{run_id}'):
        with st.spinner('Preparing Excel workbook with setup, summaries, segments and raw simulations...'):
            try:
                if planned_jobs is None:
                    planned_jobs = build_simulation_plan(config, int(config['plan_seed']))
                st.session_state[key] = create_excel_export(config, planned_jobs, run_data)
            except Exception as exc:
                st.error(f'Excel export could not be generated: {exc}')
    if st.session_state.get(key):
        st.download_button(
            'Download Excel results', data=st.session_state[key],
            file_name=f'{safe_filename(config["experiment_name"])}_{run_id[:8]}_socrates.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            type='primary', use_container_width=True,
        )


def render_results(run_state: Mapping[str, Any]) -> None:
    config = run_state['config']
    data = run_state['run_data']
    st.header(f'Results: {config["experiment_name"]}')
    st.caption(f'Completed {format_date(data["run_timestamp"])} | {len(config["questions"])} question(s) | {config["total_simulations"]:,} profiles')
    if data['valid_total'] == data['total_target']:
        st.success(f'Collected all {data["total_target"]:,} requested valid model responses.')
    else:
        st.warning(f'Collected {data["valid_total"]:,} of {data["total_target"]:,} requested valid responses. Some slots reached the {MAX_SLOT_ATTEMPTS}-attempt ceiling.')
    saved = run_state.get('saved_metadata')
    if saved:
        st.caption('This run is saved in shared GitHub history.')
    elif run_state.get('save_error'):
        st.warning(f'The experiment completed, but saving to shared history failed: {run_state["save_error"]}')

    question_tabs = st.tabs([f"{q['id']} - {q['name']}" for q in config['questions']])
    for tab, question in zip(question_tabs, config['questions']):
        with tab:
            render_question_results(config, question, data)
    st.caption('Socrates responses are synthetic model predictions, not independent human survey respondents or field-test evidence.')
    render_usage_and_export(config, data, run_state['run_id'], run_state.get('planned_jobs'))

@st.cache_data(ttl=60, show_spinner=False)
def cached_history_list(repo: str, token: str) -> list[dict[str, Any]]:
    return list_runs(repo, token)


def get_browser_url() -> str:
    configured = get_secret('SOCRATES_APP_URL')
    if configured:
        return configured.rstrip('/')
    try:
        return str(st.context.url).split('?', 1)[0].rstrip('/')
    except Exception:
        return ''


def render_history_page() -> None:
    st.header('Shared experiment history')
    repo, token = get_history_settings()
    if not repo or not token:
        st.info(
            'History needs one-time GitHub configuration. Create a separate PRIVATE GitHub '
            'repository for run data and set SOCRATES_HISTORY_REPO, '
            'SOCRATES_HISTORY_TOKEN and SOCRATES_APP_PASSWORD in Streamlit secrets. '
            'Experiments can still be run without this configuration, but results '
            'will not persist after the session closes.'
        )
        return
    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.caption('Saved runs are shared across authorised users, including after closing the browser.')
    with header_right:
        if st.button('Refresh list', key='refresh_history'):
            cached_history_list.clear()
            st.session_state.pop('_history_payload', None)
            st.rerun()
    try:
        records = cached_history_list(repo, token)
    except HistoryError as exc:
        st.error(str(exc))
        return
    if not records:
        st.info('No saved Socrates experiments yet. Completed future runs will appear here.')
        return
    by_id = {row['run_id']: row for row in records}
    run_from_url = st.query_params.get('run')
    if isinstance(run_from_url, list):
        run_from_url = run_from_url[0] if run_from_url else ''
    direct = run_from_url in by_id
    if run_from_url and not direct:
        st.warning('The requested run was not found in this GitHub history.')
    if direct:
        selected_id = str(run_from_url)
        if st.button('Back to all saved runs'):
            st.query_params.clear()
            st.session_state.pop('_history_payload', None)
            st.rerun()
    else:
        query = st.text_input('Search by experiment name', placeholder='Search saved experiments')
        visible = [r for r in records if query.casefold() in r['experiment_name'].casefold()]
        if not visible:
            st.info('No experiments match that name.')
            return
        display_rows = [{
            'Experiment': record['experiment_name'],
            'Date / time (Zurich)': format_date(record['created_at']),
            'Run ID': record['run_id'][:8],
        } for record in visible]
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
        available_ids = [r['run_id'] for r in visible]
        selected_id = st.selectbox(
            'Select a saved run', available_ids,
            format_func=lambda run_id: (
                f'{by_id[run_id]["experiment_name"]} | '
                f'{format_date(by_id[run_id]["created_at"])}'
            ),
        )
        browser_url = get_browser_url()
        if browser_url:
            st.link_button(
                'Open this run in a new tab',
                f'{browser_url}?run={quote(selected_id)}',
            )
        else:
            st.caption('Set SOCRATES_APP_URL in secrets to enable a new-tab link.')
    metadata = by_id[selected_id]
    cached = st.session_state.get('_history_payload')
    if not cached or cached.get('run_id') != selected_id:
        with st.spinner('Opening saved run from private GitHub history...'):
            try:
                cached = load_run(repo, token, metadata)
            except HistoryError as exc:
                st.error(str(exc))
                return
        st.session_state['_history_payload'] = cached
    st.divider()
    render_results({
        'run_id': selected_id,
        'config': cached['config'],
        'run_data': cached['run_data'],
        'saved_metadata': metadata,
        'planned_jobs': None,
    })


def render_new_experiment() -> None:
    api_key = get_secret('FEATHERLESS_API_KEY')
    if not api_key:
        st.error('FEATHERLESS_API_KEY is unavailable. Add it to Streamlit secrets to run model simulations.')
        st.stop()

    st.header('1. Experiment setup')
    experiment_name = st.text_input(
        'Experiment name', max_chars=100,
        placeholder='e.g. Customer communications pulse survey',
        help='Required for identifying and reopening saved experiments.',
    )
    initial_left, initial_right = st.columns(2)
    with initial_left:
        number_questions = int(st.selectbox(
            'Number of survey questions', options=list(range(1, MAX_QUESTIONS + 1)), index=0,
        ))
    with initial_right:
        total_n = int(st.number_input(
            'Simulated respondents per question', min_value=1,
            max_value=MAX_SIMULATIONS, value=100, step=1,
            help='The same demographic profiles answer every question in separate model calls.',
        ))
    st.caption(
        f'{total_n:,} respondent profiles x {number_questions} question(s) = '
        f'{total_n * number_questions:,} requested valid model answers.'
    )
    context = st.text_area(
        'Customer and experiment context', height=210,
        placeholder='Describe the relevant situation, journey, customer, scenario, and any important background. This context is shared by all survey questions.',
    )

    st.header('2. Survey questions')
    st.caption('Each question has its own interventions, answer options, and optional control / target response.')
    question_tabs = st.tabs([f'Question {i+1}' for i in range(number_questions)])
    questions = []
    for tab, index in zip(question_tabs, range(number_questions)):
        with tab:
            questions.append(render_question(index, total_n))

    st.header('3. Shared demographic simulation')
    st.caption('The same synthetic profiles are reused across questions; variants are randomised separately for each question.')
    use_us_preset = st.toggle(
        'Use representative U.S. sample preset (approximate marginals)',
        value=True, help=US_PRESET_NOTE,
    )
    dimensions = st.multiselect(
        'Demographic dimensions included in prompts',
        options=list(DEMOGRAPHIC_DEFINITIONS),
        default=list(DEFAULT_DEMOGRAPHIC_DIMENSIONS),
    )
    if use_us_preset:
        st.caption(US_PRESET_NOTE)
        with st.expander('U.S. preset source notes'):
            for source in US_PRESET_SOURCES:
                st.code(source, language=None)
    else:
        st.warning('Custom proportions specify independent marginal distributions, not a real-world joint population.')
    distributions, demographic_errors = render_demographics(dimensions, use_us_preset)
    settings = render_advanced_settings()

    errors = validate_study(
        experiment_name, context, total_n, questions, dimensions, demographic_errors,
    )
    current_config = None
    current_fingerprint = None
    if not errors:
        current_config = make_config(
            experiment_name, context, total_n, questions,
            use_us_preset, dimensions, distributions, settings,
        )
        current_fingerprint = config_fingerprint(current_config)

    with st.expander('Preview prompt templates'):
        if current_config:
            sample_profile = generate_profiles(
                1, dimensions, distributions, random.Random(173241)
            )[0]
            qtabs = st.tabs([q['id'] for q in current_config['questions']])
            for tab, question in zip(qtabs, current_config['questions']):
                with tab:
                    variant = st.selectbox(
                        'Example variant', [v['name'] for v in question['variants']],
                        key=f'template_variant_{question["id"]}',
                    )
                    selected = next(v for v in question['variants'] if v['name'] == variant)
                    sample_order = option_orders_for_structure(
                        question['answer_options'], 1, random.Random(42),
                        question['response_option_structure'],
                    )[0]
                    prompt, _ = build_user_prompt(
                        current_config, question, selected,
                        sample_profile['values'], sample_order,
                    )
                    st.code(SYSTEM_PROMPT, language=None)
                    st.code(prompt, language=None)
            st.caption('The exact respondent profiles, assignments and code mappings are shown after preparing the run.')
        else:
            st.caption('Complete all required fields to preview prompt templates.')

    st.divider()
    if st.button('Prepare & review exact prompts', type='primary', use_container_width=True):
        if errors:
            for error in errors:
                st.error(error)
        else:
            assert current_config is not None and current_fingerprint is not None
            plan_seed = secrets.randbits(63)
            prepared_config = {**current_config, 'plan_seed': plan_seed}
            with st.spinner('Generating shared respondent profiles and exact question plans...'):
                try:
                    planned_jobs = build_simulation_plan(prepared_config, plan_seed)
                    runtime_info = get_runtime_info(api_key)
                    st.session_state['prepared_run_v3'] = {
                        'fingerprint': current_fingerprint,
                        'config': prepared_config,
                        'planned_jobs': planned_jobs,
                        'runtime_info': runtime_info,
                    }
                except Exception as exc:
                    st.error(f'Experiment plan could not be prepared: {exc}')
    prepared = st.session_state.get('prepared_run_v3')
    if prepared and current_fingerprint != prepared['fingerprint']:
        st.info('The study has changed since preparation. Prepare and review prompts again before running.')
    if prepared and current_fingerprint == prepared['fingerprint']:
        prepared_config = prepared['config']
        planned_jobs = prepared['planned_jobs']
        runtime_info = prepared['runtime_info']
        st.subheader('Prepared run')
        details = st.columns(3)
        details[0].metric('Questions', len(prepared_config['questions']))
        details[1].metric('Respondents per question', f'{prepared_config["total_simulations"]:,}')
        details[2].metric('Planned model responses', f'{len(planned_jobs):,}')
        if runtime_info.get('error'):
            st.warning('Live Featherless model/plan details could not be loaded. Requests can still be attempted sequentially.')
        if runtime_info.get('available_on_current_plan') is False:
            st.error('Featherless reports that this model is unavailable on the current plan.')
        warning = planned_small_cell_warning(prepared_config, planned_jobs)
        if warning:
            st.warning(warning)
        with st.expander('Review exact planned prompts', expanded=True):
            render_prompt_review(prepared_config, planned_jobs)
        if len(planned_jobs) > 2500:
            confirmed = st.checkbox(
                f'I confirm I want to make {len(planned_jobs):,} billable model requests '
                '(plus any necessary retries).',
                key=f'large_run_confirm::{prepared["fingerprint"]}',
            )
        else:
            confirmed = True
        if st.button(
            'Run experiment', type='primary', use_container_width=True,
            disabled=not confirmed or runtime_info.get('available_on_current_plan') is False,
        ):
            run_data = execute_experiment(
                prepared_config, planned_jobs, api_key, runtime_info,
            )
            run_id = uuid.uuid4().hex
            run_state: dict[str, Any] = {
                'run_id': run_id,
                'fingerprint': prepared['fingerprint'],
                'config': prepared_config,
                'planned_jobs': planned_jobs,
                'run_data': run_data,
                'saved_metadata': None,
                'save_error': None,
            }
            repo, token = get_history_settings()
            if repo and token:
                try:
                    run_state['saved_metadata'] = save_run(
                        repo, token, prepared_config, run_data, run_id=run_id,
                    )
                    cached_history_list.clear()
                except HistoryError as exc:
                    run_state['save_error'] = str(exc)
            else:
                run_state['save_error'] = 'Shared GitHub history is not configured yet. Download the Excel file before closing this session.'
            st.session_state['last_run_v3'] = run_state
    last_run = st.session_state.get('last_run_v3')
    if last_run:
        st.divider()
        if last_run.get('save_error') and get_history_settings()[0]:
            if st.button('Retry saving this run to GitHub history'):
                repo, token = get_history_settings()
                try:
                    last_run['saved_metadata'] = save_run(
                        repo, token, last_run['config'],
                        last_run['run_data'], run_id=last_run['run_id'],
                    )
                    last_run['save_error'] = None
                    cached_history_list.clear()
                    st.success('Run saved to shared history.')
                except HistoryError as exc:
                    last_run['save_error'] = str(exc)
                    st.error(f'History save failed: {exc}')
        render_results(last_run)


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------
require_team_password_if_history_is_enabled()
logo = Path('logo.png')
if logo.exists():
    st.image(str(logo), width=130)
st.caption('Behavioural survey and intervention simulation | Socrates Qwen2.5 14B SFT')
st.title(APP_NAME)
st.write('Configure up to five questions, reuse a shared synthetic respondent pool, and compare survey response patterns across interventions and demographic groups.')

requested_run = st.query_params.get('run')
if isinstance(requested_run, list):
    requested_run = requested_run[0] if requested_run else ''
nav = st.sidebar.radio(
    'Navigation', ['New experiment', 'History'],
    index=1 if requested_run else 0,
)
if nav == 'History':
    render_history_page()
else:
    if requested_run:
        st.query_params.clear()
    render_new_experiment()
