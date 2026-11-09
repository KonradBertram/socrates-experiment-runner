"""Socrates model inference, respondent planning, analysis, and Excel export.

The model's repeated answers are synthetic predictions, not human observations.
All question variants use a single shared demographic sample per experiment.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter



APP_NAME = "Socrates Experiment Runner"
MODEL_ID = "socratesft/socrates-qwen2.5-14b-sft"
API_BASE = "https://api.featherless.ai/v1"

SYSTEM_PROMPT = (
    "You are simulating a survey respondent. Answer exactly as instructed, "
    "following the specified response format without additional commentary."
)

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_USE_TOP_K = False
DEFAULT_TOP_K = 50
MAX_NEW_TOKENS = 8
ESTIMATED_COMPLETION_TOKENS_PER_RESPONSE = 2

MAX_HTTP_RETRIES = 4
MAX_APP_WORKERS = 16
MAX_SIMULATIONS = 10_000
MAX_SLOT_ATTEMPTS = 3
MAX_RESPONSE_OPTIONS = 10
OPTION_CODES = tuple(str(i) for i in range(1, 11))

RESPONSE_STRUCTURE_CATEGORICAL = "Categorical / unordered"
RESPONSE_STRUCTURE_ORDERED = "Ordered / Likert scale"
RESPONSE_STRUCTURE_OPTIONS = (
    RESPONSE_STRUCTURE_CATEGORICAL,
    RESPONSE_STRUCTURE_ORDERED,
)

US_PRESET_NOTE = (
    "Approximate, rounded U.S. adult/household marginal distributions. "
    "Each demographic dimension is sampled independently, so the preset does "
    "not preserve real-world correlations between age, income, employment, "
    "education, household size, and housing tenure. It is not the empirical "
    "joint distribution of SocSci210 respondents."
)

US_PRESET_SOURCES = (
    "https://data.census.gov/table/ACSST1Y2023.S0101",
    "https://data.census.gov/table/ACSST1Y2023.S1501",
    "https://data.census.gov/table/ACSST1Y2023.S1901",
    "https://data.census.gov/table/ACSST1Y2023.S1101",
    "https://data.census.gov/table/ACSST1Y2023.B25003",
    "https://www.bls.gov/cps/",
)


# Category labels intentionally follow or closely map to the labels reported in
# the Socrates paper/model prompt. The U.S. preset is a rounded approximation,
# not a reproduction of the training sample.
DEMOGRAPHIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "Age": {
        "prompt_label": "Age",
        "categories": ["18-24", "25-34", "35-49", "50-64", "65+"],
        "preset": [12.0, 18.0, 25.0, 24.0, 21.0],
    },
    "Gender": {
        "prompt_label": "Gender",
        "categories": ["Female", "Male"],
        "preset": [51.0, 49.0],
    },
    "Income": {
        "prompt_label": "Income",
        "categories": [
            "<5K",
            "5-9K",
            "10-14K",
            "15-19K",
            "20-29K",
            "30-39K",
            "40-49K",
            "50-74K",
            "75-99K",
            "100-124K",
            "125-149K",
            "150-175K+",
            "175-200K+",
            "200K+",
        ],
        "preset": [2.0, 2.0, 3.0, 3.0, 8.0, 7.0, 6.0, 16.0, 13.0, 10.0, 8.0, 6.0, 4.0, 12.0],
    },
    "Education": {
        "prompt_label": "Education",
        "categories": [
            "Less than high school",
            "Some high school (no diploma)",
            "High school graduate or equivalent",
            "Vocational/tech school/some college/associates",
            "Bachelor's degree",
            "Post grad study/professional degree",
        ],
        "preset": [4.0, 7.0, 27.0, 28.0, 22.0, 12.0],
    },
    "Employment": {
        "prompt_label": "Employment",
        "categories": [
            "Employed as paid employee",
            "Self-employed",
            "Looking for work",
            "Temporarily laid off",
            "Retired",
            "Disabled",
            "Not working for other reasons",
        ],
        "preset": [57.0, 6.0, 3.0, 1.0, 18.0, 5.0, 10.0],
    },
    "Marital Status": {
        "prompt_label": "Marital Status",
        "categories": [
            "Married",
            "Never married",
            "Living with partner",
            "Divorced",
            "Widowed",
            "Separated",
        ],
        "preset": [46.0, 28.0, 9.0, 10.0, 5.0, 2.0],
    },
    "Household Size": {
        "prompt_label": "Household Size",
        "categories": ["1", "2", "3", "4", "5+"],
        "preset": [29.0, 34.0, 15.0, 13.0, 9.0],
    },
    "Housing Ownership": {
        "prompt_label": "Housing Ownership",
        "categories": [
            "Owned or being bought by you/someone in your household",
            "Rented for cash",
            "Occupied without payment of cash rent",
        ],
        "preset": [66.0, 32.0, 2.0],
    },
}

DEFAULT_DEMOGRAPHIC_DIMENSIONS = tuple(DEMOGRAPHIC_DEFINITIONS.keys())





class ConfigurationError(ValueError):
    """Raised when a run configuration is internally inconsistent."""


def normalize_response_option_structure(value: Any) -> str:
    """Return a validated response-option structure label.

    Older saved configurations that predate this feature default to categorical
    behavior. Any other unexpected value is rejected rather than silently
    falling back to counterbalancing.
    """
    if value is None:
        return RESPONSE_STRUCTURE_CATEGORICAL
    structure = str(value)
    if structure not in RESPONSE_STRUCTURE_OPTIONS:
        raise ConfigurationError(
            f"Unknown response option structure: {structure!r}. "
            f"Expected one of {RESPONSE_STRUCTURE_OPTIONS}."
        )
    return structure


def canonical_option_id_map(options: Sequence[str]) -> dict[str, str]:
    """Map each answer-option text to a stable internal ID.

    These IDs never depend on how response options are displayed or which
    response code is assigned in a particular simulated respondent prompt.
    """
    option_list = [str(option) for option in options]
    if len(option_list) < 2:
        raise ConfigurationError("At least two response options are required.")
    if len({option.casefold() for option in option_list}) != len(option_list):
        raise ConfigurationError("Response options must be unique.")
    return {option: f"OPT{index:02d}" for index, option in enumerate(option_list, start=1)}


def api_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": APP_NAME,
    }


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_model_and_plan(api_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    encoded_model = quote(MODEL_ID, safe="")
    headers = api_headers(api_key)
    model_response = requests.get(
        f"{API_BASE}/models/{encoded_model}",
        headers=headers,
        timeout=30,
    )
    model_response.raise_for_status()
    plan_response = requests.get(
        f"{API_BASE}/plan",
        headers=headers,
        timeout=30,
    )
    plan_response.raise_for_status()
    return model_response.json(), plan_response.json()


def get_runtime_info(api_key: str) -> dict[str, Any]:
    try:
        model_info, plan_info = get_model_and_plan(api_key)
        pricing = model_info.get("pricing", {}) or {}
        prompt_price = float(pricing.get("prompt", 0) or 0)
        completion_price = float(pricing.get("completion", 0) or 0)

        plan_concurrency = int(plan_info.get("concurrency", 1) or 1)
        model_concurrency_cost = int(model_info.get("concurrency_cost", 1) or 1)
        available_workers = max(1, plan_concurrency // max(1, model_concurrency_cost))
        workers = min(MAX_APP_WORKERS, available_workers)

        model_context = model_info.get("context_length")
        plan_context = plan_info.get("max_context_length")
        context_candidates = [
            int(value)
            for value in (model_context, plan_context)
            if value is not None and int(value) > 0
        ]
        effective_context = min(context_candidates) if context_candidates else None

        return {
            "model_info": model_info,
            "plan_info": plan_info,
            "prompt_price": prompt_price,
            "completion_price": completion_price,
            "pricing_available": prompt_price > 0 or completion_price > 0,
            "workers": workers,
            "available_on_current_plan": model_info.get("available_on_current_plan", True),
            "effective_context_length": effective_context,
            "error": None,
        }
    except Exception as exc:  # The app can still attempt a run sequentially.
        return {
            "model_info": {},
            "plan_info": {},
            "prompt_price": 0.0,
            "completion_price": 0.0,
            "pricing_available": False,
            "workers": 1,
            "available_on_current_plan": None,
            "effective_context_length": None,
            "error": str(exc),
        }


def parse_answer_options(text: str) -> list[str]:
    options = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return options


def largest_remainder_counts(total: int, percentages: Sequence[float]) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not percentages:
        return []
    exact = [total * float(p) / 100.0 for p in percentages]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(exact)),
        key=lambda i: (exact[i] - counts[i], -i),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def distributions_from_preset(dimensions: Iterable[str]) -> dict[str, dict[str, float]]:
    distributions: dict[str, dict[str, float]] = {}
    for dimension in dimensions:
        definition = DEMOGRAPHIC_DEFINITIONS[dimension]
        distributions[dimension] = {
            category: float(percentage)
            for category, percentage in zip(definition["categories"], definition["preset"])
        }
    return distributions


def validate_distribution(distribution: Mapping[str, float], tolerance: float = 0.01) -> bool:
    total = sum(float(value) for value in distribution.values())
    return abs(total - 100.0) <= tolerance and all(float(value) >= 0 for value in distribution.values())


def sample_prompt_value(dimension: str, segment: str, rng: random.Random) -> str:
    if dimension == "Age":
        ranges = {
            "18-24": (18, 24),
            "25-34": (25, 34),
            "35-49": (35, 49),
            "50-64": (50, 64),
            "65+": (65, 90),
        }
        lower, upper = ranges[segment]
        return str(rng.randint(lower, upper))
    if dimension == "Household Size":
        if segment == "5+":
            return str(rng.randint(5, 8))
        return segment
    return segment


def generate_profiles(
    count: int,
    dimensions: Sequence[str],
    distributions: Mapping[str, Mapping[str, float]],
    rng: random.Random,
) -> list[dict[str, dict[str, str]]]:
    if count <= 0:
        return []
    profiles = [
        {"segments": {}, "values": {}}
        for _ in range(count)
    ]

    for dimension in dimensions:
        definition = DEMOGRAPHIC_DEFINITIONS[dimension]
        categories = list(definition["categories"])
        distribution = distributions[dimension]
        percentages = [float(distribution.get(category, 0.0)) for category in categories]
        if not validate_distribution(dict(zip(categories, percentages))):
            raise ConfigurationError(f"{dimension} percentages must total 100%.")

        counts = largest_remainder_counts(count, percentages)
        assignments: list[str] = []
        for category, category_count in zip(categories, counts):
            assignments.extend([category] * category_count)
        rng.shuffle(assignments)

        for profile, segment in zip(profiles, assignments):
            profile["segments"][dimension] = segment
            profile["values"][dimension] = sample_prompt_value(dimension, segment, rng)

    return profiles


def counterbalanced_option_orders(
    options: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[list[str]]:
    """Create near-exactly balanced answer positions using cyclic Latin blocks.

    Each full block contains every cyclic rotation of the options and every
    cyclic rotation of the reversed options. Consequently, every option appears
    equally often in every response-code position within a full block.
    """

    if count <= 0:
        return []
    option_list = list(options)
    if len(option_list) < 2:
        raise ConfigurationError("At least two response options are required.")

    def rotations(items: list[str]) -> list[list[str]]:
        return [items[offset:] + items[:offset] for offset in range(len(items))]

    base_block = rotations(option_list) + rotations(list(reversed(option_list)))
    orders: list[list[str]] = []
    while len(orders) < count:
        block = [list(order) for order in base_block]
        rng.shuffle(block)
        orders.extend(block)
    return orders[:count]


def option_orders_for_structure(
    options: Sequence[str],
    count: int,
    rng: random.Random,
    response_option_structure: str | None,
) -> list[list[str]]:
    """Build the prompt-facing option order without changing canonical identity.

    Ordered/Likert options always retain the exact order entered by the
    researcher. Categorical options use the existing counterbalancing scheme.
    In both cases, the canonical internal option IDs remain fixed.
    """
    structure = normalize_response_option_structure(response_option_structure)
    option_list = list(options)
    canonical_option_id_map(option_list)  # validates count and uniqueness

    if count <= 0:
        return []
    if structure == RESPONSE_STRUCTURE_ORDERED:
        return [list(option_list) for _ in range(count)]
    return counterbalanced_option_orders(option_list, count, rng)


def normalize_model_answer(
    raw_text: str,
    code_to_option: Mapping[str, str],
) -> tuple[str | None, str | None]:
    """Map a Socrates completion to one of the allowed response options.

    Socrates is instructed to return one of the prompt-facing response codes.
    The code is decoded through the mapping stored on that exact respondent job,
    so categorical counterbalancing never changes the canonical answer identity.
    We also tolerate a small set of harmless wrappers such as "Option 2" or
    "The answer is 2" so formatting noise does not burn simulation slots.
    """
    text = (raw_text or "").strip()
    if not text:
        return None, None

    text = text.replace("```", "").strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    cleaned = first_line.strip(" \t\r\n\"'`.,;:()[]{}")

    def match_code(candidate: str) -> str | None:
        for code in code_to_option:
            if candidate.casefold() == str(code).casefold():
                return str(code)
        return None

    # Best case: the model follows the instruction exactly.
    matched_code = match_code(cleaned)
    if matched_code is not None:
        return matched_code, code_to_option[matched_code]

    # Also accept the exact option text if the model returns the label rather than
    # the requested code.
    for code, option in code_to_option.items():
        if cleaned.casefold() == option.strip().casefold():
            return str(code), option

    allowed = sorted((str(code) for code in code_to_option), key=len, reverse=True)
    escaped = "|".join(re.escape(code) for code in allowed)

    # Common wrappers produced by instruction-tuned chat models.
    wrapper_patterns = [
        rf"^\s*(?:option|choice|response|answer)\s*[:=#-]?\s*({escaped})(?:\b|$)",
        rf"^\s*(?:i\s+(?:would\s+)?(?:choose|select|pick))\s*[:=#-]?\s*(?:option\s*)?({escaped})(?:\b|$)",
        rf"^\s*(?:the\s+)?(?:answer|choice|response)\s+(?:is|would\s+be)\s*[:=#-]?\s*({escaped})(?:\b|$)",
        rf"^\s*({escaped})\s*[\).,:;-]",
    ]
    for pattern in wrapper_patterns:
        match = re.search(pattern, first_line, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1)
            matched_code = match_code(candidate)
            if matched_code is not None:
                return matched_code, code_to_option[matched_code]

    return None, None


def extract_chat_content(choice: Mapping[str, Any]) -> str:
    """Extract text robustly from OpenAI-compatible chat completion shapes."""
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, Mapping) else None

    if isinstance(content, str):
        return content

    # Some OpenAI-compatible providers may return a list of content parts.
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        if parts:
            return "".join(parts)

    # Defensive fallback for providers exposing completion text directly.
    direct_text = choice.get("text")
    return direct_text if isinstance(direct_text, str) else ""


def run_one(
    job: Mapping[str, Any],
    api_key: str,
    settings: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": job["system_prompt"]},
            {"role": "user", "content": job["user_prompt"]},
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": float(settings["temperature"]),
        "top_p": float(settings["top_p"]),
    }
    if settings.get("use_top_k"):
        data["top_k"] = int(settings["top_k"])

    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    retry_events = 0
    http_attempts = 0
    last_error: str | None = None

    for attempt in range(MAX_HTTP_RETRIES):
        http_attempts += 1
        try:
            response = requests.post(
                f"{API_BASE}/chat/completions",
                headers=api_headers(api_key),
                json=data,
                timeout=(20, 180),
            )

            if response.status_code in retryable_statuses:
                last_error = f"Temporary Featherless error {response.status_code}: {response.text[:300]}"
                if attempt < MAX_HTTP_RETRIES - 1:
                    retry_events += 1
                    time.sleep((1.7**attempt) + random.uniform(0.1, 0.8))
                    continue
                break

            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            raw = extract_chat_content(choice)
            selected_code, selected_option = normalize_model_answer(raw, job["code_to_option"])
            selected_option_id = (
                job["option_to_canonical_id"].get(selected_option)
                if selected_option is not None
                else None
            )
            if selected_option is not None and selected_option_id is None:
                raise ConfigurationError(
                    "A decoded answer option was missing from the canonical option mapping."
                )

            usage = payload.get("usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)

            cost = None
            if runtime_info.get("pricing_available"):
                cost = (
                    prompt_tokens * float(runtime_info["prompt_price"])
                    + completion_tokens * float(runtime_info["completion_price"])
                )

            status = "valid" if selected_option is not None else "invalid_output"
            return {
                **dict(job),
                "raw_completion": raw,
                "selected_code": selected_code,
                "selected_option": selected_option,
                "selected_option_id": selected_option_id,
                "status": status,
                "error": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "http_attempts": http_attempts,
                "retry_events": retry_events,
                "finish_reason": choice.get("finish_reason"),
            }

        except requests.RequestException as exc:
            last_error = str(exc)
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            should_retry = status_code is None or status_code in retryable_statuses
            if should_retry and attempt < MAX_HTTP_RETRIES - 1:
                retry_events += 1
                time.sleep((1.7**attempt) + random.uniform(0.1, 0.8))
                continue
            break
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"Unexpected API response: {exc}"
            if attempt < MAX_HTTP_RETRIES - 1:
                retry_events += 1
                time.sleep((1.7**attempt) + random.uniform(0.1, 0.8))
                continue
            break
        except Exception as exc:  # Defensive: keep the batch alive.
            last_error = str(exc)
            if attempt < MAX_HTTP_RETRIES - 1:
                retry_events += 1
                time.sleep((1.7**attempt) + random.uniform(0.1, 0.8))
                continue
            break

    return {
        **dict(job),
        "raw_completion": "",
        "selected_code": None,
        "selected_option": None,
        "selected_option_id": None,
        "status": "api_error",
        "error": last_error or "Unknown API error",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": None,
        "http_attempts": http_attempts,
        "retry_events": retry_events,
        "finish_reason": None,
    }


def _selected_canonical_id(
    row: Mapping[str, Any],
    option_to_id: Mapping[str, str],
) -> str | None:
    """Read and verify the stable internal answer identity.

    New runs carry an explicit canonical ID. Older rows can still be interpreted
    from selected_option, but any disagreement between the two representations
    is treated as a hard configuration error instead of being silently counted.
    """
    valid_ids = set(option_to_id.values())
    selected_option = row.get("selected_option")
    selected_id = row.get("selected_option_id")

    if selected_id is not None:
        selected_id = str(selected_id)
        if selected_id not in valid_ids:
            raise ConfigurationError(
                f"Unknown canonical option ID in result row: {selected_id!r}."
            )
        if selected_option is not None:
            expected_id = option_to_id.get(selected_option)
            if expected_id != selected_id:
                raise ConfigurationError(
                    "Result row contains inconsistent selected_option and "
                    "selected_option_id values."
                )
        return selected_id

    if selected_option is None:
        return None
    fallback_id = option_to_id.get(selected_option)
    if fallback_id is None:
        raise ConfigurationError(
            f"Unknown selected answer option in result row: {selected_option!r}."
        )
    return fallback_id


def _excel_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        if len(value) > 32_760:
            value = value[:32_730] + "\n[TRUNCATED FOR EXCEL CELL LIMIT]"
        if value.startswith(("=", "+", "-", "@")):
            value = "'" + value
    return value


def _format_worksheet(ws, freeze_panes: str | None = None) -> None:
    if freeze_panes:
        ws.freeze_panes = freeze_panes
    ws.sheet_view.showGridLines = False
    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        max_length = 0
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, min(len(value), 80))
        ws.column_dimensions[letter].width = min(max(max_length + 2, 10), 55)


def _write_table(ws, start_row: int, headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> int:
    header_fill = PatternFill("solid", fgColor="171B23")
    accent_fill = PatternFill("solid", fgColor="00FF9B")
    header_font = Font(color="F3F4F6", bold=True)
    accent_font = Font(color="000000", bold=True)
    thin_gray = Side(style="thin", color="39404C")

    for column, header in enumerate(headers, start=1):
        cell = ws.cell(start_row, column, _excel_safe(header))
        cell.fill = accent_fill if column == 1 else header_fill
        cell.font = accent_font if column == 1 else header_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin_gray)

    for row_index, row in enumerate(rows, start=start_row + 1):
        for column, header in enumerate(headers, start=1):
            cell = ws.cell(row_index, column, _excel_safe(row.get(header)))
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if header.endswith("(%)") or header.endswith("(pp)"):
                cell.number_format = "0.0"
            elif header in {"Cost", "Calculated API cost"}:
                cell.number_format = '$0.0000'
    return start_row + 1 + len(rows)


def build_run_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

# ---------------------------------------------------------------------------
# Multi-question experiment planning. A respondent profile is generated once
# and reused across all questions; each question randomises the assignment.
# ---------------------------------------------------------------------------
MAX_QUESTIONS = 5
MAX_VARIANTS = 5


def question_by_id(config: Mapping[str, Any], question_id: str) -> Mapping[str, Any]:
    for question in config['questions']:
        if question['id'] == question_id:
            return question
    raise ConfigurationError(f'Unknown question ID: {question_id}')


def build_user_prompt(
    config: Mapping[str, Any],
    question: Mapping[str, Any],
    variant: Mapping[str, Any],
    profile_values: Mapping[str, str],
    option_order: Sequence[str],
) -> tuple[str, dict[str, str]]:
    canonical = list(question['answer_options'])
    canonical_option_id_map(canonical)
    if len(option_order) != len(canonical) or set(option_order) != set(canonical):
        raise ConfigurationError('Prompt options must contain every canonical option exactly once.')
    codes = OPTION_CODES[:len(canonical)]
    code_to_option = dict(zip(codes, option_order))
    profile_lines = [
        f"- {DEMOGRAPHIC_DEFINITIONS[d]['prompt_label']}: {profile_values[d]}"
        for d in config['demographic_dimensions']
    ]
    profile_text = '\n'.join(profile_lines) if profile_lines else '- Not specified'
    context = config['context'].strip()
    if not context:
        raise ConfigurationError('Experiment context is required.')

    intervention = str(variant.get('text') or '').strip()
    stimulus_section = (
        f'\nTOUCHPOINT / INTERVENTION SHOWN TO YOU\n'
        f'--- BEGIN VARIANT ---\n{intervention}\n--- END VARIANT ---\n'
        if intervention else ''
    )
    option_lines = '\n'.join(f'{code}: {code_to_option[code]}' for code in codes)
    allowed_codes = ', '.join(codes)
    prompt = f"""You are a survey respondent with the following demographic profile:
{profile_text}

Read the situation below and answer the question as this person most likely would.
Use their likely perspective rather than a socially desirable or ideal response.
Follow the response instructions precisely.

SITUATION CONTEXT
{context}
{stimulus_section}
QUESTION
{question['outcome_question'].strip()}

RESPONSE OPTIONS
{option_lines}

The response codes are technical labels used only to record your answer.
Only return the response code corresponding to your answer from this list: {allowed_codes}.
Return the code only, nothing else."""
    return prompt, code_to_option


def _balanced_variant_assignments(
    profiles: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str],
    distributions: Mapping[str, Mapping[str, float]],
    rng: random.Random,
) -> dict[int, int]:
    """Randomised, quota-aware assignment with exact variant sample sizes.

    The common sample's marginal demographics are exact (within rounding).
    We attempt to balance all dimensions within each arm, but their joint
    distributions need not permit simultaneous exact quota satisfaction.
    """
    counts = [int(v['count']) for v in variants]
    if sum(counts) != len(profiles) or any(c <= 0 for c in counts):
        raise ConfigurationError('Per-question variant counts must sum to the respondent count.')
    targets: list[dict[str, dict[str, int]]] = []
    for size in counts:
        dimension_targets = {}
        for dimension in dimensions:
            category_shares = distributions[dimension]
            category_names = list(category_shares)
            expected_counts = largest_remainder_counts(size, list(category_shares.values()))
            dimension_targets[dimension] = dict(zip(category_names, expected_counts))
        targets.append(dimension_targets)

    used = [{dimension: Counter() for dimension in dimensions} for _ in variants]
    remaining = list(counts)
    order = list(range(len(profiles)))
    rng.shuffle(order)
    allocation: dict[int, int] = {}
    for profile_index in order:
        segment_values = profiles[profile_index]['segments']
        choices = [index for index, capacity in enumerate(remaining) if capacity > 0]
        if not choices:
            raise ConfigurationError('No variant capacity available.')
        best_index = None
        best_score = float('-inf')
        for variant_index in choices:
            score = sum(
                targets[variant_index][dimension].get(segment_values[dimension], 0)
                - used[variant_index][dimension][segment_values[dimension]]
                for dimension in dimensions
            )
            # Break ties randomly; the seeded PRNG makes the plan reproducible.
            score += rng.random() * 0.25
            if score > best_score:
                best_index = variant_index
                best_score = score
        assert best_index is not None
        allocation[profile_index] = best_index
        remaining[best_index] -= 1
        for dimension in dimensions:
            used[best_index][dimension][segment_values[dimension]] += 1
    return allocation


def build_simulation_plan(config: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    questions = config['questions']
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ConfigurationError(f'Choose 1 to {MAX_QUESTIONS} survey questions.')
    n = int(config['total_simulations'])
    if not 1 <= n <= MAX_SIMULATIONS:
        raise ConfigurationError(f'Respondents per question must be 1 to {MAX_SIMULATIONS:,}.')
    rng = random.Random(seed)
    profiles = generate_profiles(
        n, config['demographic_dimensions'], config['demographic_distributions'], rng
    )
    jobs: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question['id'])
        variants = question['variants']
        if not 1 <= len(variants) <= MAX_VARIANTS:
            raise ConfigurationError(f'{qid} must have between 1 and {MAX_VARIANTS} arms.')
        question_rng = random.Random(rng.getrandbits(64))
        assigned = _balanced_variant_assignments(
            profiles, variants, config['demographic_dimensions'],
            config['demographic_distributions'], question_rng,
        )
        canonical = list(question['answer_options'])
        option_to_id = canonical_option_id_map(canonical)
        structure = normalize_response_option_structure(question['response_option_structure'])
        for variant_index, variant in enumerate(variants):
            assigned_ids = [idx for idx, arm in assigned.items() if arm == variant_index]
            question_rng.shuffle(assigned_ids)
            orders = option_orders_for_structure(
                canonical, len(assigned_ids), question_rng, structure
            )
            for respondent_index, order in zip(assigned_ids, orders):
                respondent_id = f'R{respondent_index + 1:05d}'
                profile = profiles[respondent_index]
                user_prompt, code_to_option = build_user_prompt(
                    config, question, variant, profile['values'], order
                )
                jobs.append({
                    'slot_id': f'{qid}-{respondent_id}',
                    'respondent_id': respondent_id,
                    'question_id': qid,
                    'question_name': question['name'],
                    'slot_attempt': 1,
                    'variant': variant['name'],
                    'variant_index': variant_index,
                    'profile_segments': dict(profile['segments']),
                    'profile_values': dict(profile['values']),
                    'response_option_structure': structure,
                    'canonical_option_order': list(canonical),
                    'option_to_canonical_id': dict(option_to_id),
                    'option_order': list(order),
                    'code_to_option': code_to_option,
                    'code_to_canonical_id': {
                        code: option_to_id[option] for code, option in code_to_option.items()
                    },
                    'system_prompt': SYSTEM_PROMPT,
                    'user_prompt': user_prompt,
                })
    validate_plan_response_mappings(config, jobs)
    rng.shuffle(jobs)
    return jobs


def validate_plan_response_mappings(
    config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]
) -> None:
    if len(jobs) != int(config['total_simulations']) * len(config['questions']):
        raise ConfigurationError('Unexpected number of planned jobs.')
    questions = {q['id']: q for q in config['questions']}
    respondent_profiles: dict[str, tuple[dict, dict]] = {}
    seen = set()
    arm_counts = Counter()
    for job in jobs:
        qid = job['question_id']
        if qid not in questions:
            raise ConfigurationError(f'Unexpected question ID: {qid}')
        question = questions[qid]
        response_options = list(question['answer_options'])
        codes = OPTION_CODES[:len(response_options)]
        canonical_ids = canonical_option_id_map(response_options)
        order = list(job['option_order'])
        mapping = dict(job['code_to_option'])
        if len(order) != len(response_options) or set(order) != set(response_options):
            raise ConfigurationError(f'{job["slot_id"]}: response options were corrupted.')
        if list(mapping) != list(codes) or list(mapping.values()) != order:
            raise ConfigurationError(f'{job["slot_id"]}: response-code mapping is inconsistent.')
        if job['option_to_canonical_id'] != canonical_ids or job['code_to_canonical_id'] != {
            code: canonical_ids[option] for code, option in mapping.items()
        }:
            raise ConfigurationError(f'{job["slot_id"]}: canonical IDs were corrupted.')
        if question['response_option_structure'] == RESPONSE_STRUCTURE_ORDERED:
            if order != response_options:
                raise ConfigurationError(f'{job["slot_id"]}: Likert order was changed.')
        identity = (qid, job['respondent_id'])
        if identity in seen:
            raise ConfigurationError(f'Duplicate respondent/question pair: {identity}')
        seen.add(identity)
        profile = (job['profile_segments'], job['profile_values'])
        prior_profile = respondent_profiles.setdefault(job['respondent_id'], profile)
        if prior_profile != profile:
            raise ConfigurationError('A respondent profile changed across questions.')
        arm_counts[(qid, job['variant'])] += 1
    if len(respondent_profiles) != int(config['total_simulations']):
        raise ConfigurationError('The shared respondent pool has the wrong size.')
    for question in config['questions']:
        for variant in question['variants']:
            if arm_counts[(question['id'], variant['name'])] != int(variant['count']):
                raise ConfigurationError('Variant allocations differ from the requested counts.')


def _median_ordinal(counts: Counter, option_ids: Sequence[str], n: int) -> float | None:
    if not n:
        return None
    positions = ((n - 1) // 2, n // 2)
    median_ranks = []
    cursor = 0
    for rank, oid in enumerate(option_ids, start=1):
        next_cursor = cursor + counts.get(oid, 0)
        for position in positions:
            if cursor <= position < next_cursor:
                median_ranks.append(rank)
        cursor = next_cursor
    return sum(median_ranks) / 2 if len(median_ranks) == 2 else None


def _summarize_response_group(
    question: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    control_target_rate: float | None = None,
) -> dict[str, Any]:
    options = question['answer_options']
    option_to_id = canonical_option_id_map(options)
    counts = Counter(_selected_canonical_id(row, option_to_id) for row in rows)
    n = len(rows)
    target = question.get('target_behavior') or None
    target_id = option_to_id[target] if target else None
    target_count = counts.get(target_id, 0) if target_id else None
    target_rate = (target_count / n * 100) if n and target_id else None
    effect = (
        target_rate - control_target_rate
        if target_rate is not None and control_target_rate is not None
        else None
    )
    result: dict[str, Any] = {
        'Valid N': n,
        'Target behavior': target,
        'Target count': target_count,
        'Target rate (%)': target_rate,
        'Effect vs control (pp)': effect,
    }
    for option in options:
        count = counts.get(option_to_id[option], 0)
        result[f'{option} count'] = count
        result[f'{option} rate (%)'] = count / n * 100 if n else None
    if question['response_option_structure'] == RESPONSE_STRUCTURE_ORDERED:
        option_ids = [option_to_id[option] for option in options]
        result['Mean ordinal score'] = (
            sum(rank * counts.get(oid, 0) for rank, oid in enumerate(option_ids, start=1)) / n
            if n else None
        )
        result['Median ordinal score'] = _median_ordinal(counts, option_ids, n)
    return result


def overall_result_rows(
    question: Mapping[str, Any], accepted_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    qid = question['id']
    rows_by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in accepted_rows:
        if row['question_id'] == qid:
            rows_by_variant[str(row['variant'])].append(row)
    control = question.get('control_variant') or None
    target = question.get('target_behavior') or None
    control_rate = None
    if target and control:
        control_stats = _summarize_response_group(question, rows_by_variant.get(control, []))
        control_rate = control_stats['Target rate (%)']
    output = []
    for variant in question['variants']:
        name = variant['name']
        stats = _summarize_response_group(question, rows_by_variant.get(name, []), control_rate)
        output.append({
            'Question ID': qid,
            'Question': question['name'],
            'Variant': name,
            'Control': name == control if control else False,
            'Target N': int(variant['count']),
            **stats,
        })
    return output


def segment_result_rows(
    config: Mapping[str, Any], question: Mapping[str, Any],
    accepted_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    qid = question['id']
    qrows = [row for row in accepted_rows if row['question_id'] == qid]
    control = question.get('control_variant') or None
    target = question.get('target_behavior') or None
    output = []
    for dimension in config['demographic_dimensions']:
        for segment in config['demographic_distributions'][dimension]:
            segment_rows = [
                row for row in qrows
                if row.get('profile_segments', {}).get(dimension) == segment
            ]
            by_variant = defaultdict(list)
            for row in segment_rows:
                by_variant[row['variant']].append(row)
            control_rate = None
            if target and control:
                control_stats = _summarize_response_group(
                    question, by_variant.get(control, [])
                )
                control_rate = control_stats['Target rate (%)']
            for variant in question['variants']:
                name = variant['name']
                stats = _summarize_response_group(
                    question, by_variant.get(name, []), control_rate
                )
                output.append({
                    'Question ID': qid,
                    'Question': question['name'],
                    'Demographic': dimension,
                    'Segment': segment,
                    'Variant': name,
                    'Control': name == control if control else False,
                    **stats,
                })
    return output


def profile_balance_rows(
    config: Mapping[str, Any], question: Mapping[str, Any],
    planned_jobs: Sequence[Mapping[str, Any]],
    accepted_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    qid = question['id']
    output = []
    planned_counts = Counter(
        (job['variant'], dimension, job['profile_segments'].get(dimension))
        for job in planned_jobs if job['question_id'] == qid
        for dimension in config['demographic_dimensions']
    )
    accepted_counts = Counter(
        (row['variant'], dimension, row.get('profile_segments', {}).get(dimension))
        for row in accepted_rows if row['question_id'] == qid
        for dimension in config['demographic_dimensions']
    )
    for dimension in config['demographic_dimensions']:
        for segment in config['demographic_distributions'][dimension]:
            for variant in question['variants']:
                key = (variant['name'], dimension, segment)
                output.append({
                    'Question ID': qid,
                    'Question': question['name'],
                    'Demographic': dimension,
                    'Segment': segment,
                    'Variant': variant['name'],
                    'Planned N': planned_counts[key],
                    'Valid N': accepted_counts[key],
                })
    return output


def compact_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    """Do not duplicate the full prompt thousands of times in run history."""
    keys = (
        'attempt_id', 'slot_id', 'respondent_id', 'question_id', 'question_name',
        'slot_attempt', 'variant', 'variant_index', 'profile_segments',
        'profile_values', 'response_option_structure', 'canonical_option_order',
        'option_order', 'code_to_option', 'code_to_canonical_id',
        'selected_code', 'selected_option', 'selected_option_id',
        'raw_completion', 'status', 'error', 'prompt_tokens', 'completion_tokens',
        'cost', 'http_attempts', 'retry_events', 'finish_reason',
    )
    return {key: row.get(key) for key in keys}



def _union_headers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    headers: list[str] = []
    seen = set()
    for row in rows:
        for header in row:
            if header not in seen:
                headers.append(header)
                seen.add(header)
    return headers

def create_excel_export(
    config: Mapping[str, Any],
    planned_jobs: Sequence[Mapping[str, Any]],
    run_data: Mapping[str, Any],
) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = 'Setup'
    dark = PatternFill('solid', fgColor='171B23')
    accent = PatternFill('solid', fgColor='00FF9B')
    label_font = Font(color='F3F4F6', bold=True)
    title_font = Font(color='000000', bold=True, size=12)

    def section(title: str, row: int) -> int:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        cell = ws.cell(row, 1, title)
        cell.fill = accent
        cell.font = title_font
        return row + 1

    def key_value(label: str, value: Any, row: int) -> int:
        label_cell = ws.cell(row, 1, label)
        label_cell.fill = dark
        label_cell.font = label_font
        value_cell = ws.cell(row, 2, _excel_safe(value))
        value_cell.alignment = Alignment(wrap_text=True, vertical='top')
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=4)
        return row + 1

    row = section('Experiment and run metadata', 1)
    metadata = [
        ('Experiment name', config['experiment_name']),
        ('Run timestamp (UTC)', run_data.get('run_timestamp')),
        ('Model', MODEL_ID),
        ('Plan seed', config.get('plan_seed')),
        ('Number of questions', len(config['questions'])),
        ('Valid respondent target per question', config['total_simulations']),
        ('Total target responses', len(config['questions']) * config['total_simulations']),
        ('Experiment context', config['context']),
        ('Representative US preset', config['use_us_preset']),
        ('Demographic preset note', US_PRESET_NOTE if config['use_us_preset'] else 'Custom marginal distributions'),
    ]
    for label, value in metadata:
        row = key_value(label, value, row)
    row += 1
    row = section('Model and sampling settings', row)
    for label, value in [
        ('Temperature', config['settings']['temperature']),
        ('Top-p', config['settings']['top_p']),
        ('Top-k', config['settings']['top_k'] if config['settings']['use_top_k'] else 'Disabled'),
        ('Maximum generated tokens', MAX_NEW_TOKENS),
        ('Maximum attempts per response slot', MAX_SLOT_ATTEMPTS),
        ('Respondent profile reuse', 'Same respondent profile across all questions'),
        ('Intervention assignment', 'Independently randomised per question; exact arm Ns'),
        ('Demographic balancing', 'Exact overall marginals; approximate within-arm balancing'),
        ('Categorical responses', 'Counterbalanced codes decoded to fixed canonical IDs'),
        ('Ordered responses', 'Fixed answer order and code mappings'),
    ]:
        row = key_value(label, value, row)
    row += 1
    row = section('Demographic marginal distributions', row)
    demographics = [
        {'Dimension': dimension, 'Category': category, 'Share (%)': share}
        for dimension in config['demographic_dimensions']
        for category, share in config['demographic_distributions'][dimension].items()
    ]
    row = _write_table(ws, row, ['Dimension', 'Category', 'Share (%)'], demographics) + 1
    row = section('Run quality and actual usage', row)
    for label, key in [
        ('Valid responses', 'valid_total'), ('Requested responses', 'total_target'),
        ('Invalid outputs', 'invalid_outputs'), ('API-failed attempts', 'api_failed_attempts'),
        ('Temporary API retries', 'retry_events'), ('Simulation attempts', 'simulation_attempts'),
        ('Underlying HTTP requests', 'http_requests'), ('Input tokens', 'input_tokens'),
        ('Output tokens', 'output_tokens'), ('Calculated API cost', 'actual_cost'),
    ]:
        row = key_value(label, run_data.get(key), row)
    ws.column_dimensions['A'].width = 38
    ws.column_dimensions['B'].width = 45
    ws.column_dimensions['C'].width = 26
    ws.column_dimensions['D'].width = 76
    ws.sheet_view.showGridLines = False

    setup = wb.create_sheet('Question Setup')
    question_setup = [
        {
            'Question ID': q['id'], 'Question name': q['name'],
            'Survey question': q['outcome_question'],
            'Intervention used': q['has_intervention'],
            'Variant': v['name'], 'Intervention text': v['text'],
            'Allocation (%)': v['allocation'], 'Target N': v['count'],
            'Response structure': q['response_option_structure'],
            'Answer options': '\n'.join(q['answer_options']),
            'Canonical IDs': canonical_option_id_map(q['answer_options']),
            'Target behavior': q.get('target_behavior'),
            'Control variant': q.get('control_variant'),
        }
        for q in config['questions'] for v in q['variants']
    ]
    _write_table(setup, 1, list(question_setup[0]), question_setup)
    _format_worksheet(setup, 'A2')
    setup.auto_filter.ref = setup.dimensions

    overall = wb.create_sheet('Overall Results')
    overall_rows = [
        record for question in config['questions']
        for record in overall_result_rows(question, run_data['accepted_rows'])
    ]
    _write_table(overall, 1, _union_headers(overall_rows), overall_rows)
    _format_worksheet(overall, 'A2')
    overall.auto_filter.ref = overall.dimensions

    segments = wb.create_sheet('Segment Results')
    segment_rows = [
        record for question in config['questions']
        for record in segment_result_rows(config, question, run_data['accepted_rows'])
    ]
    _write_table(segments, 1, _union_headers(segment_rows) if segment_rows else ['Question ID'], segment_rows)
    _format_worksheet(segments, 'A2')
    segments.auto_filter.ref = segments.dimensions

    balance = wb.create_sheet('Profile Balance')
    balance_rows = [
        record for question in config['questions']
        for record in profile_balance_rows(config, question, planned_jobs, run_data['accepted_rows'])
    ]
    _write_table(balance, 1, _union_headers(balance_rows) if balance_rows else ['Question ID'], balance_rows)
    _format_worksheet(balance, 'A2')
    balance.auto_filter.ref = balance.dimensions

    full_distribution = wb.create_sheet('Overall Distributions')
    distribution_rows = [
        {
            'Question ID': question['id'], 'Question': question['name'],
            'Variant': row['Variant'], 'Response option': option,
            'Option position': index, 'Response count': row[f'{option} count'],
            'Response share (%)': row[f'{option} rate (%)'],
            'Valid N': row['Valid N'],
        }
        for question in config['questions']
        for row in overall_result_rows(question, run_data['accepted_rows'])
        for index, option in enumerate(question['answer_options'], start=1)
    ]
    _write_table(full_distribution, 1, list(distribution_rows[0]), distribution_rows)
    _format_worksheet(full_distribution, 'A2')
    full_distribution.auto_filter.ref = full_distribution.dimensions

    segmented_distribution = wb.create_sheet('Segment Distributions')
    segment_distribution_rows = [
        {
            'Question ID': question['id'], 'Question': question['name'],
            'Demographic': row['Demographic'], 'Segment': row['Segment'],
            'Variant': row['Variant'], 'Response option': option,
            'Option position': index, 'Response count': row[f'{option} count'],
            'Response share (%)': row[f'{option} rate (%)'],
            'Valid N': row['Valid N'],
        }
        for question in config['questions']
        for row in segment_result_rows(config, question, run_data['accepted_rows'])
        for index, option in enumerate(question['answer_options'], start=1)
    ]
    _write_table(segmented_distribution, 1,
                 list(segment_distribution_rows[0]) if segment_distribution_rows else ['Question ID'],
                 segment_distribution_rows)
    _format_worksheet(segmented_distribution, 'A2')
    segmented_distribution.auto_filter.ref = segmented_distribution.dimensions

    raw = wb.create_sheet('Raw Simulations')
    raw_rows = []
    for result in run_data['rows']:
        data = {
            'Attempt ID': result.get('attempt_id'),
            'Slot ID': result.get('slot_id'),
            'Respondent ID': result.get('respondent_id'),
            'Question ID': result.get('question_id'),
            'Question': result.get('question_name'),
            'Variant': result.get('variant'),
            'Slot attempt': result.get('slot_attempt'),
            'Status': result.get('status'),
            'Selected response code': result.get('selected_code'),
            'Selected response text': result.get('selected_option'),
            'Canonical option ID': result.get('selected_option_id'),
            'Raw model completion': result.get('raw_completion'),
            'Error': result.get('error'),
            'Prompt option order': result.get('option_order'),
            'Code-to-option map': result.get('code_to_option'),
            'Prompt tokens': result.get('prompt_tokens'),
            'Completion tokens': result.get('completion_tokens'),
            'Cost': result.get('cost'),
            'HTTP attempts': result.get('http_attempts'),
            'Temporary retries': result.get('retry_events'),
            'Finish reason': result.get('finish_reason'),
        }
        for dimension in config['demographic_dimensions']:
            data[f'{dimension} segment'] = (result.get('profile_segments') or {}).get(dimension)
            data[f'{dimension} prompt value'] = (result.get('profile_values') or {}).get(dimension)
        raw_rows.append(data)
    _write_table(raw, 1, list(raw_rows[0]) if raw_rows else ['Attempt ID'], raw_rows)
    _format_worksheet(raw, 'A2')
    raw.auto_filter.ref = raw.dimensions

    sources = wb.create_sheet('Sources and Notes')
    source_rows = [
        {'Type': 'Model card', 'Source': 'https://huggingface.co/socratesft/socrates-qwen2.5-14b-sft'},
        {'Type': 'Socrates paper', 'Source': 'https://arxiv.org/abs/2509.05830'},
        {'Type': 'Caveat', 'Source': 'Synthetic model responses are not human participants or causal field evidence.'},
        {'Type': 'Preset caveat', 'Source': US_PRESET_NOTE},
    ] + [{'Type': 'US preset source', 'Source': source} for source in US_PRESET_SOURCES]
    _write_table(sources, 1, ['Type', 'Source'], source_rows)
    _format_worksheet(sources, 'A2')

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()
