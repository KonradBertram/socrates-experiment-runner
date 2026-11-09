"""Private GitHub repository backed, persistent experiment history.

Each run is a unique, compressed JSON file. No central index is overwritten,
so concurrent team members cannot clobber one another's records. Use a
*separate* private repository so each saved run does not redeploy Streamlit.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import quote

import requests

API_ROOT = 'https://api.github.com'
MAX_GITHUB_COMPRESSED_BYTES = 12_000_000
FILENAME_PATTERN = re.compile(
    r'^(?P<stamp>\d{8}T\d{6}Z)__(?P<title>[^/]+)__(?P<id>[a-f0-9]{32})\.json\.gz$'
)
REPO_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
SHA_PATTERN = re.compile(r'^[0-9a-f]{40}$')


class HistoryError(RuntimeError):
    """An actionable GitHub history storage error."""


def _repo_path(repo: str) -> str:
    if not REPO_PATTERN.fullmatch(repo.strip()):
        raise HistoryError('History repository must have the format owner/repository.')
    return repo.strip()


def _headers(token: str) -> dict[str, str]:
    if not token:
        raise HistoryError('Missing GitHub history token.')
    return {
        'Accept': 'application/vnd.github+json',
        'Authorization': f'Bearer {token}',
        'X-GitHub-Api-Version': '2022-11-28',
    }


def _request(method: str, url: str, token: str, **kwargs: Any) -> requests.Response:
    try:
        response = requests.request(method, url, headers=_headers(token), timeout=45, **kwargs)
    except requests.RequestException as exc:
        raise HistoryError(f'Cannot connect to GitHub history: {exc}') from exc
    if response.status_code in (401, 403):
        raise HistoryError(
            f'GitHub rejected history access (HTTP {response.status_code}). '
            'Check the token, repository access, and Contents permission.'
        )
    if response.status_code >= 400 and response.status_code != 404:
        raise HistoryError(f'GitHub history API error {response.status_code}: {response.text[:250]}')
    return response


def _filename(name: str, run_id: str, timestamp: datetime) -> str:
    encoded = base64.urlsafe_b64encode(name.encode('utf-8')).decode('ascii').rstrip('=')
    if len(encoded) <= 175:
        title = 'b-' + encoded
    else:
        normalized = unicodedata.normalize('NFKD', name)
        slug = re.sub(r'[^A-Za-z0-9_-]+', '-', normalized.encode('ascii', 'ignore').decode('ascii'))
        title = 's-' + slug.strip('-')[:85]
    return f'{timestamp.strftime("%Y%m%dT%H%M%SZ")}__{title}__{run_id}.json.gz'


def _filename_metadata(filename: str, path: str, sha: str, byte_size: int) -> dict[str, Any] | None:
    match = FILENAME_PATTERN.fullmatch(filename)
    if not match:
        return None
    encoded_title = match.group('title')
    if encoded_title.startswith('b-'):
        try:
            encoded = encoded_title[2:]
            padding = '=' * (-len(encoded) % 4)
            title = base64.urlsafe_b64decode(encoded + padding).decode('utf-8')
        except (ValueError, UnicodeError, binascii.Error):
            title = '(unknown name)'
    else:
        title = encoded_title[2:].replace('-', ' ') + '...' if encoded_title.startswith('s-') else encoded_title
    stamp = datetime.strptime(match.group('stamp'), '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    return {
        'run_id': match.group('id'),
        'experiment_name': title,
        'created_at': stamp.isoformat(),
        'path': path,
        'sha': sha,
        'size': byte_size,
    }


def save_run(repo: str, token: str, config: Mapping[str, Any], run_data: Mapping[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """Save a completed run; retry using the same run_id to avoid duplicates."""
    repo = _repo_path(repo)
    run_id = run_id or uuid.uuid4().hex
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise HistoryError('Invalid run ID.')
    timestamp_value = run_data.get('run_timestamp')
    if not timestamp_value:
        raise HistoryError('Run has no completion timestamp.')
    try:
        timestamp = datetime.fromisoformat(str(timestamp_value).replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)
    except ValueError as exc:
        raise HistoryError('Invalid run timestamp.') from exc
    exp_name = str(config.get('experiment_name') or '').strip()
    if not exp_name:
        raise HistoryError('An experiment name is required for history.')

    filename = _filename(exp_name, run_id, timestamp)
    path = f'history/{timestamp.strftime("%Y-%m")}/{filename}'
    payload = {
        'schema_version': 2,
        'run_id': run_id,
        'experiment_name': exp_name,
        'created_at': timestamp.isoformat(),
        'config': dict(config),
        'run_data': dict(run_data),
    }
    compressed = gzip.compress(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'),
        compresslevel=6,
    )
    if len(compressed) > MAX_GITHUB_COMPRESSED_BYTES:
        raise HistoryError(
            f'Run history record is {len(compressed) / 1_000_000:.1f} MB compressed. '
            'It exceeds the configured GitHub storage limit. Download the Excel results instead.'
        )
    body = {
        'message': f'Save Socrates experiment {run_id[:12]}',
        'content': base64.b64encode(compressed).decode('ascii'),
    }
    url = f'{API_ROOT}/repos/{repo}/contents/{quote(path, safe="/")}'
    # The initial write may succeed even if the client times out. A read-first
    # retry makes saving idempotent; 409 branch conflicts are retried safely.
    for attempt in range(4):
        existing = _request('GET', url, token)
        if existing.status_code == 200:
            existing_data = existing.json()
            return _filename_metadata(
                filename, path, existing_data.get('sha', ''),
                int(existing_data.get('size') or len(compressed)),
            ) or {}
        try:
            response = _request('PUT', url, token, json=body)
            if response.status_code not in (200, 201):
                raise HistoryError(f'Could not save history record: HTTP {response.status_code}.')
            created_sha = response.json().get('content', {}).get('sha', '')
            return _filename_metadata(filename, path, created_sha, len(compressed)) or {}
        except HistoryError as exc:
            error_text = str(exc)
            if attempt == 3 or not any(code in error_text for code in ('409', '422')):
                raise
            time.sleep(0.4 * (attempt + 1))
    raise HistoryError('GitHub could not save the history record after retries.')


def list_runs(repo: str, token: str) -> list[dict[str, Any]]:
    """List all run metadata without downloading the (potentially large) runs."""
    repo = _repo_path(repo)
    root_url = f'{API_ROOT}/repos/{repo}/contents/history'
    root_response = _request('GET', root_url, token)
    if root_response.status_code == 404:
        verify = _request('GET', f'{API_ROOT}/repos/{repo}', token)
        if verify.status_code == 404:
            raise HistoryError('History repository was not found or the GitHub token cannot access it.')
        return []
    try:
        months = [
            item for item in root_response.json()
            if item.get('type') == 'dir' and re.fullmatch(r'\d{4}-\d{2}', item.get('name', ''))
        ]
    except (ValueError, TypeError) as exc:
        raise HistoryError('GitHub returned an invalid history directory listing.') from exc
    records = []
    for month in sorted(months, key=lambda item: item['name'], reverse=True):
        response = _request('GET', f'{root_url}/{month["name"]}', token)
        if response.status_code == 404:
            continue
        items = response.json()
        if not isinstance(items, list):
            continue
        for item in items:
            if item.get('type') != 'file':
                continue
            metadata = _filename_metadata(
                item.get('name', ''), item.get('path', ''),
                item.get('sha', ''), int(item.get('size', 0)),
            )
            if metadata is not None:
                records.append(metadata)
    return sorted(records, key=lambda item: (item['created_at'], item['run_id']), reverse=True)


def load_run(repo: str, token: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch by immutable Git object SHA, supporting compressed blobs over 1 MB."""
    repo = _repo_path(repo)
    sha = str(metadata.get('sha') or '')
    if not SHA_PATTERN.fullmatch(sha):
        raise HistoryError('The history record has no valid Git blob SHA.')
    response = _request('GET', f'{API_ROOT}/repos/{repo}/git/blobs/{sha}', token)
    if response.status_code == 404:
        raise HistoryError('This saved run is no longer available in the history repository.')
    try:
        blob = response.json()
        if blob.get('encoding') != 'base64':
            raise HistoryError('GitHub did not return a base64 encoded history record.')
        contents = base64.b64decode(blob['content'])
        payload = json.loads(gzip.decompress(contents).decode('utf-8'))
        if payload.get('schema_version') != 2 or payload.get('run_id') != metadata['run_id']:
            raise HistoryError('Saved history has an incompatible or invalid format.')
        if not isinstance(payload.get('config'), dict) or not isinstance(payload.get('run_data'), dict):
            raise HistoryError('Saved history has missing configuration or results.')
        return payload
    except (ValueError, KeyError, OSError, TypeError, binascii.Error) as exc:
        raise HistoryError('The saved run could not be decoded or is corrupted.') from exc
