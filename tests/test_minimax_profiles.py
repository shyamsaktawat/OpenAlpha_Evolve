"""Exercise documented profiles through LiteLLM's real HTTP serialization."""
import importlib
import json
import re
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from code_generator.agent import CodeGeneratorAgent
from config import settings


ROOT = Path(__file__).resolve().parents[1]
PROFILES = re.findall(
    r'# LITELLM_DEFAULT_MODEL="([^"]+)"\n'
    r'# LITELLM_DEFAULT_BASE_URL="([^"]+)"',
    (ROOT / '.env.example').read_text(),
)


def test_documented_profiles():
    assert len(PROFILES) == 4
    readme = (ROOT / 'README.md').read_text()
    for model, base_url in PROFILES:
        assert model.split('/', 1)[1] == 'MiniMax-M3.1-Flash'
        assert f'LITELLM_DEFAULT_MODEL="{model}"' in readme
        assert f'LITELLM_DEFAULT_BASE_URL="{base_url}"' in readme


@pytest.mark.parametrize('model,base_url', PROFILES)
@pytest.mark.parametrize('selection', ['default', 'm3', 'saved_legacy', 'custom_override'])
def test_profile_http_boundary(monkeypatch, model, base_url, selection):
    import asyncio

    provider = model.split('/', 1)[0]
    configured = model
    override = None
    if selection == 'm3':
        configured = f'{provider}/MiniMax-M3'
    elif selection == 'saved_legacy':
        configured = f'{provider}/MiniMax-M2.7'
    elif selection == 'custom_override':
        override = f'{provider}/custom-saved-model'
    monkeypatch.setenv('LITELLM_DEFAULT_MODEL', configured)
    monkeypatch.setenv('LITELLM_DEFAULT_BASE_URL', base_url)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'offline-test-key')
    captured = []

    async def send(client, request, **kwargs):
        body = json.loads(request.content)
        captured.append((str(request.url), body))
        if provider == 'anthropic':
            response = {
                'id': 'msg_offline', 'type': 'message', 'role': 'assistant',
                'model': body['model'], 'content': [{'type': 'text', 'text': 'pass'}],
                'stop_reason': 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 1, 'output_tokens': 1},
            }
        else:
            response = {
                'id': 'offline', 'object': 'chat.completion', 'created': 1,
                'model': body['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'pass'},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
        return httpx.Response(200, json=response, request=request)

    original = vars(settings).copy()
    try:
        importlib.reload(settings)
        with patch.object(httpx.AsyncClient, 'send', send), patch.object(
            httpx.Client, 'send', side_effect=AssertionError('Unexpected sync HTTP request')
        ):
            result = asyncio.run(CodeGeneratorAgent().generate_code('Write code.', model_name=override))
        assert result == 'pass'
        assert len(captured) == 1
        url, payload = captured[0]
        suffix = '/v1/messages' if provider == 'anthropic' else '/chat/completions'
        assert url == base_url + suffix
        assert payload['model'] == (override or configured).split('/', 1)[1]
        content = [{'type': 'text', 'text': 'Write code.'}] if provider == 'anthropic' else 'Write code.'
        assert payload['messages'] == [{'role': 'user', 'content': content}]
        for unsupported in ('thinking', 'reasoning_effort', 'output_config', 'service_tier', 'reasoning_split'):
            assert unsupported not in payload
    finally:
        vars(settings).update(original)


def test_non_minimax_http_boundary(monkeypatch):
    # The generic provider path and explicit model selection remain unchanged.
    test_profile_http_boundary(monkeypatch, 'openai/gpt-4o', 'https://api.openai.com/v1', 'default')
