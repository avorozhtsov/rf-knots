import os
import sys
from unittest.mock import patch
from compact_child_env_v31a import child_env


def test_pinned_spawn_override_keeps_explicit_reader():
    markers = dict(RF_KNOTS_COMPACT_V31='1', RF_KNOTS_COMPACT_ROOT='/example', RF_KNOTS_COMPACT_WRITE='1')
    given = {'PYTHONPATH':'src:scripts', 'other':'kept'}
    with patch.dict(os.environ, markers):
        result = child_env([sys.executable, 'script.py'], given, '/example')
    assert result['PYTHONPATH'] == '/example/scripts/compact_runtime_v31a:src:scripts'
    assert result['RF_KNOTS_COMPACT_WRITE'] == '1'
    assert result['other'] == 'kept' and given == {'PYTHONPATH':'src:scripts', 'other':'kept'}


def test_non_python_subprocess_unchanged():
    env = {'PATH':'same'}
    assert child_env(['ps','-axo','command='], env, '/example') is env
