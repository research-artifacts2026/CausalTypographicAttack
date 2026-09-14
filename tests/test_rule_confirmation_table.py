import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('confirmation_table', ROOT / 'scripts/make_rule_confirmation_table.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EVIDENCE = ROOT / 'evidence/rule_explicit_confirmation_n128/analysis.json'


def test_replay_preserves_positive_and_null_models(tmp_path):
    result = module.generate(EVIDENCE, tmp_path)
    a, b = result['results']
    assert (a['plus'], a['minus'], a['holm_p']) == (15, 0, 0.0001220703125)
    assert (b['plus'], b['minus'], b['holm_p']) == (9, 9, 1.)
    assert (tmp_path / 'generated_table.tex').is_file()


@pytest.mark.parametrize('mutation', ['missing_model', 'changed_count', 'incomplete'])
def test_rejects_incomplete_or_inconsistent_evidence(tmp_path, mutation):
    data = json.loads(EVIDENCE.read_text())
    if mutation == 'missing_model': data['results'].pop()
    elif mutation == 'changed_count': data['results'][0]['clean_correct_subset']['rule_false']['target_count'] -= 1
    else: data['actual_victim_calls'] -= 1
    damaged = tmp_path / 'damaged.json'
    damaged.write_text(json.dumps(data))
    with pytest.raises(ValueError): module.generate(damaged, tmp_path / 'out')
