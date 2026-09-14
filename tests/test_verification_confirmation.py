import pytest
from scripts.analyze_verification_confirmation import expected_keys, validate_calls
from scripts.register_verification_confirmation import choose_disjoint
from cta.verification_workbench import sha

ARMS=['read_then_verify','self_check']


def test_call_coverage_rejects_duplicate_hiding_missing_cell():
    calls=[{'key':k} for k in sorted(expected_keys(['scene'],ARMS))]
    assert len(calls)==14
    validate_calls(calls,['scene'],ARMS)
    calls[-1]=dict(calls[0])
    with pytest.raises(ValueError,match='duplicate'):
        validate_calls(calls,['scene'],ARMS)


def test_call_coverage_rejects_extra_scene_and_missing_probe():
    calls=[{'key':k} for k in sorted(expected_keys(['scene'],ARMS))]
    with pytest.raises(ValueError): validate_calls(calls[:-1],['scene'],ARMS)
    with pytest.raises(ValueError): validate_calls(calls,['another'],ARMS)


def test_selection_excludes_both_ids_and_duplicate_source_bytes(tmp_path):
    rows=[]; excluded_ids=set(); excluded_hashes=set()
    for family in range(8):
        for index in range(10):
            item=f'scene-{family}-{index}'
            path=tmp_path/(item+'.dat'); path.write_bytes(item.encode())
            rows.append({'item_id':item,'condition':'source_absent',
                         'family':str(family),'source_path':str(path)})
            if index==0: excluded_ids.add(item)
            if index==1: excluded_hashes.add(sha(path))
    selected,hashes=choose_disjoint(rows,excluded_ids,excluded_hashes)
    assert len(selected)==64
    assert not set(hashes)&excluded_ids
    assert not set(hashes.values())&excluded_hashes
    assert {r['item_id'] for r in selected}=={r['item_id'] for r in rows if int(r['item_id'].split('-')[-1])>=2}
