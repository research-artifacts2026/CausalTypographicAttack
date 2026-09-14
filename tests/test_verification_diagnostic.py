import pytest
from scripts.analyze_verification_diagnostic import paired_test


def test_paired_exact_test_uses_discordances_not_marginals():
    t=paired_test([0]*15+[1]*5,[1]*20,['coco']*10+['voc']*10)
    assert (t['plus'],t['minus'])==(15,0)
    assert t['p']==pytest.approx(0.00006103515625)
    assert t['delta']==.75


def test_equal_marginal_counts_retain_opposing_discordances():
    t=paired_test([1,0,1,0],[0,1,0,1],['coco','coco','voc','voc'])
    assert (t['plus'],t['minus'])==(2,2)
    assert t['p']==1
    assert t['delta']==0
