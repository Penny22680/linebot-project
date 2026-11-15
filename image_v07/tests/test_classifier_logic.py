from classifier import triage, combine_scores

def test_frozen_math():
    assert abs(combine_scores(.2, .7) - .6) < 1e-9
    assert combine_scores(.42, None) == .42
    assert triage(.9, .8, .82) == '疑似詐騙'
    assert triage(.2, .1, .12).startswith('未偵測到明顯')
    assert triage(.9, .1, .26) == '需要人工確認'
    assert triage(.9, None, .9) == '需要人工確認'
    assert triage(.52, .51, .512) == '需要人工確認'
