from app import push_target, format_result

def test_private_chat_only():
    assert push_target({'type':'user','userId':'Uabcdef'}) == 'Uabcdef'
    assert push_target({'type':'group','groupId':'Cxyz','userId':'Uabcdef'}) is None

def test_output_never_claims_safe():
    out = format_result({'label':'未偵測到明顯線索（不代表安全）','ocr_status':'empty'})
    assert '不代表安全' in out and '研究模型' in out
