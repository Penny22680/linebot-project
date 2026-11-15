import unittest
from evidence_rules import find_text_clues
from app import format_result


class ImageTextExplanationTests(unittest.TestCase):
    def test_investment_sample(self):
        text = '每週固定收益\n18%\n專人帶單操作\n保證獲利 , 絕不虧損\n電話 0912345678'
        clues = find_text_clues(text)
        self.assertEqual(len(clues), 3)
        result = format_result({'label': '疑似詐騙', 'ocr_status': 'ok', 'text_clues': clues})
        self.assertIn('保證獲利', result)
        self.assertIn('固定收益', result)
        self.assertNotIn('0912345678', result)

    def test_cjk_spacing_and_limit(self):
        clues = find_text_clues('保 證 獲 利\n立 即 驗 證\n24 小 時 後 失 效')
        self.assertEqual(len(clues), 3)
        self.assertEqual(clues[0]['quote'], '保證獲利')

    def test_no_invented_evidence(self):
        self.assertEqual(find_text_clues('生日快樂，祝你天天開心'), [])
        message = format_result({'label': '需要人工確認', 'ocr_status': 'ok', 'text_clues': []})
        self.assertIn('未找到本版規則', message)
        self.assertNotIn('保證獲利', message)

    def test_failed_ocr_never_shows_clues(self):
        message = format_result({'label': '需要人工確認', 'ocr_status': 'unavailable',
                                 'text_clues': [{'quote': '保證獲利', 'explanation': 'test'}]})
        self.assertNotIn('保證獲利', message)
        self.assertIn('無法提供具體文字線索', message)

    def test_negation(self):
        self.assertEqual(find_text_clues('這個產品不保證獲利'), [])

    def test_at_most_three(self):
        self.assertLessEqual(len(find_text_clues('保證獲利 固定收益 專人帶單 立即驗證 24小時後失效 先匯款')), 3)


if __name__ == '__main__':
    unittest.main()
