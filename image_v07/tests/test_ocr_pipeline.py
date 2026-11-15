"""Offline tests; no tokens, model weights, or live LINE requests required."""
import unittest
from unittest.mock import patch
from PIL import Image

from ocr_pipeline import OCRResult, _quality, extract_text


class OCRPipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (320, 200), 'white')

    def test_high_quality_first_pass_is_accepted(self):
        with patch('ocr_pipeline._extract_once', return_value=('您好 這是一段測試文字 123456', 15, 87.0)) as mock:
            result = extract_text(self.image)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.method, 'psm6')
        # Short text retries to check alternative sparse layout, conservatively.
        self.assertEqual(mock.call_count, 2)

    def test_long_high_quality_does_not_retry(self):
        with patch('ocr_pipeline._extract_once', return_value=('銀行官方客服訊息這是一整段待確認的內容', 25, 82.0)) as mock:
            result = extract_text(self.image)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(mock.call_count, 1)

    def test_fallback_can_recover_empty_first_pass(self):
        def fake(_image, psm):
            return ('', 0, None) if psm == 6 else ('請登入官方網站進行確認', 11, 68.0)
        with patch('ocr_pipeline._extract_once', side_effect=fake):
            result = extract_text(self.image)
        self.assertEqual((result.status, result.method), ('ok', 'psm11'))

    def test_no_text_is_empty_and_manual_review(self):
        with patch('ocr_pipeline._extract_once', return_value=('', 0, None)):
            result = extract_text(self.image)
        self.assertEqual(result.status, 'empty')
        self.assertEqual(result.text, '')

    def test_ocr_exception_is_unavailable(self):
        with patch('ocr_pipeline._extract_once', side_effect=TimeoutError('timeout')):
            result = extract_text(self.image)
        self.assertEqual(result.status, 'unavailable')

    def test_short_garbage_is_not_scored(self):
        with patch('ocr_pipeline._extract_once', return_value=('?', 0, 95.0)):
            result = extract_text(self.image)
        self.assertEqual(result.status, 'empty')
        self.assertEqual(result.text, '')

    def test_low_confidence_is_not_scored(self):
        with patch('ocr_pipeline._extract_once', return_value=('這段文字可能不清楚', 8, 11.0)):
            result = extract_text(self.image)
        self.assertEqual(result.status, 'low_quality')
        self.assertEqual(result.text, '')


if __name__ == '__main__':
    unittest.main()
