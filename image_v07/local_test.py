"""Test the real trained DeiT/OCR weights offline, without a LINE credential."""
import argparse
import json
from pathlib import Path

from classifier import FusionEngine

p = argparse.ArgumentParser()
p.add_argument('--image', required=True, type=Path)
args = p.parse_args()
engine = FusionEngine()
print('Device:', engine.device)
print(json.dumps(engine.evaluate(args.image.read_bytes()), ensure_ascii=False, indent=2))
