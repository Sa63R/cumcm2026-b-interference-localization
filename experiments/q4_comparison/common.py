from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
for name in ('vendor_v4', 'vendor_inference'):
    sys.path.insert(0, str(ROOT / name))
