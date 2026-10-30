"""CareFlow dashboard package.

Importing the package puts the project root and `src/` on the import path, so the analysis modules in
`src/` (which use plain imports such as `import bundle`) work from the app, the tests and notebooks alike.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
