"""Make the repository root importable as `src...` when running pytest
from anywhere, mirroring the sys.path setup already used by the example
scripts in examples/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
