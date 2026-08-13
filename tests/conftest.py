import os
import sys

# Make `src` importable as a package regardless of the invoking cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
