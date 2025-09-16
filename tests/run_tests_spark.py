import glob
import os
import pytest

base_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(base_dir, "tests")
test_modules = glob.glob(tests_dir)

#pytest.main(test_modules)
pytest.main(["tests/test_merging.py"
             ])