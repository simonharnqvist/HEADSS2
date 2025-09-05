import glob
import os
import pytest

base_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(base_dir, "tests")
test_modules = glob.glob(os.path.join(tests_dir, "test_regions_*.py"))

pytest.main(test_modules)
