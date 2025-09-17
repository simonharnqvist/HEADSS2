import pytest
from pathlib import Path

tests_dir = Path(__file__).parent
pytest.main([str(tests_dir)])
