# in tests/conftest.py

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# If the CI environment variable is set, we're in CI so mock the torch package.
if os.getenv("CI") == "true":
    sys.modules["torch"] = Mock()

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Test skipped because it requires CUDA"
)
