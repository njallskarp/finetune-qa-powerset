
import os

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    torch.cuda.is_available() or os.environ.get('CI') == 'true',
    reason="Requires CUDA"
)