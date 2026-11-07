import numpy as np
import pytest

@pytest.fixture
def dummy_radiograph():
    """Синтетический рентгеновский снимок (640x640, полутоновый)."""
    return np.random.randint(0, 256, (640, 640), dtype=np.uint8)