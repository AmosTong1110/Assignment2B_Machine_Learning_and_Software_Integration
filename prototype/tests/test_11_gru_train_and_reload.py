"""One automated test using fictional input data."""
import importlib.util
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import load_site_series, predict_next
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
    def test_11_gru_train_and_reload(self):
        self.train('gru')
        _, values = load_site_series(self.data, 'A')
        self.assertGreaterEqual(predict_next(self.model_dir, 'gru', 'A', values), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
