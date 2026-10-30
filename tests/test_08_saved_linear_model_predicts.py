"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import load_site_series, predict_next
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_08_saved_linear_model_predicts(self):
        self.train()
        _, values = load_site_series(self.data, 'A')
        prediction = predict_next(self.model_dir, 'linear', 'A', values)
        self.assertGreaterEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
