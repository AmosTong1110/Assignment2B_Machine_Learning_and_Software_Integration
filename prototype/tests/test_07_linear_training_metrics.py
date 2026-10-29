"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_07_linear_training_metrics(self):
        r = self.train()
        self.assertGreaterEqual(r['mae'], 0)
        self.assertGreaterEqual(r['rmse'], 0)
        self.assertEqual(r['train_count'], 56)


if __name__ == "__main__":
    unittest.main(verbosity=2)
