"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_05_chronological_split(self):
        r = self.train()
        self.assertLess(r['train_end'], r['test_start'])
        self.assertEqual(r['test_count'], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
