"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import load_site_series
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_02_unknown_site_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No records'):
            load_site_series(self.data, 'missing')


if __name__ == "__main__":
    unittest.main(verbosity=2)
