"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import load_site_series
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_03_invalid_flow_is_rejected(self):
        rows = [r[:] for r in self.rows]
        rows[2][2] = 'bad'
        self.write_rows(rows)
        with self.assertRaises(ValueError):
            load_site_series(self.data, 'A')


if __name__ == "__main__":
    unittest.main(verbosity=2)
