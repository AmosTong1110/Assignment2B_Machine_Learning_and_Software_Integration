"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing import find_routes
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_15_missing_hourly_flow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing hourly flow'):
            find_routes(self.edges, 'A', 'D', {}, self.converter)


if __name__ == "__main__":
    unittest.main(verbosity=2)
