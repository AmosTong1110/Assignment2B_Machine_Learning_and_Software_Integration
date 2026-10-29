"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_14_disconnected_route_returns_empty(self):
        self.assertEqual(self.routes(origin='D', destination='A'), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
