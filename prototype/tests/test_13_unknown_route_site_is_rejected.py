"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_13_unknown_route_site_is_rejected(self):
        with self.assertRaises(ValueError):
            self.routes(destination='unknown')


if __name__ == "__main__":
    unittest.main(verbosity=2)
