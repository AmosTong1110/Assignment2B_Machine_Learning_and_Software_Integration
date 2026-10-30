"""One automated test using fictional input data."""
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_12_routes_are_ranked_and_distinct(self):
        routes = self.routes()
        self.assertEqual([r.minutes for r in routes], [3.0, 3.5, 4.5])
        self.assertEqual(len({r.sites for r in routes}), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
