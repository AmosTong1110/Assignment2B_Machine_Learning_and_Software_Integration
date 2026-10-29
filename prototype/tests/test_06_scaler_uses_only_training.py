"""One automated test using fictional input data."""
import json
import sys
import unittest
from pathlib import Path

# Supports test discovery and direct execution from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import _artifact_path
from tests.support import PrototypeFixture


class TestCase(PrototypeFixture, unittest.TestCase):
    def test_06_scaler_uses_only_training(self):
        rows = [r[:] for r in self.rows]
        rows[-1][2] = '9999'
        self.write_rows(rows)
        self.train()
        artifact = json.loads(_artifact_path(self.model_dir, 'linear', 'A').read_text())
        self.assertLess(artifact['train_min'] + artifact['train_range'], 9999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
