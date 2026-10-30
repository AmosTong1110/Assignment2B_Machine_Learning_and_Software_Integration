"""Shared fictional data and setup for the 15 prototype tests.

These fixtures are educational examples; they are not the course dataset.
"""
import csv
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from forecast import train_and_evaluate
from routing import Edge, find_routes


class PrototypeFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data.csv'
        self.model_dir = self.root / 'models'
        self.rows = [
            ['A', (datetime(2026, 1, 1) + timedelta(minutes=15*i)).isoformat(), str(50+i%16)]
            for i in range(80)
        ]
        self.write_rows(self.rows)
        self.edges = [Edge('A', 'B', 1), Edge('B', 'D', 1), Edge('A', 'C', 1.5),
                      Edge('C', 'D', 1), Edge('A', 'D', 4)]
        self.flow = {'B': 100, 'C': 100, 'D': 100}
        # Fictional test-only formula. It is not the course conversion.
        self.converter = lambda edge, flow: edge.distance_km + 0.5

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows):
        with self.data.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['site_id', 'timestamp', 'flow'])
            writer.writerows(rows)

    def train(self, model='linear'):
        return train_and_evaluate(self.data, 'A', model, self.model_dir,
                                  window=4, epochs=1, train_fraction=0.75)

    def routes(self, origin='A', destination='D', **kwargs):
        return find_routes(self.edges, origin, destination, self.flow, self.converter, **kwargs)
