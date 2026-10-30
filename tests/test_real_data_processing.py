"""Adapter tests using small artificial rows with the real file schema."""
import csv
import tempfile
import unittest
from pathlib import Path
from data_processing import prepare_detector, load_site_series

class DetectorPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.source=self.root/'VSDATA_20261001.csv'; self.output=self.root/'out.csv'
        self.fields=['NB_SCATS_SITE','QT_INTERVAL_COUNT','NB_DETECTOR']+[f'V{i:02d}' for i in range(96)]
    def tearDown(self): self.tmp.cleanup()
    def write(self,missing=False,duplicate=False):
        rows=[]
        for detector,value in [('1','3'),('2','8')]:
            row={k:value for k in self.fields[3:]};row.update(NB_SCATS_SITE='2000',QT_INTERVAL_COUNT='2026-10-01',NB_DETECTOR=detector)
            if missing and detector=='1': row['V01']='-1'
            rows.append(row)
        if duplicate: rows.append(rows[0])
        with self.source.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=self.fields);writer.writeheader();writer.writerows(rows)
    def test_selected_detector_is_not_summed(self):
        self.write();r=prepare_detector(self.root,'2000','1',self.output)
        stamps, values=load_site_series(self.output,'2000')
        self.assertEqual(len(stamps),96);self.assertEqual(stamps[1],'2026-10-01T00:15:00');self.assertEqual(set(values),{3.0});self.assertEqual(r['detector'],'1')
    def test_missing_requires_explicit_policy(self):
        self.write(missing=True)
        with self.assertRaisesRegex(ValueError,'contiguous'): prepare_detector(self.root,'2000','1',self.output)
        self.assertFalse(self.output.exists())
    def test_longest_segment_reports_discarded(self):
        self.write(missing=True);r=prepare_detector(self.root,'2000','1',self.output,True)
        self.assertEqual(r['invalid_or_missing_values'],1);self.assertEqual(r['retained_observations'],94);self.assertEqual(r['valid_observations_discarded'],1)
    def test_duplicate_day_is_rejected(self):
        self.write(duplicate=True)
        with self.assertRaisesRegex(ValueError,'Duplicate'):prepare_detector(self.root,'2000','1',self.output)
if __name__=='__main__': unittest.main()
