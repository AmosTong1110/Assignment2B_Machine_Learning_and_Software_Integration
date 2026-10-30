import unittest, tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from prepare_2006 import prepare_2006
from data_processing import load_site_series
class WorkbookAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.output=Path(self.temp.name)/'out.csv'
        self.row={'SCATS Number':'0970','Location':'Example','HF VicRoads Internal':6246,'Date':datetime(2006,10,1,0,15)}
        self.row.update({f'V{i:02d}':i for i in range(96)})
    def tearDown(self):self.temp.cleanup()
    def test_midnight_and_leading_zero(self):
        with patch('prepare_2006.iter_rows',return_value=iter([self.row])):r=prepare_2006('unused','0970','Example','6246',self.output)
        ts,values=load_site_series(self.output,'0970');self.assertEqual(ts[0],'2006-10-01T00:00:00');self.assertEqual(ts[1],'2006-10-01T00:15:00');self.assertEqual(values[1],1);self.assertEqual(r['retained_observations'],96)
    def test_duplicate_key_rejected(self):
        with patch('prepare_2006.iter_rows',return_value=iter([self.row,self.row])):
            with self.assertRaisesRegex(ValueError,'Duplicate'):prepare_2006('unused','0970','Example','6246',self.output)
    def test_hf_id_selects_exact_series(self):
        other=dict(self.row);other['HF VicRoads Internal']=999
        with patch('prepare_2006.iter_rows',return_value=iter([self.row,other])):r=prepare_2006('unused','0970','Example','6246',self.output)
        self.assertEqual(r['days'],1)
    def test_missing_day_requires_explicit_policy(self):
        later=dict(self.row);later['Date']=datetime(2006,10,3)
        with patch('prepare_2006.iter_rows',return_value=iter([self.row,later])):
            with self.assertRaisesRegex(ValueError,'continuous'):prepare_2006('unused','0970','Example','6246',self.output)
if __name__=='__main__':unittest.main()
