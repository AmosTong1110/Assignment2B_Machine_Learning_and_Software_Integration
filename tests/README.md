# Test files

Each `test_*.py` file has one test case. `support.py` supplies the shared fictional data and temporary setup; `__init__.py` lets Python import the test folder. Run the commands from the project root (the folder containing `forecast.py` and `routing.py`).

Run all 15: `python -m unittest discover -s tests -p 'test_*.py' -v`

Run one: `python -m unittest -v tests.test_01_load_regular_site_data`

Alternatively: `python tests/test_01_load_regular_site_data.py`

The tests are for the fictional prototype, not the real assignment. The LSTM and GRU tests are skipped unless PyTorch is installed. See the main README for other limitations.
