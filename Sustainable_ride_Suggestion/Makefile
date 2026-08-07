# Convenience targets. On Windows without make, run the underlying commands
# directly -- each is a single `python -m sustainable_ride.cli` invocation.

PYTHON ?= python

.PHONY: help install download prepare train evaluate pipeline synthetic test serve app clean

help:
	@echo "install    install dependencies"
	@echo "pipeline   download -> prepare -> train -> evaluate (real data, ~420MB)"
	@echo "synthetic  same pipeline on generated data (no download)"
	@echo "test       run the test suite"
	@echo "serve      start the API on :8000"
	@echo "app        start the Streamlit demo"

install:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) -m sustainable_ride.cli download

prepare:
	$(PYTHON) -m sustainable_ride.cli prepare

train:
	$(PYTHON) -m sustainable_ride.cli train

evaluate:
	$(PYTHON) -m sustainable_ride.cli evaluate

pipeline:
	$(PYTHON) -m sustainable_ride.cli pipeline

synthetic:
	$(PYTHON) -m sustainable_ride.cli pipeline --synthetic

test:
	$(PYTHON) -m pytest

serve:
	$(PYTHON) -m sustainable_ride.cli serve

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/interim/* data/processed/* models/*.joblib reports/figures/*.png
