#!/bin/sh
# Build the processed data if it is missing, then start the dashboard.
set -e

if [ ! -f data/processed/sepsis_events.csv ] || [ ! -f data/processed/sepsis_cases.csv ] || [ ! -f data/processed/sepsis_clean.csv ]; then
  echo "Building the data (first start only)..."
  [ -f data/raw/sepsis.xes ] || python src/download_data.py
  python src/clean_data.py
  python src/prepare_log.py
fi

exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
