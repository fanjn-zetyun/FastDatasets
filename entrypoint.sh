#!/bin/bash
set -e

echo "Building FastDatasets command..."
python3 /app/cmd_builder.py

echo "Starting FastDatasets..."
bash /tmp/fastdatasets_cmd.sh

echo "FastDatasets completed successfully!"
