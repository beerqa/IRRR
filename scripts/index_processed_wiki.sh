#! /bin/bash

# Instructions:
# - make sure you have already run `setup.sh` and are using the correct python environment
# - call this script from the root directory of this project
# - please feel free to modify any of the script inputs below

set -e  # stop script if any command fails

echo "Running Elasticsearch and indexing Wikipedia documents..."
bash scripts/launch_elasticsearch_6.7.sh

python -m scripts.index_processed_wiki $@

echo "Done Indexing $1 Wiki!"
