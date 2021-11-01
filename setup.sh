set -e

python -c 'import sys; print(sys.version_info[:])'
echo "Please make sure you are running python version 3.6.X"

echo "Installing required Python packages..."
python -m pip install -r requirements.txt

echo "Downloading models..."
bash scripts/download_irrr_models.sh

echo "Getting BeerQA dataset..."
bash scripts/download_qa_data.sh

echo "Download CoreNLP..."
bash scripts/download_corenlp.sh

echo "Downloading Elasticsearch..."
bash scripts/download_elastic_6.7.sh

echo "NOTE: we set jvm options -Xms and -Xmx for Elasticsearch to be 4GB"
echo "We suggest you set them as large as possible in: elasticsearch-6.7.0/config/jvm.options"
cp search/jvm.options elasticsearch-6.7.0/config/jvm.options

echo "Downloading wikipedia source documents..."
bash scripts/download_processed_wiki.sh

echo "Running Elasticsearch and indexing Wikipedia documents..."
bash scripts/launch_elasticsearch_6.7.sh
python -m scripts.index_processed_wiki
