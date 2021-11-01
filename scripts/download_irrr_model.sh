mkdir -p downloads/model
pushd downloads/model
wget https://nlp.stanford.edu/projects/beerqa/irrr_models.tar.gz
tar -xzvf irrr_models.tar.gz
popd
