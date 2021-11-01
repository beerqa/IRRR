# download the wiki dump file
mkdir -p data
wget https://nlp.stanford.edu/projects/beerqa/enwiki-20200801-pages-articles-tokenized.tar.bz2 -O data/enwiki-20200801-pages-articles-tokenized.tar.bz2
# verify that we have the whole thing
unameOut="$(uname -s)"
case "${unameOut}" in
    Darwin*)    MD5SUM="md5 -r";;
    *)          MD5SUM=md5sum
esac
if [ `$MD5SUM data/enwiki-20200801-pages-articles-tokenized.tar.bz2 | awk '{print $1}'` == "d7b50ec812c164681b4b2a7f5dfde898" ]; then
    echo "Downloaded the processed Wikipedia dump from the BeerQA website. Everything's looking good, so let's extract it!"
else
    echo "The md5 doesn't seem to match what we expected, try again?"
    exit 1
fi
cd data
tar -xjvf enwiki-20200801-pages-articles-tokenized.tar.bz2
# clean up
rm enwiki-20200801-pages-articles-tokenized.tar.bz2
echo 'Done!'
