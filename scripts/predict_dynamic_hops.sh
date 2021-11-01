#! /bin/bash

# Instructions:
# - make sure you have already run `setup.sh` and are using the correct python environment
# - call this script from the root directory of this project
# - please feel free to modify any of the script inputs below

set -e  # stop script if any command fails

# Script parameters for users (feel free to edit):
OUTDIR=${1}  # suggested convention is "[dataset_name]_eval"
QUESTION_FILE=${2}
IRRR_MODEL=${3}
TOP_N=${4:-10}
MAX_HOPS=${5:-3}
MODEL_SCOPE=${6:-electra}
INDEX=${7:-latest_wiki_doc_para}

ADV=3
DEBUG=false
RECOMPUTE_ALL=true  # change to `true` to force recompute everything

ROOT_DIR=$PWD

MUST_DO=true # added for debugging purposes

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

export CORENLP_HOME=`realpath stanford-corenlp-full-2018-10-05`

REAL_OUTDIR=`realpath $OUTDIR`

echo "Placing temporary evaluation files in: $OUTDIR"
mkdir -p $OUTDIR

echo "Trying to connect to ES at localhost:9200..."
if ! curl -s -I localhost:9200 > /dev/null;
then
  echo 'running "sh scripts/launch_elasticsearch_6.7.sh"'
  sh scripts/launch_elasticsearch_6.7.sh
  while ! curl -I localhost:9200;
  do
    sleep 2;
  done
fi
echo "ES is up and running"


# Convert input data into SQuAD format
mkdir -p $OUTDIR/hop0
cp $QUESTION_FILE $OUTDIR/hop0/input_orig.json

python -m scripts.e_to_e_helpers.squadify_hop1_questions $QUESTION_FILE $OUTDIR/hop0/input.json

for hop in `seq 0 $MAX_HOPS`; do
    cd model
    echo "Running reranker for hop${hop} and generating queries for hop$((hop+1))..."
    python run_irrr.py --vocab_file $IRRR_MODEL/vocab.txt \
                       --bert_config_file $IRRR_MODEL/bert_config.json \
                       --init_checkpoint $IRRR_MODEL/model.ckpt \
                       --do_train=False \
                       --do_predict=True \
                       --predict_batch_size=16 \
                       --ranking_candidates=1 \
                       --output_dir $REAL_OUTDIR/hop${hop} \
                       --qg_reader_predict_file $REAL_OUTDIR/hop${hop}/input.json \
                       --advantage $ADV \
                       --use_fp16=False \
                       --ranking_candidates=1 \
                       --verbose_logging=False \
                       --do_lower_case=True --debug=$DEBUG --model_scope=$MODEL_SCOPE
    rm $REAL_OUTDIR/hop${hop}/eval.tf_record
    cd -
        
    if $MUST_DO || $RECOMPUTE_ALL || [ ! -f $REAL_OUTDIR/hop${hop}/retrieved_titles.json ]
    then
        echo "Querying ES with hop$((hop+1)) predicted queries"
  
        mkdir -p $OUTDIR/hop$((hop+1))
        if [ $hop -ge 0 ]; then
          python -m scripts.e_to_e_helpers.merge_with_es \
          $REAL_OUTDIR/hop${hop}/query_predictions_best.json \
          $REAL_OUTDIR/hop${hop}/query_predictions.json \
          $QUESTION_FILE \
          $OUTDIR/hop${hop}/input.json \
          $OUTDIR/hop$((hop+1))/input_orig.json \
          $OUTDIR/hop${hop}/recall_metrics.txt \
          $OUTDIR/hop${hop}/retrieved_titles.json \
          --top_n=$TOP_N \
          --index $INDEX --include_prev --prev_titles $OUTDIR/hop$((hop-1))/retrieved_titles.json
        fi
        python -m scripts.e_to_e_helpers.convert_to_irrr_input $QUESTION_FILE $OUTDIR/hop$((hop+1))/input_orig.json $OUTDIR/hop$((hop+1))/input.json --keep_all
  
        echo "Created Hop$((hop+1)) input:"
    else
        echo "Using existing Hop$((hop+1)) input:"
    fi
    ls -la $REAL_OUTDIR/hop$((hop+1))/input.json        
done

python -m utils.merge_answers $REAL_OUTDIR/hop $MAX_HOPS $REAL_OUTDIR



