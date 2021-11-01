
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy as np
import sys
import glob
import shutil
from tokenization import _normalize

flags = tf.flags

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
    
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "model_scope", None,
    "Scope name of the BERT model E.g. bert or electra")
    
flags.DEFINE_string(
    "qg_reader_train_file", None,
    "Hotpot Reader training data files")
    
flags.DEFINE_string(
    "squad_reader_train_file", None,
    "SQuAD reader training data files")
    
flags.DEFINE_string(
    "qg_reader_predict_file", None,
    "Query generation prediction file")
    
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "debug", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
    
flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
    
flags.DEFINE_integer(
    "doc_stride", 384,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool(
    "do_train", False, 
    "Whether to run training.")
    
flags.DEFINE_bool(
    "do_predict", False, 
    "Whether to run eval on the dev set.")

flags.DEFINE_integer(
    "train_batch_size", 32, 
    "Total batch size for training.")
    
flags.DEFINE_integer(
    "predict_batch_size", 32,
    "Total batch size for predictions.")

flags.DEFINE_float(
    "learning_rate", 5e-5, 
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1, 
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
    
flags.DEFINE_integer(
    "n_best_size", 40,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 20,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")
      
flags.DEFINE_float(
    "null_score_diff_threshold", 10000.0,
    "If null_score - best_non_null is greater than the threshold predict null.")
      

flags.DEFINE_bool(
    "verbose_logging", True, 
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")
    
flags.DEFINE_integer(
    "save_checkpoints_steps", 1000, 
    "How often to save the model checkpoint.")
    
flags.DEFINE_integer(
    "iterations_per_loop", 1000, 
    "How many steps to make in each estimator call.")
    
flags.DEFINE_integer(
    "choice_size", 1, 
    "Max number of reasoning path choice at each hop")
    
flags.DEFINE_integer(
    "ranking_candidates", 5, 
    "The number of cadidates for reasoning path.")
        
flags.DEFINE_integer(
    "num_iteration", 100000, "Num of iteration")
    
flags.DEFINE_integer(
    "num_accumulation", 1,"Num of acculuation")
    
flags.DEFINE_integer(
    "random_seed", 10000, 
    "Random seed.")

flags.DEFINE_bool(
    "no_empty_query_loss", False, 
    "Whether to minimize loss of query generation for impossible cases.")

flags.DEFINE_bool(
    "merge_queries", False, 
    "Whether to merge queries from multiple reasoning path ")

flags.DEFINE_bool(
    "short", True, 
    "Whether to use short oracle queries")

flags.DEFINE_bool(
    "rank_network", False, 
    "Whether to use additional transformer network for reranking")     
  
flags.DEFINE_bool(
    "horovod", False, 
    "Whether to use Horovod for multi-gpu runs")

flags.DEFINE_float(
    "advantage", 2.0, 
    "Amount of boosting of positive log probs for query generation")

flags.DEFINE_bool(
    "prev_query", False, 
    "Whether to use a previous query embedding for query generation model") 

flags.DEFINE_bool(
    "soft_target", True, 
    "Whether to use 1/10 target for non-gold paragraph with connection to the gold ") 

flags.DEFINE_bool(
    "use_distant", True, 
    "Whether to use distant supervision of reader model (non-gold paragraph with answer)") 

flags.DEFINE_bool(
    "use_xla", True, 
    "Whether to enable XLA JIT compilation.")

flags.DEFINE_bool(
    "use_fp16", True, 
    "Whether to use fp32 or fp16 arithmetic on GPU.")
                    
flags.DEFINE_integer(
    "split_count", 1, 
    "Max number of reasoning path choice at each hop")
    
FLAGS = flags.FLAGS

def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_predict:
    if not FLAGS.qg_reader_predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))
