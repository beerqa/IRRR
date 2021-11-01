# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

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
import irrr_flags
from tqdm import tqdm
from irrr_flags import validate_flags_or_throw
from irrr_data import *
from irrr_data_qg import *
from irrr_data_reader import *
from irrr_data_qg_reader import *
from irrr_model import *
from tokenization import _normalize
  
FLAGS = irrr_flags.FLAGS

def train(bert_config, hvd, run_config, tokenizer):
        
  num_train_steps = FLAGS.num_iteration
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  
  def _get_tfrecord(train_file, train_writer, loader, converter):    
    if hvd:    
      def _get_local_train_files(train_file):      
        input_files = []
        for input_file_dir in train_file.split(","):
          input_files.extend(tf.gfile.Glob(os.path.join(input_file_dir, "*")))
          
        local_train_files = []
        for ti, train_file in enumerate(input_files):
          if ti%hvd.size() == hvd.rank():
            local_train_files.append(train_file)
        return ",".join(local_train_files)
      
      train_files = _get_local_train_files(train_file)
      
      tf.compat.v1.logging.info("Train files " + str(hvd.rank()) + ": " + train_files)
    else:
      train_files = []
      for input_file_dir in train_file.split(","):
        train_files.extend(tf.gfile.Glob(os.path.join(input_file_dir, "*")))
        train_files = ",".join(train_files)
                
    train_examples = loader(
        input_file=train_files, is_training=True)
        
    rng = random.Random(FLAGS.random_seed)
    rng.shuffle(train_examples) 
    
          
    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.

    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    tf.logging.info("***** Running training *****")
    converter(
          examples=train_examples,
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=True,
          output_fn=train_writer.process_feature)
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    del train_examples
      
  steps = 0
  init_checkpoint = FLAGS.init_checkpoint
  late_ckpt = init_checkpoint    
  
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  
  if hvd: 
    train_file = os.path.join(FLAGS.output_dir, "train_" + str(hvd.rank()) + ".tf_record")
  else:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    
  train_writer = FeatureWriter(
      filename=train_file,
      is_training=True)
          
  if FLAGS.squad_reader_train_file != None:
    _get_tfrecord(FLAGS.squad_reader_train_file, 
                  train_writer,
                  read_squad_examples, 
                  convert_reader_examples_to_features)
                  
  if FLAGS.qg_reader_train_file != None:
    _get_tfrecord(FLAGS.qg_reader_train_file, 
                  train_writer,
                  read_qg_reader_examples, 
                  convert_qg_reader_examples_to_features)
                  
  train_writer.close()
  
  train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      drop_remainder=True,
      num_steps=steps)
  if init_checkpoint == None:
      init_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.output_dir)
      
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_one_hot_embeddings=False,
      hvd=None if not FLAGS.horovod else hvd)
 
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config)
    
  training_hooks = []
  if FLAGS.horovod and hvd.size() > 1:
    class MyBroadcast(hvd.BroadcastGlobalVariablesHook):
      def after_run(self, run_context, run_values):
        self.step += 1
        if self.step % 100 == 0:
          self.session.run(self.bcast_op)
          #tf.logging.info("Broadcast Model!!!!! = %d", self.step)
      def after_create_session(self, session, coord):
        self.session = session
        self.step = 0
        session.run(self.bcast_op)

    training_hooks.append(MyBroadcast(0))

  estimator.train(input_fn=train_input_fn, 
                  hooks=training_hooks, 
                  steps=num_train_steps)

def merge_json(files, out_file):  
  res = {}
  for ifile in files:
    with tf.gfile.Open(ifile, "r") as reader:
      items = json.load(reader)
      for key in items.keys():        
        res[key] = items[key]
  
  with tf.gfile.GFile(out_file, "w") as writer:
    writer.write(json.dumps(res, indent=4, ensure_ascii=False))
      
def predict_query_gen_and_reader(bert_config, run_config, tokenizer):
  FLAGS.ranking_candidates = 1
  eval_examples = read_qg_reader_examples(
      input_file=FLAGS.qg_reader_predict_file, is_training=False)
      
  if FLAGS.split_count != 1: 
    eval_size = len(eval_examples)
    split_index = [ int(sp*(eval_size/FLAGS.split_count)) for sp in range(FLAGS.split_count)]
  
    eval_example_list = [eval_examples[s:int(s+eval_size/FLAGS.split_count) + (FLAGS.split_count if si == FLAGS.split_count-1 else 0)] for si, s in enumerate(split_index)]
  else:
    eval_example_list = [eval_examples]
  for evi in range(1, len(eval_example_list)):  
    while eval_example_list[evi][0].qas_id.endswith("_0") == False:
      eval_example_list[evi-1].append(eval_example_list[evi][0])
      del eval_example_list[evi][0]
  
  for evi, eval_examples in enumerate(eval_example_list):
    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []
  
    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)
  
    convert_qg_reader_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)      
        
    while len(eval_features) % FLAGS.predict_batch_size != 0:
      append_feature(eval_features[-1])    
      
    eval_writer.close()
  
    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
  
    all_results = []
  
    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        drop_remainder=False,
        num_steps=0)
  
    # If running eval on the TPU, you will need to specify the number of
    # steps.
    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=0,
      num_warmup_steps=0,
      use_one_hot_embeddings=False,
      hvd=None)
      
    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)
      
    
    all_results_reader = []
    all_results_qg = []
    
    for result in estimator.predict(predict_input_fn, yield_single_examples=True) if FLAGS.verbose_logging else \
             tqdm(estimator.predict(predict_input_fn, yield_single_examples=True), total=len(eval_features)*FLAGS.ranking_candidates):
      if len(all_results_qg) % 100 == 0:
        tf.logging.info("Processing evaluation : %d / %d" % 
                         (len(all_results_qg), len(eval_features*FLAGS.ranking_candidates)))
                         
      unique_id = int(result["unique_ids"])
      query_log_probs = [[float(x[0]),float(x[1])] for x in result["query_log_probs"]]
      na_log_probs = float(result["ranking"])
      reader_log_probs  = float(result["reader_switch"][1])
      
      all_results_qg.append(
          QGRawResult(
              unique_id=unique_id,
              query_log_probs=query_log_probs,
              na_log_probs=na_log_probs,
              reader_log_probs=reader_log_probs))
              
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits   = [float(x) for x in result["end_logits"].flat]
      ans_log_probs = float(result["reader_switch"][0])
      na_log_probs  = float(result["reader_switch"][1])
      yes_log_probs = float(result["reader_switch"][2])
      no_log_probs  = float(result["reader_switch"][3])
  
      all_results_reader.append(
          ReaderRawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits, 
              ans_log_probs=ans_log_probs, 
              na_log_probs=na_log_probs, 
              yes_log_probs=yes_log_probs, 
              no_log_probs=no_log_probs))
    

    if FLAGS.split_count != 1:
      output_prediction_file = os.path.join(FLAGS.output_dir, "answer_predictions_"+str(evi)+".json")
      output_nbest_file = os.path.join(FLAGS.output_dir, "answer_nbest_predictions_"+str(evi)+".json")
      output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds_"+str(evi)+".json")
      output_title_file = os.path.join(FLAGS.output_dir, "predictions_titles_"+str(evi)+".json")
    else:
      output_prediction_file = os.path.join(FLAGS.output_dir, "answer_predictions.json")
      output_nbest_file = os.path.join(FLAGS.output_dir, "answer_nbest_predictions.json")
      output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")
      output_title_file = os.path.join(FLAGS.output_dir, "predictions_titles.json")
  
    write_reader_predictions(eval_examples, eval_features, all_results_reader,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, output_title_file,
                      True, False)
    del all_results_reader
        
    if FLAGS.split_count != 1:
      output_prediction_file = os.path.join(FLAGS.output_dir, "query_predictions_"+str(evi)+".json")
      output_prediction_best_file = os.path.join(FLAGS.output_dir, "query_predictions_best_"+str(evi)+".json")
    else:
      output_prediction_file = os.path.join(FLAGS.output_dir, "query_predictions.json")
      output_prediction_best_file = os.path.join(FLAGS.output_dir, "query_predictions_best.json")
    
    write_qg_predictions(tokenizer,
                         eval_features,
                         all_results_qg,
                         output_prediction_file,
                         output_prediction_best_file)                              
    del all_results_qg
    
  if FLAGS.split_count != 1:
    output_prediction_files = []
    output_prediction_best_files = []
    output_answer_prediction_files = []
    output_nbest_files = []
    output_null_log_odds_files = []
    output_title_files = []
    
    for evi in range(len(eval_example_list)):  
        output_prediction_files.append(os.path.join(FLAGS.output_dir, "query_predictions_"+str(evi)+".json"))
        output_prediction_best_files.append(os.path.join(FLAGS.output_dir, "query_predictions_best_"+str(evi)+".json"))      
        output_answer_prediction_files.append(os.path.join(FLAGS.output_dir, "answer_predictions_"+str(evi)+".json"))
        output_nbest_files.append(os.path.join(FLAGS.output_dir, "answer_nbest_predictions_"+str(evi)+".json"))
        output_null_log_odds_files.append(os.path.join(FLAGS.output_dir, "null_odds_"+str(evi)+".json"))
        output_title_files.append(os.path.join(FLAGS.output_dir, "predictions_titles_"+str(evi)+".json"))
        
    output_prediction_file = os.path.join(FLAGS.output_dir, "query_predictions.json")
    output_prediction_best_file = os.path.join(FLAGS.output_dir, "query_predictions_best.json")
    output_answer_prediction_file = os.path.join(FLAGS.output_dir, "answer_predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "answer_nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")
    output_title_file = os.path.join(FLAGS.output_dir, "predictions_titles.json")
        
    merge_json(output_prediction_files, output_prediction_file)
    merge_json(output_prediction_best_files, output_prediction_best_file)
    merge_json(output_answer_prediction_files, output_answer_prediction_file)
    merge_json(output_nbest_files, output_nbest_file)
    merge_json(output_null_log_odds_files, output_null_log_odds_file)
    merge_json(output_title_files, output_title_file)
        
def main(_):
  import os
  import tensorflow as tf
  if FLAGS.verbose_logging:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)

  config = tf.ConfigProto()
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  if FLAGS.horovod:
    import horovod.tensorflow as hvd
    hvd.init()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
  else:
    hvd = None

  if FLAGS.horovod is False or hvd.rank() == 0:
    tf.logging.info("Making Output_dir = %s", FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    tf.compat.v1.logging.info("***** Configuaration *****")
    for key in FLAGS.__flags.keys():
        tf.compat.v1.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
    tf.compat.v1.logging.info("**************************")

  saving_steps = FLAGS.save_checkpoints_steps if not FLAGS.horovod or hvd.rank() == 0 else None
  run_config = tf.estimator.RunConfig(
      model_dir=FLAGS.output_dir,
      session_config=config,
      save_checkpoints_steps=saving_steps,
      keep_checkpoint_max=1)

  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  if FLAGS.do_train:
    train(bert_config, hvd, run_config, tokenizer)

  if FLAGS.do_predict and (not FLAGS.horovod or hvd.rank() == 0):
    if FLAGS.qg_reader_predict_file is not None:
      predict_query_gen_and_reader(bert_config, run_config, tokenizer)
      
if __name__ == "__main__":
  tf.compat.v1.app.run()
