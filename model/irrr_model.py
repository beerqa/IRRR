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
from tokenization import _normalize

FLAGS = irrr_flags.FLAGS

"""
  Global varible declaration
"""

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True, *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
       float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, 
                      shape, 
                      dtype=storage_dtype,
                      initializer=initializer, 
                      regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable
    
compute_type = tf.float16 if FLAGS.use_fp16 else tf.float32
custom_getter = float32_variable_storage_getter if FLAGS.use_fp16 else None


def input_fn_builder(input_file, 
                     seq_length, 
                     max_query_length, 
                     is_training, 
                     drop_remainder, 
                     num_steps,
                     num_cpu_threads=4):

  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  
  name_to_features = {
      "unique_ids"  : tf.FixedLenFeature([FLAGS.ranking_candidates           ], tf.int64),
      "input_ids"   : tf.FixedLenFeature([FLAGS.ranking_candidates*seq_length], tf.int64),
      "input_mask"  : tf.FixedLenFeature([FLAGS.ranking_candidates*seq_length], tf.int64),
      "segment_ids" : tf.FixedLenFeature([FLAGS.ranking_candidates*seq_length], tf.int64),
      "p_query_ids" : tf.FixedLenFeature([FLAGS.ranking_candidates*seq_length], tf.int64),
      "query_loss"  : tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64),
      "answer_loss" : tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64),
      "rank_loss"   : tf.FixedLenFeature([], tf.int64),
  }
  if is_training:
    name_to_features["query_ids"]     = tf.FixedLenFeature([FLAGS.ranking_candidates*seq_length], tf.int64)
    name_to_features["query_ids2"]    = tf.FixedLenFeature([FLAGS.ranking_candidates*seq_length], tf.int64)
    name_to_features["gt_index"]    = tf.FixedLenFeature([FLAGS.ranking_candidates], tf.float32)
    name_to_features["start_pos"]     = tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64)
    name_to_features["end_pos"]       = tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64)
    name_to_features["start_pos2"]    = tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64)
    name_to_features["end_pos2"]      = tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64)
    name_to_features["reader_switch"] = tf.FixedLenFeature([FLAGS.ranking_candidates], tf.int64)
  
  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example

  if is_training:
    batch_size = FLAGS.train_batch_size
  else:
    batch_size = FLAGS.predict_batch_size

  def input_fn(params):
    """The actual input function."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.TFRecordDataset(input_file, num_parallel_reads=4)
        d = d.apply(tf.data.experimental.ignore_errors())
        d = d.shuffle(buffer_size=100)
        d = d.repeat()
    else:
        d = tf.data.TFRecordDataset(input_file)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d
  return input_fn
  
def get_query_loss(bert_config, input_tensor, labels, query_loss_weight, mask):
  """Get loss and log probs for the query prediction."""
  with tf.variable_scope("cls/query"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if labels[0] is not None:
      mask = tf.cast(mask, dtype=tf.float32)
      def _per_example_loss(labels, log_probs):
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        per_example_loss = per_example_loss*mask
        return per_example_loss
      per_example_loss1 = _per_example_loss(labels[0], log_probs)
      per_example_loss2 = _per_example_loss(labels[1], log_probs)

      per_example_loss = tf.reduce_min(tf.concat([tf.expand_dims(per_example_loss1, axis=-1),
                                                  tf.expand_dims(per_example_loss2, axis=-1)], 
                                                 axis=-1), axis=-1)
      query_loss_weight = tf.cast(query_loss_weight,dtype=tf.float32)
      loss = tf.reduce_sum(tf.reduce_sum(per_example_loss, axis=-1)*
                           query_loss_weight) / (tf.reduce_sum(mask* tf.expand_dims(query_loss_weight, axis=-1))+1e-10)

      return loss, log_probs
    else:
      return None, log_probs
      
def get_ranking_loss(config, pooled_output, rank_loss_weight, gt_index):
  with tf.variable_scope("cls/ranker"):
    if FLAGS.rank_network:
      with tf.variable_scope("ranker"):
        pooled_output = modeling.transformer_model(pooled_output,
                                                   hidden_size=config.hidden_size,
                                                   num_hidden_layers=1,
                                                   num_attention_heads=config.num_attention_heads,
                                                   intermediate_size=config.intermediate_size)
                                 
    ranking_logits = tf.layers.dense(
            pooled_output,
            1,
            activation=None,
            kernel_initializer=modeling.create_initializer(config.initializer_range))
     
    if gt_index != None:
      per_example_ranking_loss = tf.nn.softmax_cross_entropy_with_logits(
              labels=gt_index, 
              logits=tf.squeeze(ranking_logits, axis=-1))
              
      rank_loss_weight = tf.cast(rank_loss_weight,dtype=tf.float32)
      ranking_loss = tf.reduce_sum(per_example_ranking_loss*
                                   rank_loss_weight, axis=-1) / (tf.reduce_sum(rank_loss_weight)+1e-10)
     
      return ranking_loss, ranking_logits
    else:
      return None, ranking_logits

def get_span_loss(final_hidden, start_positions, end_positions, answer_loss_weight):

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]
  
  output_weights = tf.get_variable(
      "cls/reader/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/reader/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                  [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0, name='unstack')

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
  
  if start_positions != None or end_positions != None:
    def compute_loss(logits, positions, loss_weight):
      one_hot_positions = tf.one_hot(
          positions, depth=seq_length, dtype=tf.float32)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      
      loss_weight = tf.cast(loss_weight, dtype=tf.float32)
      loss = -tf.reduce_sum(
              tf.reduce_sum(one_hot_positions * log_probs * 
                            tf.expand_dims(loss_weight, axis=-1), axis=-1)) / (tf.reduce_sum(loss_weight)+1e-10)
      return loss
      
    start_loss = compute_loss(start_logits, start_positions[0], answer_loss_weight)
    end_loss = compute_loss(end_logits, end_positions[0], answer_loss_weight)
  
    start_loss2 = compute_loss(start_logits, start_positions[1], answer_loss_weight)
    end_loss2 = compute_loss(end_logits, end_positions[1], answer_loss_weight)
    return tf.reduce_min([(start_loss + end_loss),(start_loss2 + end_loss2)], axis=-1) / 2.0, start_logits, end_logits
  else:  
    return None, start_logits, end_logits
  
def get_reader_switch_loss(bert_config, input_tensor, labels, loss_weight):
  """Get loss and log probs for the reader switch prediction."""

  with tf.variable_scope("cls/reader_switch"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[4, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[4], initializer=tf.zeros_initializer())

    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if labels is not None:
      labels = tf.reshape(labels, [-1])
      one_hot_labels = tf.one_hot(labels, depth=4, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss_weight = tf.cast(loss_weight,dtype=tf.float32)
      loss = tf.reduce_sum(per_example_loss*loss_weight) / (tf.reduce_sum(loss_weight)+1e-10)
    else:
      loss = None
    return (loss, logits)
    
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings, hvd=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    if is_training:
      batch_size = FLAGS.train_batch_size
    else:
      batch_size = FLAGS.predict_batch_size
      
    if hvd == None or hvd.rank() == 0:
      tf.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    
    example_count = batch_size*FLAGS.ranking_candidates
    
    rank_loss_weight = features["rank_loss"]
    query_loss_weight = tf.reshape(features["query_loss"], [example_count])
    answer_loss_weight = tf.reshape(features["answer_loss"], [example_count])
    
    unique_ids  = tf.reshape(features["unique_ids"],  [example_count])
    input_ids   = tf.reshape(features["input_ids"],   [example_count, FLAGS.max_seq_length]) 
    input_mask  = tf.reshape(features["input_mask"],  [example_count, FLAGS.max_seq_length])
    segment_ids = tf.reshape(features["segment_ids"], [example_count, FLAGS.max_seq_length])
    p_query_ids = tf.reshape(features["p_query_ids"], [example_count, FLAGS.max_seq_length]) 
    

    if is_training:
      query_ids = [tf.reshape(features["query_ids"], [example_count, FLAGS.max_seq_length]),
                   tf.reshape(features["query_ids2"], [example_count, FLAGS.max_seq_length])]
      start_positions = [tf.reshape(features["start_pos"], [example_count]),
                         tf.reshape(features["start_pos2"], [example_count])]
      end_positions = [tf.reshape(features["end_pos"], [example_count]),
                       tf.reshape(features["end_pos2"], [example_count])]
      reader_switch = tf.reshape(features["reader_switch"], [example_count])
      gt_index = features["gt_index"]
      
    else:
      query_ids = [None, None]
      start_positions = None
      end_positions = None
      gt_index = None
      reader_switch = None
        
    # BERT encode 
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               prev_query_ids=p_query_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings,
                               compute_type=compute_type,
                               scope=FLAGS.model_scope)
        
    # Query loss 
    query_loss, query_log_probs = get_query_loss(bert_config, 
                                model.get_sequence_output(), 
                                query_ids, 
                                query_loss_weight,
                                input_mask) 
         
    # Ranking Loss
    pooled_output = tf.reshape(model.get_pooled_output(), 
                               [batch_size, FLAGS.ranking_candidates, -1])
    
    ranking_loss, ranking_logits = get_ranking_loss(bert_config, 
                                                    pooled_output, 
                                                    rank_loss_weight,
                                                    gt_index)
     
    # Span loss
    span_loss, start_logits, end_logits = get_span_loss(model.get_sequence_output(), 
                                                        start_positions, end_positions,
                                                        answer_loss_weight)
    
    # Reader switch loss
    reader_switch_loss, reader_switch_log_probs = get_reader_switch_loss(
                                                        bert_config, 
                                                        model.get_pooled_output(), 
                                                        reader_switch,
                                                        answer_loss_weight)
         
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
           tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if hvd is None or hvd.rank() == 0:
      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      total_loss = query_loss + ranking_loss + span_loss + reader_switch_loss
      
      train_op = optimization.create_optimizer(
          total_loss, 
          learning_rate, 
          num_train_steps, 
          num_warmup_steps,
          hvd, 
          True, 
          False, 
          FLAGS.num_accumulation, 
          "adam")
          
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "query_log_probs": query_log_probs,
          "ranking": tf.reshape(ranking_logits, [-1]),
          "reader_switch": reader_switch_log_probs,
          "start_logits": start_logits,
          "end_logits": end_logits
      }
      output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn

