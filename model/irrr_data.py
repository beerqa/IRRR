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

from enum import IntEnum
class ReaderSwitch(IntEnum):
  HAS_ANSWER=0
  NO_ANSWER=1
  YES=2
  NO=3

class QAExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """
  def __init__(self, qas_id, question_text,
               doc_tokens, doc_text,
               is_last_non_gt=False, is_alternative=False, reader_switch=False,
               query=None, prev_query=None, last_doc=None,
               orig_answer_text=None, 
               answer_span=None, 
               all_titles=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.doc_text = doc_text
    self.is_last_non_gt = is_last_non_gt
    self.is_alternative = is_alternative
    self.reader_switch = reader_switch
    self.query = query
    self.prev_query = prev_query
    self.last_doc = last_doc
    self.orig_answer_text = orig_answer_text
    self.answer_span = answer_span
    self.all_titles = all_titles

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.is_last_non_gt:
      s += ", is_last_non_gt: %r" % (self.is_last_non_gt)
    if self.reader_switch:
      s += ", reader_switch: %r" % (self.reader_switch)
    return s

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               qas_id, unique_id,
               example_index, doc_span_index,
               tokens, token_to_orig_map, token_is_max_context,
               input_ids, input_mask, segment_ids,
               is_last_non_gt=None, is_alternative=None, reader_switch=None,
               gen_query=None, gen_query_ids=None, gen_query_ids2=None,
               gen_query_mask=None, prev_query_ids=None,
               original_text=None, last_doc=None,
               start_position=0, end_position=0,
               start_position2=0, end_position2=0):
    self.qas_id = qas_id
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.is_last_non_gt = is_last_non_gt
    self.is_alternative = is_alternative
    self.reader_switch = reader_switch
    self.gen_query = gen_query
    self.gen_query_ids = gen_query_ids
    self.gen_query_ids2 = gen_query_ids2
    self.gen_query_mask = gen_query_mask
    self.prev_query_ids = prev_query_ids
    self.original_text = original_text
    self.last_doc = last_doc
    self.start_position = start_position
    self.end_position = end_position
    self.start_position2 = start_position2
    self.end_position2 = end_position2

class InputFeaturesGroup(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """
  def __init__(self,
               qas_id,
               reader=False,
               qa_reader=False):
    self.qas_id = qas_id
    self.examples = []
    self.gt_index = []
    self.reader = reader
    self.qa_reader = qa_reader
    
    self.rank_loss = 0    
    if reader:
      self.query_loss = [0 for i in range(FLAGS.ranking_candidates)]
      self.answer_loss = [1 for i in range(FLAGS.ranking_candidates)]
      self.gt_index = [1.0 for i in range(FLAGS.ranking_candidates)]
    elif qa_reader:
      self.query_loss = [1 for i in range(FLAGS.ranking_candidates)]
      self.answer_loss = [1 for i in range(FLAGS.ranking_candidates)]    
    else:
      self.query_loss = [1 for i in range(FLAGS.ranking_candidates)]
      self.answer_loss = [0 for i in range(FLAGS.ranking_candidates)]

  def append(self, example):
    self.examples.append(example)
    if not self.reader:
      self.gt_index.append(float(1-example.is_last_non_gt))
      if example.is_alternative and FLAGS.soft_target: 
        self.gt_index[-1] += 0.1
    ## Apply loss when there is impossible case 
    if example.is_last_non_gt and not self.reader:
      self.rank_loss = 1
    if np.sum(example.gen_query_ids) == 0 and FLAGS.no_empty_query_loss:
      self.query_loss[len(self.examples)-1] = 0
  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    if self.gt_index:
      s += ", gt_index: %s" % (str(self.gt_index))
    return s


def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index
  
  
class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)
    self.ign = 0
    
  def process_feature(self, in_features):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1
    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(np.array(values).reshape(-1))))
      return feature

    def create_float_feature(values):
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return feature

    if len(in_features.examples) != FLAGS.ranking_candidates:
      while len(in_features.examples) != FLAGS.ranking_candidates:
        in_features.examples.append(in_features.examples[-1])
        if len(in_features.gt_index) != FLAGS.ranking_candidates:
          in_features.gt_index.append(in_features.gt_index[-1])
          
    if len(in_features.examples) != len(in_features.gt_index):
      print("### num example != gt_index")
      for exm in in_features.examples:
        print(exm.qas_id)
      print(in_features.gt_index)
      print(in_features.query_loss)
      print(in_features.answer_loss)
      print(in_features.rank_loss)
      return

    if self.is_training:
      gt_sum = np.sum(in_features.gt_index)   
      if gt_sum != 0: 
        in_features.gt_index = [ inf / gt_sum for inf in in_features.gt_index ]

      else:
        in_features.gt_index = [1.0/float(FLAGS.ranking_candidates) for i in range(FLAGS.ranking_candidates)]
        in_features.rank_loss = 0
    
    features = collections.OrderedDict()
    features["unique_ids" ] = create_int_feature([feature.unique_id for feature in in_features.examples])
    features["input_ids"  ] = create_int_feature([feature.input_ids for feature in in_features.examples])
    features["input_mask" ] = create_int_feature([feature.input_mask for feature in in_features.examples])
    features["segment_ids"] = create_int_feature([feature.segment_ids for feature in in_features.examples])
    features["query_ids"  ] = create_int_feature([feature.gen_query_ids for feature in in_features.examples])
    features["query_ids2" ] = create_int_feature([feature.gen_query_ids2 for feature in in_features.examples])
    features["p_query_ids"] = create_int_feature([feature.prev_query_ids for feature in in_features.examples])
    features["rank_loss"  ] = create_int_feature([in_features.rank_loss])
    features["query_loss"  ] = create_int_feature(in_features.query_loss)
    features["answer_loss"  ] = create_int_feature(in_features.answer_loss)
    if self.is_training:
      features["start_pos"  ] = create_int_feature([feature.start_position for feature in in_features.examples])
      features["end_pos"    ] = create_int_feature([feature.end_position for feature in in_features.examples])
      features["start_pos2"  ] = create_int_feature([feature.start_position2 for feature in in_features.examples])
      features["end_pos2"    ] = create_int_feature([feature.end_position2 for feature in in_features.examples])
      features["reader_switch"] = create_int_feature([int(feature.reader_switch) for feature in in_features.examples])      
      features["gt_index"   ] = create_float_feature(in_features.gt_index)
      
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()
    
QGRawResult = collections.namedtuple("QGRawResult", 
                                    ["unique_id", 
                                     "query_log_probs", 
                                     "na_log_probs",                                    
                                     "reader_log_probs"])

ReaderRawResult = collections.namedtuple("ReaderRawResult", 
                                        ["unique_id", 
                                         "ans_log_probs",
                                         "na_log_probs",
                                         "yes_log_probs",
                                         "no_log_probs", 
                                         "start_logits", 
                                         "end_logits"])
