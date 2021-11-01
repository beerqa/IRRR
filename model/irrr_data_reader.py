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
from irrr_data import *
from tokenization import _normalize

FLAGS = irrr_flags.FLAGS

def read_squad_examples(input_file, is_training, version_2_with_negative=True, input_data=None):
  """Return list of SquadExample from input_data or input_file (SQuAD json file)"""
  input_data = []
  input_files = input_file.split(",")
  for ifile in input_files:
    with tf.gfile.Open(ifile, "r") as reader:
      input_data.extend(json.load(reader)["data"])
    
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data if FLAGS.verbose_logging else tqdm(input_data):
    if FLAGS.debug and len(examples) >= 100:
      break
    title = entry['title']
    if '_' in title:
      title = title.split('_')[0]
    for paragraph in entry["paragraphs"]:
      paragraph_text = "[SEP] " + title + " [et] " +paragraph["context"]
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

          if version_2_with_negative:
            is_impossible = qa["is_impossible"]
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")
          spans = []
          if not is_impossible:
            for qaans in qa["answers"][:2]:
              answer = qa["answers"][0]
              orig_answer_text = answer["text"]
              answer_offset = answer["answer_start"] + len("[SEP] " + title + " [et] ")
              answer_length = len(orig_answer_text)
              start_position = char_to_word_offset[answer_offset]
              end_position = char_to_word_offset[answer_offset + answer_length -
                                                 1]
              # Only add answers where the text can be exactly recovered from the
              # document. If this CAN'T happen it's likely due to weird Unicode
              # stuff so we will just skip the example.
              #
              # Note that this means for training mode, every example is NOT
              # guaranteed to be preserved.
              actual_text = " ".join(
                  doc_tokens[start_position:(end_position + 1)])
              cleaned_answer_text = " ".join(
                  tokenization.whitespace_tokenize(orig_answer_text))
              if actual_text.find(cleaned_answer_text) == -1:
                tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                continue
              spans.append([start_position, end_position])
          else:
            start_position = -1
            end_position = -1
            spans.append([-1, -1])
            orig_answer_text = ""
        example = QAExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            doc_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            answer_span=spans,
            is_last_non_gt=is_impossible,
            reader_switch=1 if is_impossible else 0,
            all_titles=[title])
        examples.append(example)

  return examples
  
def read_reader_examples(input_file, is_training, version_2_with_negative=True, input_data=None):
  """Return list of SquadExample from input_data or input_file (SQuAD json file)"""
  input_data = []
  input_files = input_file.split(",")
  for ifile in input_files:
    with tf.gfile.Open(ifile, "r") as reader:
      input_data.extend(json.load(reader)["data"])

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data if FLAGS.verbose_logging else tqdm(input_data):
    for paragraph in entry["paragraphs"]:
      if FLAGS.debug and len(examples) >= 1000:
        break
      if paragraph["context"].startswith("[SEP]") == False and paragraph["context"].startswith("<t>") == False:
        paragraph["context"] = "[SEP] " + paragraph["context"]
      paragraph_text = paragraph["context"].replace("<t>","[SEP]").replace("</t>","[et]")
      para_split = paragraph_text.split("[SEP]")
      if 'para_titles' in paragraph:
        all_title= paragraph['para_titles']
      else:
        all_title = []
        for para in para_split[1:]:
          title = para.split('[et]')[0].strip()
          if len(title) != 0:
            all_title.append(title)
      #print(paragraph_text)
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True
      for c in paragraph_text:
        if is_whitespace(c):
          prev_is_whitespace = True
        else:
          if prev_is_whitespace:
            doc_tokens.append(c)
          else:
            doc_tokens[-1] += c
          prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        #if not is_training and (qas_id != "5a906a685542990a9849362e"):
        #  continue
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        reader_switch = ReaderSwitch.HAS_ANSWER
        
        spans = []
        if is_training:
          if version_2_with_negative:
            is_impossible = qa["is_impossible"]
            if FLAGS.is_grr and is_impossible and len(qa["answers"]) != 0 and qa["answers"][0]['text'] in ['yes', 'no']:
              is_impossible = False
            if is_impossible:
              reader_switch = ReaderSwitch.NO_ANSWER

          if not is_impossible or (FLAGS.use_distant and 
                                   'answers' in qa and 
                                   len(qa["answers"])>0 
                                   and qa["answers"][0]['answer_start'] !=-1):
            for answer in qa["answers"][:2]:
              orig_answer_text = answer["text"].replace("\xa0", " ").replace("\u2009", " ")
              #print(orig_answer_text)
              if orig_answer_text in ['[YES]', '[NO]', 'yes', 'no']:
                answer_offset = 0
                answer_length = 0
                start_position = 0
                end_position = 0
                spans.append([0,0])
                if is_impossible == False:
                  reader_switch = ReaderSwitch.YES if orig_answer_text in ['[YES]', 'yes'] else ReaderSwitch.NO
              else:
                answer_offset = answer["answer_start"]
                if answer_offset == -1 or paragraph_text[answer_offset:answer_offset+len(orig_answer_text)] != orig_answer_text:
                  answer_offset = paragraph_text.find(orig_answer_text)
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length -
                                                 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                  tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                     actual_text, cleaned_answer_text)
                  continue
                spans.append([start_position, end_position])
          else:
            start_position = -1
            end_position = -1
            spans.append([-1, -1])
            orig_answer_text = ""

        example = QAExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            doc_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            answer_span=spans,
            is_last_non_gt=is_impossible,
            reader_switch=reader_switch,
            all_titles=all_title)
        examples.append(example)

  return examples
    
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)
  
def convert_reader_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, verbose_logging=False):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  last_group = InputFeaturesGroup("", reader=True)

  for (example_index, example) in enumerate(examples if FLAGS.verbose_logging else tqdm(examples)):
    query_tokens = tokenizer.tokenize(example.question_text.replace("[YES] [NO] ", ""))

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    #print(str(example.doc_tokens))
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      if token in ['[SEP]', '[t]', '[et]', '[YES]', '[NO]']:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(token)
      else:
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
          tok_to_orig_index.append(i)
          all_doc_tokens.append(sub_token)

    tok_spans = []
    for span in example.answer_span:
      tok_start_position = None
      tok_end_position = None
      if is_training and span[0] == -1:
        tok_start_position = -1
        tok_end_position = -1
      if is_training and span[0] != -1:
        tok_start_position = orig_to_tok_index[span[0]]
        if span[1] < len(example.doc_tokens) - 1:
          tok_end_position = orig_to_tok_index[span[1] + 1] - 1
        else:
          tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)
      tok_spans.append([tok_start_position, tok_end_position])

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      span_positions = []
      for si, span in enumerate(example.answer_span):
        start_position = None
        end_position = None
        if is_training and not span[0] == -1:      
          if is_training and example.orig_answer_text in ['[YES]', '[NO]', 'yes', 'no']:
            start_position = 0
            end_position = 0
          else:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_spans[si][0] >= doc_start and
                    tok_spans[si][1] <= doc_end):
              out_of_span = True
            if out_of_span:
              start_position = 0
              end_position = 0
            else:
              doc_offset = len(query_tokens) + 1
              start_position = tok_spans[si][0] - doc_start + doc_offset
              end_position = tok_spans[si][1] - doc_start + doc_offset
  
        if is_training and span[0] == -1:
          start_position = 0
          end_position = 0
          
        span_positions.append([start_position, end_position])
        
      if len(span_positions) == 0:
        span_positions.append([0,0])
      if len(span_positions) == 1:
        span_positions.append(span_positions[-1])
        
      if example_index < 20:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("unique_id: %s" % (unique_id))
        tf.compat.v1.logging.info("qas_id: %s" % (example.qas_id))
        tf.compat.v1.logging.info("example_index: %s" % (example_index))
        tf.compat.v1.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        if is_training:
          tf.compat.v1.logging.info("reader_switch: %s" % (str(example.reader_switch)))
        if is_training and start_position != 0:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.compat.v1.logging.info("start_position: %d" % (span_positions[0][0]))
          tf.compat.v1.logging.info("end_position: %d" % (span_positions[0][1]))
          tf.compat.v1.logging.info("start_position2: %d" % (span_positions[1][0]))
          tf.compat.v1.logging.info("end_position2: %d" % (span_positions[1][1]))
          tf.compat.v1.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))
          tf.compat.v1.logging.info(
              "ori_answer: %s" % example.orig_answer_text)

      feature = InputFeatures(
          unique_id=unique_id,
          qas_id=example.qas_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          gen_query_ids=[0]*max_seq_length,
          gen_query_ids2=[0]*max_seq_length,
          prev_query_ids=[0]*max_seq_length,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=span_positions[0][0] if is_training else 0,
          end_position=span_positions[0][1] if is_training else 0,
          start_position2=span_positions[1][0] if is_training else 0,
          end_position2=span_positions[1][1] if is_training else 0,
          is_last_non_gt=example.is_last_non_gt,
          reader_switch=example.reader_switch)
             
      if len(last_group.examples) == FLAGS.ranking_candidates:
        output_fn(last_group)
        last_group = InputFeaturesGroup("", reader=True)
      last_group.append(feature)
      unique_id += 1
      
  if len(last_group.examples) != 0:
    output_fn(last_group)
      

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  
  tok_text = " ".join(tokenizer.tokenize(orig_text))

  orig_text = orig_text.replace("[SEP]", "").replace("[t]", "").replace("[et]", "").replace("  ", " ").strip()
  tok_text = tok_text.replace("[SEP]", "").replace("[ t ]", "").replace("[ et ]", "").replace("  ", " ").strip()
  pred_text = pred_text.replace("[SEP]", "").replace("[t]", "").replace("[et]", "").replace("  ", " ").strip()
  
  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if verbose_logging:
      tf.compat.v1.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, tok_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if verbose_logging:
      tf.compat.v1.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if verbose_logging:
      tf.compat.v1.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if verbose_logging:
      tf.compat.v1.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text

def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes
  
def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs

def get_reader_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, 
  do_lower_case, version_2_with_negative, verbose_logging):
  """Get final predictions"""
  
  #all_results = fix_reasoning_probs(all_features, all_results)
  
  allf = []
  for al in all_features:
    for f in al.examples:
      allf.append(f)
  all_features = allf
  
  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction", ["feature_index", 
                           "start_index", 
                           "end_index", 
                           "start_logit", 
                           "end_logit", 
                           "ans_log_prob", 
                           "yes_log_prob", 
                           "no_log_prob",
                           "na_log_prob" ])

  all_predictions = collections.OrderedDict()
  all_titles = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples if FLAGS.verbose_logging else tqdm(all_examples)):
    features = example_index_to_features[example_index]
    if len(features) ==0:
      continue
    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      
      if version_2_with_negative:
        feature_null_score = result.na_log_probs #/ result.ranking_probs# - max([result.ans_log_probs, result.yes_log_probs, result.no_log_probs])
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.na_log_probs
          null_end_logit = result.na_log_probs
          null_ans_logit = result.ans_log_probs
          null_yes_logit = result.yes_log_probs
          null_no_logit = result.no_log_probs
          null_na_logit = result.na_log_probs
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
            
          if result.yes_log_probs >  max([result.ans_log_probs]) or \
             result.no_log_probs >  max([result.ans_log_probs]):
            start_index = end_index = 0
            
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index]-result.start_logits[0],
                  end_logit=result.end_logits[end_index]-result.end_logits[0],
                  ans_log_prob=result.ans_log_probs,
                  yes_log_prob=result.yes_log_probs,
                  no_log_prob=result.no_log_probs,
                  na_log_prob=result.na_log_probs))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "ans", "na", "yes", "no"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      start_logit = pred.start_logit
      end_logit = pred.end_logit
      title_ids = []
      is_title = 0
      for t in feature.tokens:
        if t=='[SEP]':
          is_title=1
        title_ids.append(is_title)
        if t=='[et]':
          is_title=0
          
      if pred.start_index > 0:  # this is a non-null prediction
        if title_ids[pred.start_index] == 1 or title_ids[pred.end_index] == 1:
          continue 
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
        if final_text in seen_predictions:
          continue
        if '[et]' in orig_text or '[t]' in orig_text or '[SEP]' in orig_text:
          continue
        seen_predictions[final_text] = True
      else:
        if pred.yes_log_prob >  max([pred.ans_log_prob, pred.no_log_prob, pred.na_log_prob]):
          final_text = 'yes'
          start_logit = end_logit = pred.yes_log_prob 
        elif pred.no_log_prob >  max([pred.ans_log_prob, pred.yes_log_prob, pred.na_log_prob]):
          final_text = 'no'
          start_logit = end_logit = pred.no_log_prob 
        else:
          final_text = ""
        seen_predictions[final_text] = True

      if len(final_text) != 0: 
        nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=start_logit,
              end_logit=end_logit,
              ans=pred.ans_log_prob,
              yes=pred.yes_log_prob,
              no=pred.no_log_prob,
              na=pred.na_log_prob))

    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", 
                           start_logit=0.0, 
                           end_logit=0.0,
                           ans=0.0,
                           yes=0.0,
                           no=0.0,
                           na=100.0))

    assert len(nbest) >= 1
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry
    if best_non_null_entry is None:
      best_non_null_entry = nbest[0]

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["ans"] = entry.ans
      output["na"] = entry.na
      output["yes"] = entry.yes
      output["no"] = entry.no
      nbest_json.append(output)
      
    ori_id = example.qas_id.split('_')[0]
    assert len(nbest_json) >= 1
    if not version_2_with_negative:
      all_predictions[ori_id] = nbest_json[0]["text"]
    else:
      if best_non_null_entry.text == 'yes':
        score_diff = best_non_null_entry.na - best_non_null_entry.yes
      elif best_non_null_entry.text == 'no':
        score_diff = best_non_null_entry.na - best_non_null_entry.no
      else:
        w = 0.5
        score_diff =  (best_non_null_entry.na - best_non_null_entry.ans) - (best_non_null_entry.start_logit + best_non_null_entry.end_logit)/2.0 #
        #score_diff = score_diff/2.0
      if ori_id in all_predictions:
        if scores_diff_json[ori_id] < score_diff:
          continue
        
      
      # predict "" iff the null score - the score of best non-null > threshold
      scores_diff_json[ori_id] = score_diff 
      if score_diff > FLAGS.null_score_diff_threshold:
        all_predictions[ori_id] = ""
      else:
        all_predictions[ori_id] = best_non_null_entry.text
      all_titles[ori_id] = example.all_titles

    all_nbest_json[ori_id] = nbest_json
  return all_predictions, all_nbest_json, scores_diff_json, all_titles
  

def write_reader_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, output_title_file,
                      version_2_with_negative, verbose_logging):
  """Write final predictions to the json file and log-odds of null if needed."""

  tf.compat.v1.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.compat.v1.logging.info("Writing nbest to: %s" % (output_nbest_file))

  all_predictions, all_nbest_json, scores_diff_json, all_titles, = get_reader_predictions(all_examples, all_features, 
    all_results, n_best_size, max_answer_length, do_lower_case, version_2_with_negative, verbose_logging)

  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if version_2_with_negative:
    with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
      
  with tf.io.gfile.GFile(output_title_file, "w") as writer:
    writer.write(json.dumps(all_titles, indent=4) + "\n")
      

