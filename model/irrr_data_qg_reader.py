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

def read_qg_reader_examples(input_file, is_training, version_2_with_negative=True, input_data=None):
  """Return list of SquadExample from input_data or input_file (SQuAD json file)"""
  input_data = []
  input_files = input_file.split(",")
  for ifile in input_files:
    with tf.gfile.Open(ifile, "r") as reader:
      input_data.extend(json.load(reader)["data"])
      if FLAGS.debug and len(input_data) >= 1000:
        break

  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = []
  for entry in input_data if FLAGS.verbose_logging else tqdm(input_data):
    for paragraph in entry["paragraphs"]:
      if FLAGS.debug and len(examples) >= 1000:
        break
      question = paragraph["qas"][0]['question']
      if paragraph["context"].find(question) != -1:
        paragraph["context"] = paragraph["context"][len(question)+1:]

      paragraph_text = _normalize(paragraph["context"]).replace("<t>","[SEP]").replace("</t>","[et]")
      if paragraph_text.startswith("[SEP]") == False and len(paragraph_text) != 0:
        paragraph_text = "[SEP] " + paragraph_text
      question_text = paragraph['qas'][0]["question"]
      question_only = len(paragraph_text) == 0
      paragraph_text_with_question = question_text + " " + paragraph_text # append question
      if question_only:
        paragraph_text = paragraph_text_with_question

      para_split = paragraph_text.split("[SEP]")
      if 'para_titles' in paragraph:
        all_title= paragraph['para_titles']
      else:
        all_title = []
        for para in para_split[1:]:
          title = para.split('[et]')[0].strip()
          if len(title) != 0:
            all_title.append(title)
      last_title = all_title[-1] if len(all_title) > 0 else ""

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


      if is_training:
        query_key = 'query_short' if FLAGS.short and 'query_short' in paragraph else 'query'
        query = paragraph[query_key]
      else:
        query = ""
      if isinstance(query, str):
        query = [query]
      query = [_normalize(q).strip() for q in query]
      if len(query) == 1:
        query.append(query[0])

      if FLAGS.prev_query:
        prev_query = paragraph['prev_query'] if 'prev_query' in paragraph else ""
        if 'hop1_query' in paragraph:
          prev_query = paragraph['hop1_query']
      else:
        prev_query = ""

      if query[0] == "":

        new_query = paragraph['prev_query_short'] if 'prev_query_short' in paragraph else ""
        if 'hop1_query' in paragraph:
          new_query = paragraph['hop1_query']
        query = [_normalize(new_query).strip(),
                 _normalize(new_query).strip()]

      def fix_query(context, queries):
        if len(queries) == 0:
          return ""
        new_queries = []
        without_space = ""
        char_index = []
        context_ = "_" + context.replace(" ", "_") +"_"
        for ci, c in enumerate(context):
          if is_whitespace(c) != True:
            without_space += c
            char_index.append(ci)

        queries = queries.split(" ")
        for q in queries:
          if "_" + q + "_" in context_:
            new_queries.append(q)
          else:
            qns = q.replace("_","")
            cid = without_space.find(qns)
            if cid != -1:
              new_queries.append(context[char_index[cid]:char_index[cid+len(qns)-1]+1].strip().replace(" ","_"))

        return " ".join(new_queries)


      is_last_non_gt = False

      is_last_non_gt = paragraph["is_last_non_gt"] if 'is_last_non_gt' in paragraph else False
      is_alternative = paragraph['is_alternative'] if 'is_alternative' in paragraph else False
      is_continue = paragraph['is_continue'] if 'is_continue' in paragraph else False

      new_query = [fix_query(paragraph_text_with_question, q) for q in query]
      query = new_query

      prev_query = fix_query(paragraph_text_with_question, prev_query)

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        
        question_text = qa["question"]
        is_last_non_gt = qa["is_last_non_gt"] if 'is_last_non_gt' in qa else is_last_non_gt
        is_alternative = qa['is_alternative'] if 'is_alternative' in qa else is_alternative
        is_continue = qa['is_impossible'] if 'is_impossible' in qa else is_continue

        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        reader_switch = ReaderSwitch.HAS_ANSWER

        spans = []
        if is_training:
          if version_2_with_negative:
            is_impossible = qa["is_impossible"]
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
                answer_offset = answer["answer_start"]# + len(question_text) + 1
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
            question_text="" if question_only else question_text,
            doc_tokens=doc_tokens,
            doc_text=paragraph_text_with_question,
            orig_answer_text=orig_answer_text,
            answer_span=spans,
            is_last_non_gt=is_last_non_gt,
            is_alternative=is_alternative,
            reader_switch=reader_switch,
            all_titles=all_title,
            query= query,
            prev_query=prev_query,
            last_doc = last_title)
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

def convert_qg_reader_examples_to_features(examples, tokenizer, max_seq_length,
                                           doc_stride, max_query_length, is_training,
                                           output_fn, verbose_logging=False):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  input_features = {}

  last_group = InputFeaturesGroup("", reader=False, qa_reader=True)

  for (example_index, example) in enumerate(examples if FLAGS.verbose_logging else tqdm(examples)):
    def _get_query(query):
      gen_query = []
      gen_query_ids = []
      if query:
        query_split = query.split(' ')
        for qs in query_split:
          gen_query.append(tokenizer.tokenize(qs.replace("_", " ")))
          gen_query_ids.append(tokenizer.convert_tokens_to_ids(gen_query[-1]))
      return gen_query, gen_query_ids

    gen_query,  gen_query_ids  = _get_query(example.query[0])
    gen_query2, gen_query_ids2 = _get_query(example.query[1])
    prev_query, prev_query_ids = _get_query(example.prev_query)

    query_tokens = tokenizer.tokenize(example.question_text)
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
      seg_id = 1 if example.question_text != "" else 0
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        if token[-1] == '[SEP]':
          seg_id = 1
        segment_ids.append(seg_id)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      def _make_query_label(input_ids, gen_query, gen_query_ids):
        query_label = [0 for iid in input_ids]
        iix = 0
        num_pop = 0
        for qgix, gqi in enumerate(gen_query_ids):
          not_found = True
          start_iix = iix
          for fi in range(iix, len(input_ids)-len(gqi)):
            if gqi == input_ids[fi:fi+len(gqi)]:
              num_pop += 1
              not_found = False
              for mi in range(len(gqi)):
                query_label[fi+mi] = 1
              break
            iix+= 1
          if not_found:
            print ("NOT_FOUND: " + " ".join(gen_query[qgix]))
            iix=0
        gen_query_ids = gen_query_ids[num_pop:]
        gen_query = gen_query[num_pop:]
        return query_label, gen_query, gen_query_ids

      query_label,  gen_query,  gen_query_ids  = _make_query_label(input_ids, gen_query,  gen_query_ids)
      query_label2, gen_query2, gen_query_ids2 = _make_query_label(input_ids, gen_query2, gen_query_ids2)
      prev_query_label, prev_query, prev_query_ids = _make_query_label(input_ids, prev_query, prev_query_ids)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        query_label.append(0)
        query_label2.append(0)
        prev_query_label.append(0)

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
        tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
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
          tf.compat.v1.logging.info("last_non_gt example")
        if example.is_alternative:
          tf.compat.v1.logging.info("alternative example")
        tf.compat.v1.logging.info(
            "query_label: %s" % " ".join([str(x) for x in query_label]))
        tf.compat.v1.logging.info(
            "prev_query_label: %s" % " ".join([str(x) for x in prev_query_label]))

      input_feature = InputFeatures(
          unique_id=unique_id,
          qas_id=example.qas_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          gen_query=gen_query,
          gen_query_mask=input_mask,
          gen_query_ids=query_label,
          gen_query_ids2=query_label2,
          prev_query_ids=prev_query_label,
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
          reader_switch=example.reader_switch,
          is_alternative=example.is_alternative,
          original_text=example.doc_text,
          last_doc=example.last_doc)

      if not is_training:
        if len(example.qas_id.split('_')) == 1:
          if len(last_group.examples) == FLAGS.ranking_candidates:
            output_fn(last_group)
            last_group = InputFeaturesGroup("", reader=False, qa_reader=True)
          last_group.append(input_feature)
        else:
          ori_id = example.qas_id[:len(example.qas_id)-2]
          if ori_id not in input_features:
            input_features[ori_id] = InputFeaturesGroup(ori_id, reader=False, qa_reader=True)
          if len(input_features[ori_id].examples) == FLAGS.ranking_candidates:
            output_fn(input_features[ori_id])
            input_features[ori_id] = InputFeaturesGroup(ori_id, reader=False, qa_reader=True)

          input_features[ori_id].append(input_feature)
      elif len(example.qas_id.split('_')) <= 2:
        if len(last_group.examples) == FLAGS.ranking_candidates:
          output_fn(last_group)
          last_group = InputFeaturesGroup("", reader=False, qa_reader=True)
        last_group.append(input_feature)
      else:
        ori_id = example.qas_id[:len(example.qas_id)-2]
        if ori_id not in input_features:
          input_features[ori_id] = InputFeaturesGroup(ori_id, reader=False, qa_reader=True)
        if len(input_features[ori_id].examples) == FLAGS.ranking_candidates:
          output_fn(input_features[ori_id])
          input_features[ori_id] = InputFeaturesGroup(ori_id, reader=False, qa_reader=True)

        input_features[ori_id].append(input_feature)

      unique_id += 1

  if len(last_group.examples) != 0:
    output_fn(last_group)

  for ori_id in input_features.keys():
    output_fn(input_features[ori_id])


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
