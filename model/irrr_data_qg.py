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
import unicodedata
import irrr_flags
from tqdm import tqdm
from irrr_data import *
from tokenization import _normalize

FLAGS = irrr_flags.FLAGS

def read_query_gen_examples(input_file, is_training):
  """Read a SQuAD json file into a list of QAExample."""
  input_data = []
  input_files = input_file.split(",")
  for ifile in input_files:
    with tf.gfile.Open(ifile, "r") as reader:
      input_data.extend(json.load(reader)["data"])
    
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  qids=set()
  examples = []
  for entry in input_data if FLAGS.verbose_logging else tqdm(input_data):
    if FLAGS.debug and len(examples) >= 100:
      break
    for paragraph in entry["paragraphs"]:
      question = paragraph["qas"][0]['question']
      if paragraph["context"].find(question) == -1:
         if paragraph["context"].startswith("[SEP]") == False:
           paragraph["context"] = "[SEP] " + paragraph["context"]
         paragraph["context"] = question + " " + paragraph["context"]
         paragraph["context"] = paragraph["context"].strip()
      last_title = ""
      paragraph["context"]  = _normalize(paragraph["context"]).replace("<t>","[SEP]").replace("</t>","[et]")
      if "[et]" in paragraph["context"]:
        last_title = paragraph["context"].split("[et]")[-1].split("[et]")[0].strip()
      if "[SEP]" in paragraph["context"]:
        last_title = paragraph["context"].split("[SEP]")[-1].split("[et]")[0].strip()
      paragraph_text = paragraph["context"]
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
          
      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        #if qas_id != '5a85b2d95542997b5ce40028_0_0':
        #  continue
        qids.add(qas_id)
        question_text = qa["question"]
        is_last_non_gt = False
        
        is_last_non_gt = qa["is_impossible"] if 'is_impossible' in qa else False
        is_alternative = qa['is_alternative'] if 'is_alternative' in qa else False
        is_continue = qa['is_continue'] if 'is_continue' in qa else False
        reader_switch = ReaderSwitch.NO_ANSWER if is_continue else ReaderSwitch.HAS_ANSWER 
        
        if is_last_non_gt and not is_alternative:
          new_query = ["", ""]
        else:
          new_query = [fix_query(paragraph_text, q) for q in query]
        query = new_query
        if query != new_query:
          query = new_query
        prev_query = fix_query(paragraph_text, prev_query)

        example = QAExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            doc_text=paragraph_text,
            is_last_non_gt=is_last_non_gt,
            is_alternative=is_alternative,
            reader_switch=reader_switch,
            query= query,
            prev_query=prev_query,
            last_doc = last_title)
        examples.append(example)
        
  return examples
  
def convert_qg_examples_to_features(examples, tokenizer, max_seq_length,
                                    doc_stride, max_query_length, is_training,
                                    output_fn, verbose_logging=False):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  input_features = {}

  last_group = InputFeaturesGroup("")

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
  
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
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

    # The -2 accounts for [CLS] and [SEP]
    max_tokens_for_doc = max_seq_length - 2

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
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
      break

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      seg_id = 0
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        if tokens[-1] in ['[SEP]', '[t]']:
          seg_id = 1
        segment_ids.append(seg_id)
      tokens.append("[SEP]")
      segment_ids.append(seg_id)

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
        query_label.append(0)
        query_label2.append(0)
        prev_query_label.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids)   == max_seq_length
      assert len(input_mask)  == max_seq_length
      assert len(segment_ids) == max_seq_length

      if example_index < 20:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("unique_id: %s" % (unique_id))
        tf.compat.v1.logging.info("example_index: %s" % (example_index))
        tf.compat.v1.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.compat.v1.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if example.is_last_non_gt:
          tf.compat.v1.logging.info("last_non_gt example")
        if example.is_alternative:
          tf.compat.v1.logging.info("alternative example")
        if is_training:
          tf.compat.v1.logging.info("reader_switch: %s" % (str(example.reader_switch)))
        tf.compat.v1.logging.info(
            "query_label: %s" % " ".join([str(x) for x in query_label]))
        tf.compat.v1.logging.info(
            "prev_query_label: %s" % " ".join([str(x) for x in prev_query_label]))
            
      input_feature = InputFeatures(
          qas_id=example.qas_id,
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          is_last_non_gt=example.is_last_non_gt,
          is_alternative=example.is_alternative,
          reader_switch=example.reader_switch,
          gen_query=gen_query,
          gen_query_mask=input_mask,
          gen_query_ids=query_label,
          gen_query_ids2=query_label2,
          prev_query_ids=prev_query_label,
          original_text=example.doc_text,
          last_doc=example.last_doc)
 
      if not is_training:
        if len(example.qas_id.split('_')) == 1:          
          if len(last_group.examples) == FLAGS.ranking_candidates:
            output_fn(last_group)
            last_group = InputFeaturesGroup("")
          last_group.append(input_feature)
        else:
          ori_id = example.qas_id[:len(example.qas_id)-2]
          if ori_id not in input_features:
            input_features[ori_id] = InputFeaturesGroup(ori_id)
          if len(input_features[ori_id].examples) == FLAGS.ranking_candidates:
            output_fn(input_features[ori_id])
            input_features[ori_id] = InputFeaturesGroup(ori_id)
  
          input_features[ori_id].append(input_feature)
      elif len(example.qas_id.split('_')) <= 2:
        if len(last_group.examples) == FLAGS.ranking_candidates:
          output_fn(last_group)
          last_group = InputFeaturesGroup("")
        last_group.append(input_feature)
      else:      
        ori_id = example.qas_id[:len(example.qas_id)-2]
        if ori_id not in input_features:
          input_features[ori_id] = InputFeaturesGroup(ori_id)
        if len(input_features[ori_id].examples) == FLAGS.ranking_candidates:
          output_fn(input_features[ori_id])
          input_features[ori_id] = InputFeaturesGroup(ori_id)

        input_features[ori_id].append(input_feature)

      unique_id += 1
  
  if len(last_group.examples) != 0:
    output_fn(last_group)
  
  for ori_id in input_features.keys():
    output_fn(input_features[ori_id])
    

def _remove_artifacts(text):
    rep = [["_",   " "], 
           [" ##", "" ], 
           ["[et]","" ], 
           ["[t]", "" ],
           ["[CLS]",""], 
           ["[SEP]",""]]
    for r in rep:
      text = text.replace(r[0], r[1])
    return text.strip()
    
def _ori_id(qas_id):
  if '_' not in qas_id:
    return qas_id
  id_splits = qas_id.split('_')
  return '_'.join(id_splits[:len(id_splits)-1])
  
  
def fix_reasoning_probs(tokenizer, 
                      all_features, 
                      all_results):
  val_ids = {}
  sum_ids = {}
  unique_id_to_result = {}
                      
  allf = []
  for al in all_features:
    for f in al.examples:
      allf.append(f)
  all_features = allf
  for result in all_results:
    unique_id_to_result[result.unique_id] = result
  
  for features in all_features:
    qas_id = features.qas_id
    # Making reference and result for blue metric
    if features.unique_id in unique_id_to_result.keys():
      result = unique_id_to_result[features.unique_id]
      if qas_id not in val_ids or val_ids[qas_id] < result.na_log_probs:
        val_ids[qas_id] = np.exp(result.na_log_probs) 
  for val_id in val_ids.keys():
    ori_id = _ori_id(val_id)
    if ori_id not in sum_ids:
      sum_ids[ori_id] = 0.0
    sum_ids[ori_id] += val_ids[val_id]
    
  fix_results = []
  for features in all_features:
    qas_id = features.qas_id
    # Making reference and result for blue metric
    if features.unique_id in unique_id_to_result.keys():
      result = unique_id_to_result[features.unique_id]
      fix_results.append(result._replace(na_log_probs = max(result.reader_log_probs, np.log(np.exp(result.na_log_probs) / sum_ids[_ori_id(qas_id)]))))
  return fix_results
  
def write_qg_predictions(tokenizer, 
                      all_features, 
                      all_results, 
                      output_prediction_file, 
                      output_prediction_best_file):

  # TODO : Need implementation for saving the result!
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  qas_id_to_qa_object = {}
  qas_id_to_qa_object_best = {}
  qas_id_to_qa_object_best_prob = {}
  qas_id_to_qa_object_best_na = {}
  allf = []
  for al in all_features:
    for f in al.examples:
      allf.append(f)
  all_features = allf
  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
  result_tokens_list = []
  trg_tokens_list = []

  na_correct_count = 0
  
  result_found = 0
  for features in all_features if FLAGS.verbose_logging else tqdm(all_features):
    qas_id = features.qas_id
    # Making reference and result for blue metric
    if features.unique_id in unique_id_to_result.keys():
      result = unique_id_to_result[features.unique_id]
      top_id = [result.query_log_probs[ti][0] for ti in range(len(features.tokens))]
      top_id = np.argsort(np.array(top_id))[:2]
      preds = []
      preds_ids = []
      add_word=False
      for ti, t in enumerate(features.tokens):
          if "##" in t and add_word:
              preds.append(t)
              preds_ids.append(ti)
          elif result.query_log_probs[ti][1]+FLAGS.advantage  > result.query_log_probs[ti][0] or \
               ti in top_id:
              if "##" in t:
                  curid=ti-1
                  prev_word=[]
                  prev_word_ids=[]
                  while curid>=0:
                      prev_word = [features.tokens[curid]] + prev_word
                      prev_word_ids = [curid] + prev_word_ids
                      if "##" not in features.tokens[curid] :
                          break
                      curid -= 1
                  preds = preds + prev_word
                  preds_ids = preds_ids + prev_word_ids
              preds.append(t)
              preds_ids.append(ti)
              add_word = True              
          elif "##" not in t:
              add_word = False
      
      prev_id = -99
      queries = []        
      for pi in preds_ids:
          if pi -1 == prev_id:
             queries[-1] = queries[-1] + "_" + features.tokens[pi].strip()
          else:
             queries.append(features.tokens[pi].strip())
          prev_id = pi
          

      def _run_strip_accents(text):
          """Strips accents from a piece of text."""
          text = unicodedata.normalize("NFD", text)
          output = []
          for char in text:
              cat = unicodedata.category(char)
              if cat == "Mn":
                  continue
              output.append(char)
          return "".join(output)        
      def fix_query(context, queries):
        new_queries = []
        without_space = ""
        char_index = []
        context_ori = context
        if FLAGS.do_lower_case:
          context = context.lower()
        context = _run_strip_accents(context)
        if len(context) != len(context_ori):
          context_ori = _run_strip_accents(context_ori)
        for ci, c in enumerate(context):
          if c != ' ':
            without_space += c
            char_index.append(ci)
        
        for q in queries:
          ori_q = _run_strip_accents(q)
          q = q.replace("_##","")
          if q.replace("_", " ") in context:
            q_idx = context.find(q.replace("_", " "))            
            new_queries.append(context_ori[q_idx:q_idx+len(q)])
          else:
            qns = q.replace("_","")
            cid = without_space.find(qns)
            if cid != -1:
              new_queries.append(context_ori[char_index[cid]:char_index[cid+len(qns)-1]+1].strip())
            else:
              new_queries.append(ori_q)
                              
        return " ".join(new_queries)
      tok_text = fix_query(features.original_text,queries) 
                
      # De-tokenize WordPieces that have been split off.
      tok_text = _remove_artifacts(tok_text)
    
      if qas_id not in qas_id_to_qa_object or result.na_log_probs > qas_id_to_qa_object[qas_id][1]:
        qas_id_to_qa_object[qas_id] = [tok_text, result.na_log_probs, features.last_doc]
      
        ori_id = _ori_id(qas_id)
        if ori_id not in qas_id_to_qa_object_best:
          qas_id_to_qa_object_best[ori_id] = []
        inserted=False
        for insert_id in range(len(qas_id_to_qa_object_best[ori_id])):
          if qas_id_to_qa_object_best[ori_id][insert_id][2] < result.na_log_probs:
            qas_id_to_qa_object_best[ori_id].insert(insert_id, [tok_text, features.last_doc, result.na_log_probs])
            inserted = True
            break
        if not inserted:
          qas_id_to_qa_object_best[ori_id].append([tok_text, features.last_doc, result.na_log_probs])
        
        if ori_id  not in qas_id_to_qa_object_best_prob or qas_id_to_qa_object_best_prob[ori_id] < result.na_log_probs:
          qas_id_to_qa_object_best_prob[ori_id] = result.na_log_probs
          qas_id_to_qa_object_best_na[ori_id] = False if features.is_last_non_gt else True 
         
      result_found += 1      

  qas_id_to_qa_object_best_out = {}
  qas_id_to_qa_object_best_out_probs = {}
  for _id in qas_id_to_qa_object_best.keys():
    query = []
    docs = []
    for top_n in range(len(qas_id_to_qa_object_best[_id][:FLAGS.choice_size])):
      if len(query)==0:
        query.append(qas_id_to_qa_object_best[_id][top_n][0])
        docs.append(qas_id_to_qa_object_best[_id][top_n][1])
    qas_id_to_qa_object_best_out[_id] = [" ".join(query), docs]
    qas_id_to_qa_object_best_out_probs[_id] = [" ".join(query), docs, qas_id_to_qa_object_best_prob[_id]]
  tf.logging.info("Result found %d / %d" % (result_found,len(all_features)))
  # Making SQuAD-like json form as output
  
  sum_exp = {}
  for qas_id in qas_id_to_qa_object.keys():
    ori_id = _ori_id(qas_id)
    if ori_id not in sum_exp:
      sum_exp[ori_id] = 0.0
    sum_exp[ori_id] += np.exp(qas_id_to_qa_object[qas_id][1])
    
  for qas_id in qas_id_to_qa_object_best_out_probs.keys():
    qas_id_to_qa_object_best_out_probs[qas_id].append(np.exp(qas_id_to_qa_object_best_prob[qas_id]) / sum_exp[qas_id])
    
  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(qas_id_to_qa_object, indent=4, ensure_ascii=False))
  with tf.gfile.GFile(output_prediction_best_file, "w") as writer:
    writer.write(json.dumps(qas_id_to_qa_object_best_out, indent=4, ensure_ascii=False))
  with tf.gfile.GFile(output_prediction_best_file.replace(".json","_prob.json"), "w") as writer:
    writer.write(json.dumps(qas_id_to_qa_object_best_out_probs, indent=4, ensure_ascii=False))
  acc_cnt = 0
  for na_a in qas_id_to_qa_object_best_na.keys():
    if qas_id_to_qa_object_best_na[na_a]:
      acc_cnt += 1
    else:
      if FLAGS.verbose_logging:
        print(na_a)
  tf.logging.info("NA acc2 %d / %d" % (acc_cnt,len(qas_id_to_qa_object_best_na)))
    
  return result_tokens_list, trg_tokens_list
