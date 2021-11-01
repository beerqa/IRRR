from argparse import ArgumentParser
import bz2
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from elasticsearch import Elasticsearch
from glob import glob
import html
import json
from multiprocessing import Pool
import numpy as np
import os
import pickle
import re
import sqlite3
from tqdm import tqdm
from urllib.parse import unquote

from utils.constant import WIKIPEDIA_DOC_PARA_INDEX_NAME, SQUAD_WIKIPEDIA_DOC_PARA_INDEX_NAME, HOTPOT_WIKIPEDIA_DOC_PARA_INDEX_NAME, SQUAD_WIKITITLE_MAPPING, BEERQA_WIKIPEDIA_DOC_PARA_INDEX_NAME
from utils.corenlp import get_sentence_list, bulk_tokenize
from utils.general import chunks

# stop words copied from DrQA
STOPWORDS = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
]


def process_line(line):
    data = json.loads(line)
    docid = f"hotpot_wiki-{data['id']}"
    data['docid'] = docid
    item = {'id': data['id'],
            'url': data['url'],
            'title': data['title'],
            'title_unescape': html.unescape(data['title']),
            'doc_text': ''.join(data['text']),
            'original_json': json.dumps(data)
            'docid': docid,
            'doctype': 'doc',
            }
    # tell elasticsearch we're indexing documents
    body = ["{}\n{}".format(json.dumps({ 'index': { '_id': docid, '_routing': docid } }), json.dumps(item))]

    item['doctype'] = { 'name': 'para', 'parent': docid }
    item['text'] = item['doc_text']
    del item['doc_text']
    paraid = f"{docid}-0"
    item['docid'] = paraid
    data['docid'] = paraid
    item['original_json'] = json.dumps(data)
    body.append("{}\n{}".format(json.dumps({ 'index': { '_id': paraid, '_routing': docid } }), json.dumps(item)))

    return ('\n'.join(body))

def generate_hotpot_indexing_queries_from_bz2(bz2file, dry=False):
    if dry:
        return

    with bz2.open(bz2file, 'rt') as f:
        yield from (process_line(line) for line in f)

def process_line_beerqa_bz2(line):
    data = json.loads(line)
    docid = f"wiki-{data['id']}"
    item = {'id': data['id'],
            'url': data['url'],
            'title': data['title'],
            'title_unescape': html.unescape(data['title']),
            'doc_text': ''.join(data['text']),
            'original_json': json.dumps({'id': data['id'], 'url': data['url'], 'title': data['title'], 'text': data['text'], 'docidid': docid}), #line,
            'docid': docid,
            'doctype': 'doc',
            }
    # tell elasticsearch we're indexing documents
    yield "{}\n{}".format(json.dumps({ 'index': { '_id': docid, '_routing': docid } }), json.dumps(item))

    for para_id, para in enumerate(''.join(data['text']).split('\n')):
        if len(para.strip()) == 0 or len(para.split()) <= 10:
            # skip empty paragraphs, section titles, and the page title
            continue

        paraid = f"{docid}-{para_id}"
        obj = {'title': data['title'] , 'text': get_sentence_list(para), 'docid': paraid}
        item = {'id': paraid,
                'url': 'no_url',
                'doctype': { 'name': 'para', 'parent': docid },
                'title': data['title'],
                'title_unescape': html.unescape(data['title']),
                'text': para,
                'original_json': json.dumps(obj),
                'docid': paraid
                }

        yield "{}\n{}".format(json.dumps({ 'index': { '_id': paraid, '_routing': docid } }), json.dumps(item))

def generate_indexing_queries_from_beerqa_bz2(bz2file, dry=False):
    if dry:
        return

    with bz2.open(bz2file, 'rt') as f:
        for line in f:
            yield from process_line_beerqa_bz2(line)

def generate_squad_indexing_queries(data, idx, squad_paras, dry=False):
    if dry:
        return

    data['title'] = data['title'].replace('é', 'é').replace('í', 'í')

    if data['title'] in squad_paras:
        # see if we need to add squad paragraphs to the text
        missing_paras = []

        text_paras = [x for x in data['text'].split('\n') if len(x.strip()) != 0]

        paras_from_squad = [(x[0].replace('\n', ' '), x[1]) for x in squad_paras[data['title']]]

        tokenized = bulk_tokenize([x[0] for x in paras_from_squad] + text_paras)
        n_squad_paras = len(paras_from_squad)

        assert len(tokenized) == n_squad_paras + len(text_paras)

        mapped = set()

        for p, tok_p in zip(paras_from_squad, tokenized[:n_squad_paras]):
            found = False
            best = None
            best_i = -1
            best_overlap = 0
            for i, (p1, tok_p1) in enumerate(zip(text_paras, tokenized[n_squad_paras:])):
                m = SequenceMatcher(a=tok_p, b=tok_p1, autojunk=False)
                overlap = m.ratio() * (len(tok_p) + len(tok_p1)) / 2

                if overlap > best_overlap:
                    best_overlap = overlap
                    best = p1
                    best_i = i

            if best_overlap < .8 * len(tok_p):
                # add squad para into the document if sufficient overlap can't be found
                missing_paras.append(p[0])
            elif not all(ans in best for ans in p[1]):
                if best_i not in mapped:
                    # replace wiki para with squad para if not all answers can be found in the former
                    text_paras[best_i] = p[0]
                else:
                    missing_paras.append(p[0])
            mapped.add(best_i)

        data['text'] = '\n'.join(text_paras + missing_paras)

        del squad_paras[data['title']]

    yield from _generate_squad_indexing_queries(data, idx, dry=dry)

def _generate_squad_indexing_queries(data, idx, dry=False):
    obj = {'title': data['title'] , 'text': data['text'], 'docid': f'squad_wiki-{idx}'}
    item = {'id': idx,
            'url': 'no_url',
            'doctype': 'doc',
            'title': data['title'],
            'title_unescape': html.unescape(data['title']),
            'doc_text': data['text'],
            'original_json': json.dumps(obj),
            'docid': f'squad_wiki-{idx}'
            }
    yield ("{}\n{}".format(json.dumps({ 'index': { '_id': f'squad_wiki-{idx}', "_routing": f'squad_wiki-{idx}' } }), json.dumps(item)))

    for para_id, para in enumerate(data['text'].split('\n')):
        if len(para.strip()) == 0 or len(para.split()) <= 10:
            # skip empty paragraphs, section titles, and the page title
            continue
        obj = {'title': data['title'] , 'text': get_sentence_list(para), 'docid': f'squad_wiki-{idx}-{para_id}'}
        item = {'id': idx,
                'url': 'no_url',
                'doctype': { 'name': 'para', 'parent': f'squad_wiki-{idx}' },
                'title': data['title'],
                'title_unescape': html.unescape(data['title']),
                'text': para,
                'original_json': json.dumps(obj),
                'docid': f'squad_wiki-{idx}-{para_id}'
                }

        yield ("{}\n{}".format(json.dumps({ 'index': { '_id': f'squad_wiki-{idx}-{para_id}', '_routing': f'squad_wiki-{idx}' } }), json.dumps(item)))

def index_chunk(chunk, index):
    es = Elasticsearch(timeout=600)
    res = es.bulk(index=index, doc_type='doc', body='\n'.join(chunk), timeout='600s')
    assert not res['errors'], res
    return len(chunk)

def ensure_index(index, reindex):
    es = Elasticsearch(timeout=600)
    if es.indices.exists(index=index) and reindex:
        print('deleting index...')
        es.indices.delete(index=index)
    if not es.indices.exists(index=index):
        print('creating index...')
        es.indices.create(index=index,
                body=json.dumps({
                    "mappings":{"doc":{"properties": {
                        "id": { "type": "keyword" },
                        "url": { "type": "keyword" },
                        "docid": { "type": "keyword" },
                        "doctype": { "type": "join", "relations": { "doc": "para"} },
                        "title": { "type": "text", "analyzer": "simple_bigram_analyzer" },
                        "title_unescape": { "type": "text", "analyzer": "simple_bigram_analyzer" },
                        "text": { "type": "text", "analyzer": "bigram_analyzer" },
                        "doc_text": { "type": "text", "analyzer": "bigram_analyzer", "similarity": "scripted_bm25" },
                        "original_json": { "type": "text" },
                        }},
                    },
    		    "settings": {
                        "analysis": {
                            "analyzer":{
                                "simple_bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                         "lowercase", "shingle"
                                    ]
                                },
                                "bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase", "my_stop", "shingle", "remove_filler", "remove_empty"
                                    ]
                                }
                            },
                            "filter":{
                                "my_stop": {
                                    "type": "stop",
                                    "stopwords": STOPWORDS
                                },
                                "remove_filler": {
                                    "type": "pattern_replace",
                                    "pattern": ".*_.*",
                                    "replace": "",
                                },
                                "remove_empty": {
                                    "type": "stop",
                                    "stopwords": [""]
                                }
                            }
                        },
                        "index": {
                            "similarity": {
                                "scripted_bm25": {
                                    "type": "scripted",
                                    "script": {
                                        "source": "double tf = doc.freq * (1 + 1.2) / (doc.freq + 1.2); double idf = Math.max(0, Math.log((field.docCount-term.docFreq+0.5)/(term.docFreq+0.5))); return query.boost * tf * Math.pow(idf, 2);"
                                    }
                                },
                            }
                        },
                    }
                    }))

def query_generator(index_type):
    if index_type == 'hotpot':
        filelist = glob('data/enwiki-20171001-pages-meta-current-withlinks-abstracts/*/wiki_*.bz2')
        for f in tqdm(filelist, position=1):
            yield from generate_hotpot_indexing_queries_from_bz2(f)
    if index_type == 'squad':
        squad_paras = defaultdict(list)
        with open('data/squad/train-v1.1.json') as f:
            d_train = json.load(f)
        with open('data/squad/dev-v1.1.json') as f:
            d_dev = json.load(f)
        for article in d_train['data'] + d_dev['data']:
            article_title = unquote(article['title'].replace('_', ' '))
            article_title = SQUAD_WIKITITLE_MAPPING.get(article_title, article_title)
            for para in article['paragraphs']:
                squad_paras[article_title].append((para['context'], [ans['text'] for qa in para['qas'] for ans in qa['answers']]))

        idx = 0
        conn = sqlite3.connect('DrQA/data/wikipedia/docs.db')
        rows = conn.execute("SELECT * FROM documents")
        for idx, row in tqdm(enumerate(rows), position=1):
            entry = {'title': row[0].replace('_', ' '), 'text': row[1]}
            yield from generate_squad_indexing_queries(entry, idx, squad_paras)

        print('Indexing remaining unindexed squad articles:', squad_paras.keys())
        for title in tqdm(squad_paras, position=1):
            idx += 1
            entry = {'title': title, 'text': '\n'.join([x[0].replace('\n', ' ') for x in squad_paras[title]])}
            yield from _generate_squad_indexing_queries(entry, idx)
    if index_type == "beerqa":
        #filelist = glob('data/enwiki-20200801-pages-articles-tokenized/*/wiki_*.bz2')
        filelist = glob('/john10/scr1/pengqi/enwiki-20200801-pages-articles-tokenized/*/wiki_*.bz2')
        for f in tqdm(filelist, position=1):
            yield from generate_indexing_queries_from_beerqa_bz2(f)

def main(args):
    if args.type == 'squad':
        index = SQUAD_WIKIPEDIA_DOC_PARA_INDEX_NAME
    elif args.type == 'hotpot':
        index = HOTPOT_WIKIPEDIA_DOC_PARA_INDEX_NAME
    elif args.type == 'beerqa':
        index = BEERQA_WIKIPEDIA_DOC_PARA_INDEX_NAME

    if not args.dry:
        print(f'Making index "{index}"...')
        ensure_index(index, args.reindex)

        print('Indexing...')

        pbar = tqdm(total=-1, position=0)
        def update(*a):
            pbar.update(a[0])

        pool = Pool()
        chunksize = 8192
        for i, chunk in enumerate(chunks(query_generator(args.type), chunksize)):
            pool.apply_async(index_chunk, [chunk, index], error_callback=print, callback=update)
            #index_chunk(chunk, index)
            #update(len(chunk))

        pool.close()
        pool.join()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('type', choices=['squad', 'hotpot', 'beerqa'], default='beerqa')
    parser.add_argument('--reindex', action='store_true', help="Reindex everything")
    parser.add_argument('--dry', action='store_true', help="Dry run")

    args = parser.parse_args()

    main(args)
