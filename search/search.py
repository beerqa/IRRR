from elasticsearch import Elasticsearch
import json
import re

from utils.constant import WIKIPEDIA_DOC_PARA_INDEX_NAME

es = Elasticsearch(timeout=300)

core_title_matcher = re.compile('([^()]+[^\s()])(?:\s*\(.+\))?')
core_title_filter = lambda x: core_title_matcher.match(x).group(1) if core_title_matcher.match(x) else x

def _extract_one(item, lazy=False):
    res = {k: item['_source'][k] if k != 'text' else item['_source'].get(k, item['_source'].get('doc_text', None)) for k in ['id', 'url', 'title', 'text', 'title_unescape', 'docid']}
    res['_score'] = item['_score']
    res['data_object'] = item['_source']['original_json'] if lazy else json.loads(item['_source']['original_json'])

    return res

def _single_query_constructor(query, topn=50, exclude_title=False):
    fields = ["text"] if exclude_title else ["title^1.25", "title_unescape^1.25", "text"]
    return {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {
                            "query": query,
                            "fields": fields,
                        }},
                    ],
                    "should": [
                        { "has_parent": {
                            "parent_type": "doc",
                            "score": True,
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": [x if x != 'text' else 'doc_text' for x in fields],
                                    "boost": 0.2
                                },
                            }
                        }}
                    ],
                    "filter": [
                        {"term": {
                            "doctype": "para"
                        }}
                    ],
                }
            },
            "size": topn
            }

def single_doc_query(query, topn=10, lazy=False, index=WIKIPEDIA_DOC_PARA_INDEX_NAME, title=None, match_phrase=False, title_only=True):
    query_type = 'match_phrase' if match_phrase else 'match'
    query_field = 'title' if title_only else 'doc_text'
    body = {"query": { "bool": { "must": [{query_type: {query_field: query}}], "filter":[{"term": {"doctype":"doc"}}]}}, "size": topn}
    if title is not None:
        body['query']['bool']['must'] = [{'match_phrase': {'title_unescape': title}}]
        del body['query']['bool']['should']
    res = es.search(index=index, doc_type='doc', body=json.dumps(body))

    res = [_extract_one(x, lazy=lazy) for x in res['hits']['hits']]
    res = rerank_with_query(query, res)[:topn]

    return res

def single_text_query(query, topn=10, lazy=False, rerank_topn=50, index=WIKIPEDIA_DOC_PARA_INDEX_NAME, title=None):
    body = _single_query_constructor(query, topn=max(topn, rerank_topn))
    if title is not None:
        body['query']['bool']['must'] = [{'match_phrase': {'title_unescape': title}}]
        del body['query']['bool']['should']
    res = es.search(index=index, doc_type='doc', body=json.dumps(body))

    res = [_extract_one(x, lazy=lazy) for x in res['hits']['hits']]
    res = rerank_with_query(query, res)[:topn]

    return res

def bulk_text_query(queries, topn=10, lazy=False, rerank_topn=50, index=WIKIPEDIA_DOC_PARA_INDEX_NAME):
    body = ["{}\n" + json.dumps(_single_query_constructor(query, topn=max(topn, rerank_topn))) for query in queries]
    res = es.msearch(index=index, doc_type='doc', body='\n'.join(body))

    res = [[_extract_one(x, lazy=lazy) for x in r['hits']['hits']] for r in res['responses']]
    res = [rerank_with_query(query, results)[:topn] for query, results in zip(queries, res)]

    return res

def rerank_with_query(query, results):
    def score_boost(item, query):
        score = item['_score']
        core_title = core_title_filter(item['title_unescape'])
        if query.startswith('The ') or query.startswith('the '):
            query1 = query[4:]
        else:
            query1 = query
        if query == item['title_unescape'] or query1 == item['title_unescape']:
            score *= 1.5
        elif query.lower() == item['title_unescape'].lower() or query1.lower() == item['title_unescape'].lower():
            score *= 1.2
        elif item['title'].lower() in query:
            score *= 1.1
        elif query == core_title or query1 == core_title:
            score *= 1.2
        elif query.lower() == core_title.lower() or query1.lower() == core_title.lower():
            score *= 1.1
        elif core_title.lower() in query.lower():
            score *= 1.05

        item['_score'] = score
        return item

    return list(sorted([score_boost(item, query) for item in results], key=lambda item: -item['_score']))

if __name__ == "__main__":
    print([x['title'] for x in single_text_query("In which city did Mark Zuckerberg go to college?")])
    print([[y['title'] for y in x] for x in bulk_text_query(["In which city did Mark Zuckerberg go to college?"])])
