"""
Query ES and merge results with original hotpot data.

Input:
    - query file
    - hotpotqa data
    - output filename
    - whether this is for hop1 or hop2

Outputs:
    - json file containing a list of:
        {'context', 'question', '_id', 'query', 'json_context'}
        context -- the concatentation of the top n paragraphs for the given query
            to ES.
        json_context -- same as context, but in json structure same as original
            hotpot data.
        question, _id -- identical to those from the original HotPotQA data
"""

import argparse
from tqdm import tqdm
from search.search import bulk_text_query
from utils.io import load_json_file, write_json_file
from utils.general import chunks, make_context
from collections import Counter
from nltk.corpus import stopwords
import os

STOP_WORDS = set(stopwords.words('english') + [',', '.', ';', '?', '"', '\'', '(', ')', '&', '?'])

def main(query_file, query_all_file, question_file, input_file, out_file, recall_out_file, retrieved_titles_file, top_n, include_prev, prev_titles, args):
    ### additions for recall metrics ###
    #top_n = top_n + PREV_COUNT
    Ns = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,100,200,500]
    max_n = max(Ns + [top_n*3])
    para1 = Counter()
    para2 = Counter()
    processed = 0
    PREV_COUNT = int(top_n/5)

    query_data = load_json_file(query_file)
    query_all_data = load_json_file(query_all_file)
    question_data = load_json_file(question_file)
    input_data = load_json_file(input_file)

    input_map = {}
    input_all_map = {}
    context_map = {}
    context_titles = {}
    context_titles_all = {}

    if args.expand_ids:
        query_data = query_all_data

    expand_ids = set()
    for input_datum in input_data['data']:
        _id = input_datum['paragraphs'][0]['qas'][0]['id']
        ori_id = _id if args.expand_ids else _id.split('_')[0]

        if _id not in query_all_data:
            continue            
            
        if input_datum['paragraphs'][0].get('expand', True):
            expand_ids.add(ori_id)

        #TODO: not beam search friendly
        if ori_id not in input_map or query_all_data[_id][1] > input_map[ori_id][0]:
            input_map[ori_id] = (query_all_data[_id][1], input_datum)
        if ori_id not in input_all_map:
            input_all_map[ori_id] = []
        input_all_map[ori_id].append([query_all_data[_id][1], input_datum])

        context_split = input_datum['paragraphs'][0]['context'].split('<t>')
        q = context_split[0].strip()
        if len(context_split) > 1:
            context = " <t> " + context_split[1].strip()
            title = context_split[1].split("</t>")[0].strip()
            if ori_id not in context_map:
                context_map[ori_id] = q
                context_titles[ori_id] = []
                context_titles_all[ori_id] = []
            if title in query_data[ori_id][1]:
                context_map[ori_id] += context
                context_titles[ori_id].append(title)
            context_titles_all[ori_id].append(title)
        else:
            context = ""
            title = ""
            context_map[ori_id] = input_datum['paragraphs'][0]['context']

    question_data = [ q for q in question_data['data'] if q['id'] in query_data and q['id'] in expand_ids ]

    prev_titles_data = {}
    if include_prev:
        if os.path.isfile(prev_titles):
            prev_titles_data = load_json_file(prev_titles)
        else:
            for key in input_all_map.keys():
                prev_titles_data[key] = []

        for key in input_all_map.keys():
            input_all_map[key] = sorted(input_all_map[key], key=lambda datum: datum[0], reverse=True)


    ### for debugging purposes ###
    #question_data = question_data[:100]
    #print("len(question_data)", len(question_data))
    out_data = []
    recall_retrieved_titles_data = {}

    if isinstance(query_data, list):
        qd = {}
        for q in query_data:
            qd[q['id']] = q['query']
        query_data = qd

    for chunk in tqdm(list(chunks(question_data, 10))):
        queries = []
        contexts = []
        for datum in chunk:
            _id = datum['id']
            q = query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0].replace("_", " ")
            queries.append(q)
            contexts.append(datum['question'])

        es_results = bulk_text_query(queries, topn=max_n+args.hop, lazy=False, index=args.index)
        for es_result, datum in zip(es_results, chunk):
            _id = datum['id']
            
            if include_prev:
                es_result = [res for res in es_result if res['docid'] not in prev_titles_data[_id]]
                prev_result = []
                for pv in input_all_map[_id][1:1+PREV_COUNT]:
                  prev_result.append({
                    'title':pv[1]['paragraphs'][0]['retrieved_context'][-1]['title'],
                    'data_object': { 'text':pv[1]['paragraphs'][0]['retrieved_context'][-1]['context'] },
                    'text': "".join(pv[1]['paragraphs'][0]['retrieved_context'][-1]['context']),
                    'docid': ""
                  })

                es_result = prev_result + es_result
            
            retrieved_context = input_map[_id][1]['paragraphs'][0].get('retrieved_context', [])

            # filter out retrieved paragraphs
            retrieved_docids = set([x['docid'] for x in retrieved_context])
            filtered_es_result = [x for x in es_result if x['docid'] not in retrieved_docids]
            if 'supporting_facts' in datum:
                supporting_titles = sorted(set(x[0] for x in datum['supporting_facts'])) # paragraph titles
            else:
                supporting_titles = set()#(x[0], ''.join(x[1])) for x in datum.get('original_context', datum['context']))

            if args.ensure_gold:
                filtered_es_result_cand = filtered_es_result

                # if gold para is not in top_n search results, attempt to find it in more search results
                found = []
                if 'supporting_facts' not in datum:
                    found = [(i, x) for i, x in enumerate(filtered_es_result) if (x['title'], ''.join(x['text'])) in supporting_titles]
                elif 'wiki_para_id' in datum and any(x['docid'] == datum['wiki_para_id'] for x in filtered_es_result_cand):
                    found = [(i, x) for i, x in enumerate(filtered_es_result) if x['docid'] == datum['wiki_para_id']]
                elif 'wiki_para_id' not in datum and any(x['title'] in supporting_titles for x in filtered_es_result_cand):
                    found = [(i, x) for i, x in enumerate(filtered_es_result) if x['title'] in supporting_titles]

                if len(found) == 0:
                    # target paragraph can't be retrieved
                    continue
                else:
                    # target paragraph retrieved but not ranked top
                    filtered_es_result_cand = filtered_es_result_cand[:found[0][0]] + filtered_es_result_cand[found[0][0]+1:]
                    filtered_es_result_cand = [found[0][1]] + filtered_es_result_cand[:(top_n-1)]

                filtered_es_result = filtered_es_result_cand
            else:
                filtered_es_result = filtered_es_result[:top_n]

            question = datum['question']
            query = query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0].replace("_", " ")#[0]
            context = make_context(question, filtered_es_result)
            json_context = [
                [p['title'], p['data_object']['text']]
                for p in filtered_es_result
            ]

            out_data.append({
                'id': _id,
                'question': question,
                'answer': datum['answer'] if 'answer' in datum else "",
                'context': context,
                'retrieved_context': retrieved_context,
                'last_retrieved': [{k: x[k] for k in ['title', 'text', 'docid', 'data_object']} for x in filtered_es_result],
                'query': query_data[_id] if isinstance(query_data[_id], str) else query_data[_id][0],
                'json_context': json_context
            })
            ### recall calculations ##
            r = filtered_es_result
            para1_found = False
            para2_found = False
            para1_i = -1
            para2_i = -1
            
            # exclude previously found paragraphs
            for i, para in enumerate(retrieved_context):
                if ('supporting_facts' in datum and para['title'] in supporting_titles) or ('supporting_facts' not in datum and (para['title'], ''.join(para['context'])) in supporting_titles):
                    if not para1_found:
                        para1[0] += 1
                        para1_found = True
                    elif not para2_found:
                        #assert not para2_found
                        para2[0] += 1
                        para2_found = True
                    if 'supporting_facts' in datum:
                        supporting_titles.remove(para['title'])
                    else:
                        supporting_titles.remove((para['title'], ''.join(para['context'])))
            for i, para in enumerate(r):
                if ('supporting_facts' in datum and para['title'] in supporting_titles) or ('supporting_facts' not in datum and (para['title'], ''.join(para['text'])) in supporting_titles):
                    if not para1_found:
                        para1[i] += 1
                        para1_i= i
                        para1_found = True
                    elif not para2_found:
                        #assert not para2_found
                        para2[i] += 1
                        para2_i = i
                        para2_found = True
                    if 'supporting_facts' in datum:
                        supporting_titles.remove(para['title'])
                    else:
                        supporting_titles.remove((para['title'], ''.join(para['text'])))

            retrieved_titles = [para['docid'] for para in r if len(para['docid']) != 0]

            recall_retrieved_titles_data[_id] = retrieved_titles

            if include_prev and _id in prev_titles_data:
                recall_retrieved_titles_data[_id] = prev_titles_data[_id] + recall_retrieved_titles_data[_id][:(top_n-len(prev_titles_data[_id]))]
            if not para1_found:
                para1[max_n] += 1
            if not para2_found:
                para2[max_n] += 1

            #print(query)
            #print(str(para1_i) + " " + str(para2_i))
            #import pdb
            #pdb.set_trace()
        processed += len(chunk)

    if processed > 0:
        with open(recall_out_file,'w+') as f:
            for n in Ns:
                c1 = sum(para1[k] for k in range(n))
                c2 = sum(para2[k] for k in range(n))
                f.write("Recall-P1@{:2d}: {:.2f}\tRecall-P2@{:2d}: {:.2f}\n".format(n, 100 * c1 / processed, n, 100 * c2 / processed))
                print("Recall-P1@{:2d}: {:.2f}\tRecall-P2@{:2d}: {:.2f}".format(n, 100 * c1 / processed, n, 100 * c2 / processed))

    write_json_file(out_data, out_file)
    write_json_file(recall_retrieved_titles_data, retrieved_titles_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query ES and merge results with original hotpot data.')
    parser.add_argument('query_file', help='.preds file containing the best ES queries after reranking')
    parser.add_argument('query_all_file', help='.preds file containing all ES queries ')
    parser.add_argument('question_file', help='.json file containing original questions and ids')
    parser.add_argument('input_file', help='.json file containing original questions and ids')
    parser.add_argument('out_file', help='filename to write data out to')
    parser.add_argument('recall_out_file', help='filename to write recall data out to')
    parser.add_argument('retrieved_titles_file', help='filename to write recall data out to')
    parser.add_argument('--top_n', default=5,
            help='number of docs to return from  ES',
            type=int)
    parser.add_argument('--hop', default=0,
            help='Hop number',
            type=int)
    parser.add_argument('--index', default='hotpot_wikipedia_doc_para',
            type=str)
    parser.add_argument('--include_prev', action='store_true')
    parser.add_argument('--expand_ids', action='store_true')
    parser.add_argument('--ensure_gold', action='store_true', help="Make sure that each reranking set in the output has at least one gold paragraph in it")
    parser.add_argument('--prev_titles', default=None,
            type=str)
    args = parser.parse_args()

    main(args.query_file, args.query_all_file, args.question_file, args.input_file, args.out_file, args.recall_out_file, args.retrieved_titles_file, args.top_n, args.include_prev, args.prev_titles, args)
