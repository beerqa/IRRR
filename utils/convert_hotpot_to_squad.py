from argparse import ArgumentParser
import json
from scripts.index_processed_wiki import STOPWORDS
from scripts.gen_hotpot_hop123_imp_ngram_2 import _has_connection

def main(args):
    with open(args.input_file) as f:
        data = json.load(f)

    output_data = []

    for d in data:
        last_non_gt = True
        last_retrieved = None

        if 'supporting_facts' in d:
            supporting_facts_left = d['supporting_facts']
            if len(d['retrieved_context']) > 0:
                last_retrieved = d['retrieved_context'][-1]
                if 'wiki_para_id' in d:
                    if len(supporting_facts_left) > 0 and last_retrieved['docid'] == d['wiki_para_id']:
                        last_non_gt = False
                else:
                    if any(last_retrieved['title'] == x[0] for x in supporting_facts_left):
                        last_non_gt = False

        answer = []
        if isinstance(d['answer'], list):
            if any(a in d['context'] for a in d['answer']):
                for a in set(d['answer']):
                    for m in re.finditer(a, d['context']):
                        answer.append({"text": a, "answer_start": m.start()})
        elif d['answer'] in d['context']:
            if d['answer'] in ['yes', 'no']:
                if 'supporting_facts' in d and len(d['supporting_facts']) == 0:
                    answer.append({"text": d['answer'], "answer_start": -1})
            else:
                for m in re.finditer(d['answer'], d['context']):
                    answer.append({"text": d['answer'], "answer_start": m.start()})

        if len(answer) == 0:
            answer.append({"text": "", "answer_start": -1})

        output_data.append({
            'title': '',
            'retrieved_context': d['retrieved_context'],
            'paragraphs': [{
                'context': d['context'] if not args.include_question else (" [SEP] ".join([x for x in [d['question'], d['context']] if len(x) > 0])),
                'query': d['query'],
                'query_short': d['query_short'],
                'prev_query': d.get('prev_query', ''),
                'prev_query_short': d.get('prev_query_short', ''),
                'qas': [{
                    'id': d['_id'],
                    'question': d['question'],
                    'answers':  answer,
                    'is_impossible': 'supporting_facts' not in d or len(d['supporting_facts']) > 0,
                    'is_last_non_gt': last_non_gt,
                    'is_alternative': False if last_retrieved is None else _has_connection(last_retrieved, d['last_target_para']['title'], True)
                }]
            }]
        })

        if 'supporting_facts' in d:
            output_data[-1]['supporting_facts'] = d['supporting_facts']

    with open(args.output_file, 'w') as f:
        json.dump({'data': output_data}, f)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--include_question', action='store_true', help="Include question before the context for the query generator")

    args = parser.parse_args()

    main(args)
