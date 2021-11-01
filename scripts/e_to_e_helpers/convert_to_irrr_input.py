"""
Query ES and merge results with original hotpot data.

Input:
    - query file
    - hotpotqa data
    - output filename
    - whether this is for hop1 or hop2

Outputs:
    - json file containing a list of:
        {'context', 'question', 'id', 'query', 'json_context'}
        context -- the concatentation of the top n paragraphs for the given query
            to ES.
        json_context -- same as context, but in json structure same as original
            hotpot data.
        question, id -- identical to those from the original HotPotQA data
"""

import argparse
import random
import re
from tqdm import tqdm
from search.search import bulk_text_query
from utils.io import load_json_file, write_json_file
from utils.general import chunks, make_context
from collections import Counter, defaultdict

def has_connection(source, target, text_match):
    if text_match:
      return target.split('(')[0].strip().lower() in source['title'].lower() + "".join(source['text']).lower()
    else:
        links = _get_links("".join(source['text_with_links']))
        return target in links

def sufficient_context(datum, ground_truth, last_retrieved=None):
    retrieved = datum.get('retrieved_context', [])
    if last_retrieved is not None:
        retrieved = retrieved + [last_retrieved]
    if len(ground_truth) > 0 and isinstance(list(ground_truth)[0], tuple):
        # new format where all context paragraphs are supporting facts
        retrieved_paras = set((x['title'], ''.join(x.get('text', x.get('context', '')))) for x in retrieved)
        return len(retrieved_paras.intersection(ground_truth)) == len(ground_truth)
    elif 'wiki_para_id' in datum:
        return any(x['docid'] == datum['wiki_para_id'] for x in retrieved)
    else:
        retrieved_titles = set(x['title'] for x in retrieved)
        return len(ground_truth.intersection(retrieved_titles)) == len(ground_truth)

def answer_offsets(answer, context):
    if answer in ['yes', 'no'] or answer not in context:
        return [-1]
    else:
        return [context.find(answer)] #[m.start() for m in re.finditer(answer, context)]

def answers_from_answer_and_context(answer, new_context, is_impossible):
    if answer =="":
      return [{'text': '', 'answer_start': -1}]
    if answer not in new_context:
        if not is_impossible and answer in ['yes', 'no']:
            answers = [{'text': 'yes', 'answer_start': -1}]
        else:
            answers = [{'text': '', 'answer_start': -1}]
    else:
        answers = [{'text': answer, 'answer_start': st} for st in answer_offsets(answer, new_context)]

    return answers

def main(hotpot_file, input_file, output_filie, train, original_question_file=None, ranking_score_file=None, top_K=10000):
    hotpot_data = load_json_file(hotpot_file)
    gts = {}
    for hp in hotpot_data['data']:
        if hp['id'] not in gts:
            gts[hp['id']] = set()
        if 'supporting_facts' in hp:
            for sp in hp["supporting_facts"]:
                gts[hp['id']].add(sp[0])
        else:
            #for title, text in hp.get('original_context', hp['context']):
            gts[hp['id']] = set() #add((title, ''.join(text)))

    input_data = load_json_file(input_file)

    if original_question_file is not None:
        question_data = load_json_file(original_question_file)
        question_data = {x['id']: x for x in question_data}

    if ranking_score_file is not None:
        ranking_score = load_json_file(ranking_score_file)

        grouped_ranking_score = defaultdict(list)
        for k in ranking_score:
            ksplit = k.split("_")
            groupid = '_'.join(ksplit[:-1])
            ingroupid = ksplit[-1]

            grouped_ranking_score[groupid].append((ranking_score[k][1], ingroupid))

        # filter top K
        for groupid in grouped_ranking_score:
            grouped_ranking_score[groupid] = sorted(grouped_ranking_score[groupid], key=lambda x: x[0], reverse=True)[:top_K]

        retained_ids = set([f"{groupid}_{ingroup[1]}" for groupid in grouped_ranking_score for ingroup in grouped_ranking_score[groupid]])

        ranking_score = {k: ranking_score[k] for k in ranking_score if k in retained_ids}

    ### for debugging purposes ###
    #question_data = question_data[:100]
    #print("len(question_data)", len(question_data))
    out_data = []

    expand_counter = Counter()
    for datum in input_data:
        _id = datum['id']
        question = datum['question']
        label = datum['label'] if 'label' in datum else ""
        label_s = datum['label_short'] if 'label_short' in datum else ""
        label_offsets = datum['label_offsets'] if 'label_offests' in datum else 0
        hop1_label = datum['query'] if 'query' in datum else ""

        gt_start = datum['gt_start'] if 'gt_start' in datum else 0
        gt_end = datum['gt_end'] if 'gt_end' in datum else 0

        ori_id = _id if args.expand_ids else _id.split('_')[0]
        supporting_titles = gts[ori_id]
        if len(supporting_titles) != 0 and isinstance(list(supporting_titles)[0], tuple):
            tuple_supporting = True
            supporting_titles0 = set(x[0] for x in supporting_titles)
        else:
            tuple_supporting = False
            supporting_titles0 = supporting_titles

        context = datum['context']

        context_split = context.split("<t> ")

        if original_question_file is not None:
            answer = question_data[_id.split('_')[0]]['answer'] if 'answer' in question_data[_id.split('_')[0]] else question_data[_id.split('_')[0]]['answers'][0]['text']
        else:
            answer = datum['answer'] if 'answer' in datum else datum['answers'][0]['text']
        if isinstance(answer, list):
            answer = answer[0]

        if len(context_split) == 1:
            retrieved_context = datum.get('retrieved_context', [])
            is_impossible = not sufficient_context(datum, supporting_titles) #len(retrieved_context) == 0 or retrieved_context[-1]['title'] not in gts[_id.split('_')[0]]
            new_context = ' [SEP] '.join([f"{x['title']} [et] {''.join(x['context'])}" for x in retrieved_context])
            answers = answers_from_answer_and_context(answer, new_context, is_impossible)
            out_data.append({
                'title': '',
                'paragraphs': [
                    {
                        'context' : new_context,
                        'query' : label if is_impossible != True else "",
                        'query_short' : label_s if is_impossible != True else "",
                        'prev_query' : hop1_label,
                        'retrieved_context': retrieved_context,
                        'qas' : [
                            {
                                'question' : question,
                                'answers' : answers,
                                'is_impossible': is_impossible,
                                'is_last_non_gt': False,
                                'is_alternative': False,
                                'id' : _id +"_0"
                            }
                        ]
                    }
                ],
            })

            expand_counter[True] += 1
        else:
            retrieved_context = datum.get('retrieved_context', [])
            retrieved_titles = set([x['title'] for x in retrieved_context])
            prev_titles = [r['title'] for r in retrieved_context]
            prev_context = ["".join(r['context']).strip() for r in retrieved_context]
            i_m =0
            for i in range(1, len(context_split)):
                if len(context_split[i-i_m].split('</t>')) != 2:
                    i_m += 1
                    continue
                r_title, r_context = [x.strip() for x in context_split[i-i_m].split('</t>')]
                if r_title in prev_titles and r_context in prev_context:
                    continue

                new_context = ' [SEP] '.join([f"{x['title']} [et] {''.join(x['context'])}" for x in retrieved_context] + [context_split[i-i_m].replace('</t>', '[et]')])
                last_retrieved = datum['last_retrieved'][i-1-i_m]
                is_impossible = not sufficient_context(datum, supporting_titles, last_retrieved=last_retrieved)
                is_last_non_gt = (last_retrieved['title'] not in supporting_titles) if not tuple_supporting else ((last_retrieved['title'], ''.join(last_retrieved['text'])) not in supporting_titles)
                is_alternative = any(has_connection(last_retrieved, gt_title, True) for gt_title in supporting_titles0 if gt_title not in retrieved_titles) and is_last_non_gt

                expand = True
                if not args.keep_all:
                    if tuple_supporting:
                        on_gold_path = len(set((x['title'], ''.join(x['context'])) for x in retrieved_context).difference(set(supporting_titles))) == 0
                    else:
                        on_gold_path = len(set(retrieved_titles).difference(set(supporting_titles))) == 0
                    if not is_impossible:
                        # we already have all the gold documents
                        expand = False
                    elif on_gold_path and not is_last_non_gt:
                        # on the gold reasoning path
                        pass
                    elif is_last_non_gt and is_alternative:
                        # if the last paragraph has a connection to one of the gold paragraphs
                        if random.random() < .95 or (ranking_score_file is not None and _id not in ranking_score):
                            expand = False
                    else:
                        if random.random() < .99 or (ranking_score_file is not None and _id not in ranking_score):
                            expand = False

                expand_counter[expand] += 1
                assert last_retrieved['title'] == r_title

                answers = answers_from_answer_and_context(answer, new_context, is_impossible)

                out_data.append({
                    'title': '',
                    'paragraphs': [
                        {
                            'context' : new_context,
                            'expand': expand,
                            'query' : label if is_impossible != True and expand else "",
                            'query_short' : label_s if is_impossible != True and expand else "",
                            'prev_query' : hop1_label,
                            'retrieved_context': retrieved_context + \
                                [{'title': r_title, 'context': last_retrieved['data_object']['text'], 'docid': last_retrieved['docid']}],
                            'qas' : [
                                {
                                    'question' : question,
                                    'answers' : answers,
                                    'is_impossible': is_impossible,
                                    'is_last_non_gt': is_last_non_gt,
                                    'is_alternative': is_alternative,
                                    'id' : _id +"_"+str(i-1)
                                }
                            ]
                        }
                    ],
                })

    print(expand_counter)
    output_json = {'data':out_data}
    write_json_file(output_json, output_filie)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Query ES and merge results with original hotpot data.')
    parser.add_argument('hotpot_file', help='.preds file containing ES queries ')
    parser.add_argument('input_file', help='.preds file containing ES queries ')
    parser.add_argument('output_file', help='.json file containing original questions and ids')
    parser.add_argument('--expand_ids', action='store_true')
    parser.add_argument('--original_question_file', default=None, type=str)
    parser.add_argument('--ranking_score_file', default=None, type=str)
    parser.add_argument('--top_K', default=10000, type=int)
    parser.add_argument('--keep_all', action='store_true')
    args = parser.parse_args()

    main(args.hotpot_file, args.input_file, args.output_file, True, args.original_question_file, args.ranking_score_file, args.top_K)
