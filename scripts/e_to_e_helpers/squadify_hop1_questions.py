"""
Given a list of questions, produces them in SQuAD format for DrQA.

Input file should be in json, a list of objects each of which
must have at least a "question" and an "_id".
"""

from argparse import ArgumentParser
from tqdm import tqdm

from utils.io import write_json_file, load_json_file

def main(question_file, out_file):
    data = load_json_file(question_file)

    rows = []
    for entry in data['data']:
        assert 'question' in entry, 'every entry must have a question'
        assert 'id' in entry, 'every entry must have an id'
        gt_start = -1 if 'gt_start' not in entry else entry['gt_start']
        gt_end = -1 if 'gt_end' not in entry else entry['gt_end']
        
        row = {
            'title': '',
            'paragraphs': [{
                'context': entry['question'],
                'qas': [{
                    'question': entry['question'],
                    'id': entry['id'],
                    'answers': [{'answer_start': 0, 'text': ''}]
                }]
            }]
        }
        rows.append(row)

    write_json_file({'data': rows}, out_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('question_file',
        help="json file containing a list of questions and IDs")
    parser.add_argument('out_file',
        help="File to output SQuAD-formatted questions to")

    args = parser.parse_args()
    main(args.question_file, args.out_file)
