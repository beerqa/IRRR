# FIXME: temporary using SQuAD's eval scripts. HotpotQA using different official scripts.
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import random


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def f1_score_normalized(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def update_answer(metrics, prediction, gold, prefix=None):
    best_f1 = 0.0
    best_em = 0.0
    best_prec = 0.0
    best_recall = 0.0
    for gold_answer in gold:
        em = exact_match_score(prediction, gold_answer)
        f1, prec, recall = f1_score_normalized(prediction, gold_answer)
        if best_f1 < f1:
            best_f1 = f1
            best_em = em
            best_prec = prec
            best_recall = recall

    metrics['em'] += float(best_em)
    metrics['f1'] += best_f1
    metrics['prec'] += best_prec
    metrics['recall'] += best_recall

    if prefix is not None:
        metrics[f'{prefix}_em'] += float(best_em)
        metrics[f'{prefix}_f1'] += best_f1
        metrics[f'{prefix}_prec'] += best_prec
        metrics[f'{prefix}_recall'] += best_recall

    return best_em, best_prec, best_recall


def evaluate(gold_file_path, prediction, sampled=False):
    with open(gold_file_path) as f:
        gold = json.load(f)

    metrics = Counter()
    counts = Counter()
    for dp in gold['data']:
        cur_id = dp['id']
        can_eval_joint = True
        counts[dp['src']] += 1
        if cur_id not in prediction['answer']:
            can_eval_joint = False
            if sampled is False:
                print('missing answer {}'.format(cur_id))
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answers'], prefix=dp['src'])

    if sampled is True:
        N = len(prediction["answer"])
    else:
        N = len(gold['data'])
    for k in ['em', 'f1', 'prec', 'recall']:
        metrics[k] /= N
        for prefix in counts.keys():
            metrics[f'{prefix}_{k}'] /= counts[prefix]
            metrics[f'macro_{k}'] += metrics[f'{prefix}_{k}']
        metrics[f'macro_{k}'] /= len(counts.keys())

    return dict(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge hop1 and hop2 results.')
    parser.add_argument('gold_answers_file')
    parser.add_argument('answers')
    args = parser.parse_args()

    with open(args.answers) as f:
        answers = json.load(f)

    if 'answer' in answers:
      answers=answers['answer']
    res = {'answer':answers }

    metrics = evaluate(args.gold_answers_file, res, False)
    print(metrics)

