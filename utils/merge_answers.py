from argparse import ArgumentParser
from collections import Counter
import json
import os

def main(args):
    final_predictions = {}
    final_path = {}
    hops = {}
    hops_answer = {}
    hops_conf = {}
    hops_conf_all = {}
    hops_title_all = {}
    
    PAN = args.pan
    THRES = args.thres
    
    print ("PAN : " + str(PAN) +  ", THRES : " + str(THRES))
    for hop in range(1, args.max_hops + 1):
        with open(os.path.join(f"{args.input_prefix}{hop}", "answer_predictions.json")) as f:
            pred = json.load(f)
        with open(os.path.join(f"{args.input_prefix}{hop}", "null_odds.json")) as f:
            conf = json.load(f)            
        with open(os.path.join(f"{args.input_prefix}{hop}", "predictions_titles.json")) as f:
            path = json.load(f)
            
        for k in pred.keys():
            if hop == 1:
              hops_answer[k] = []
              hops_conf[k] = []
              hops_conf_all[k] = []
              hops_title_all[k] = []
              
            conf_p = conf[k] + hop*PAN
            if len(hops_conf[k]) == 0 or conf_p < hops_conf[k][-1]:
              final_predictions[k] = pred[k]
              final_path[k] = path[k] 
              hops[k] = hop
              hops_conf[k].append(conf_p)               
            
            hops_title_all[k].append(path[k])
            hops_conf_all[k].append(conf_p)
            hops_answer[k].append(pred[k])
      
    for k in hops_answer.keys():
      for hi, hop in enumerate(hops_answer[k]):
        if hops_conf_all[k][hi] < THRES:
          final_predictions[k] = hop
          hops[k]=hi+1 
          final_path[k] = hops_title_all[k][hi]
          break
    
          
    with open(os.path.join(args.output_prefix, "answer_predictions.json"), 'w') as f:
        json.dump({'answer': final_predictions}, f)

    with open(os.path.join(args.output_prefix, "answer_predictions_titles.json"), 'w') as f:
        json.dump(final_path, f)
        
    print(Counter(hops.values()))
    print(sum(hops.values()) / len(hops))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('input_prefix', type=str, help="Prefix for hop output directories")
    parser.add_argument('max_hops', type=int)
    parser.add_argument('output_prefix', type=str, help="Output dir")

    parser.add_argument('--pan', type=float, default=4.9)
    parser.add_argument('--thres', type=float, default=4.9)
    
    args = parser.parse_args()

    main(args)
