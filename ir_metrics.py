#!/usr/bin/env python

import argparse
import json
import logging
import os

import pytrec_eval

from utils import set_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scores",
                        help="Query scores")
    parser.add_argument("-t", "--targets",
                        help = "Targets")
    parser.add_argument("-l", "--log_path", type=str, default=None,
                        help="Path to save generated logs")
    parser.add_argument("-o", "--output_dir",
                        help="Path to save generated metrics")
    parser.add_argument("-n", "--name", type=str,
                        help="Evaluation scheme name")

    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, args.name)
    if args.log_path is None:
        args.log_path = output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_logger(os.path.join(args.log_path, "ir_metrics"))
    logging.info("Inputs:")
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")
    
    with open(args.scores, 'r') as f:
        scores = json.load(f)

    with open(args.targets, 'r') as f:
        targets = json.load(f)

    targets = {
            k: {adv: 1 for adv in names}
            for k, names in targets.items() if k in scores.keys()}

    evaluator = pytrec_eval.RelevanceEvaluator(
            targets, {'map', 'Rprec'})

    results = evaluator.evaluate(scores)

    per_query_ap = {}
    per_query_rprec = {}

    for query, values in results.items():
        logging.info(f"Getting scores for {query}")
        per_query_ap[query] = values['map']
        per_query_rprec[query] = values['Rprec']

    map = sum(per_query_ap.values()) * 1./len(per_query_ap)
    macro_rprec = sum(per_query_rprec.values()) * 1./len(per_query_rprec)

    with open(os.path.join(output_dir, "mAP.txt"), 'w') as f:
        f.write(str(map))

    with open(os.path.join(output_dir, "macro_rprec.txt"), 'w') as f:
        f.write(str(macro_rprec))

    with open(os.path.join(output_dir, "per_query_ap.json"), 'w') as f:
        json.dump(per_query_ap, f, indent=4)

    with open(os.path.join(output_dir, "per_query_rprec.json"), 'w') as f:
        json.dump(per_query_rprec, f, indent=4)


if __name__ == "__main__":
    main()
