#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-07-05 14:56:27.996067960 +0530
# Modify: 2022-07-27 19:18:02.467252970 +0530

"""Compute graded relevance"""

import argparse
import json
import math
import os
from collections import defaultdict

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def create_adv_scores(charges: list, charge_adv_win_ratios: dict,
                      weights: dict = None, strategy: str = 'equal',):
    """Generate scores of advocates for a given combination of charges
    depending on the weightage strategy

    Parameters
    ----------
    charges: list
        List of charges.
    charge_adv_win_ratios: dict
        Dictionary of win ratios of each advocate for each charge.
    weight: dict, default None
        Weights to be used for combining win-ratios for resultant score.
        Used when 'strategy' is not 'equal'.
    strategy: str
        Weightage strategy to use.
        'equal': Equal weightage to each charge.
        'case_fraction': Weigh charges based on fraction of cases.

    Returns
    -------
    scores: dict
        Scores of advocates.
    """
    assert strategy in ['equal', 'case_fraction'], "Invalid strategy"
    if strategy == 'case_fraction':
        assert weights is not None, ("Weights need to be specified when "
                                     "using 'case_fraction' weightage.")
    scores = defaultdict(float)
    if strategy == "equal":
        weights = {charge: 1./len(charges) for charge in charges}
        for charge in charges:
            for adv, score in charge_adv_win_ratios[charge].items():
                scores[adv] += weights[charge] * score
    else:
        print(weights)
        for charge in charges:
            for adv, score in charge_adv_win_ratios[charge].items():
                scores[adv] += weights[charge] * score

    scores = {k: v for k, v in sorted(
                            scores.items(), key=lambda x: x[1], reverse=True)}
    return scores


def create_targets(targets_dict, adv_index, case,
                   case_charges=None, adv_charges=None, threshold=None):
    """Create targets from a dictionary of targets and advocate ordering.

    Parameters
    ----------

    targets_dict : dict
        Dictionary with the targets of each case.
    adv_list : list
        List of advocates to consider.
    cases : list
        Ordered list cases.

    Returns
    -------
    result : numpy.array
        Stacked target mult-hot vectors.
    """
    actual = []
    lenient = []

    # Lenient
    if all(ele is not None
           for ele in [case_charges, adv_charges, threshold]):
        lenient = [adv
                   for adv in list(adv_index.keys())
                   if adv not in targets_dict[case] and
                   len(set(adv_charges[adv]).intersection(
                                set(case_charges[case]))) * 1./len(
                                case_charges[case]) >= threshold]

    actual = [adv
              for adv in list(adv_index.keys())
              if adv in targets_dict[case]]

    return actual, lenient


def dcg(relevance, predicted):
    score = sum([relevance[name] * 1./math.log(i+2, 2)
                 for i, name in enumerate(predicted)])
    return score


def idcg(relevance, predicted):
    relevance = {k: v for k, v in sorted(
                                    relevance.items(), key=lambda x: x[1],
                                    reverse=True)[:len(predicted)]}
    score = sum([val * 1./math.log(i+2, 2)
                for i, val in enumerate(relevance.values())])
    return score


#  def relevance_rank(actual, lenient, full):
    #  relevance = {}
    #  for adv in full:
        #  if adv in actual:
            #  relevance[adv] = 3 if lenient is not [] else 1
        #  elif adv in lenient[:int(len(lenient)/2) + 1]:
            #  relevance[adv] = 2
        #  elif adv in lenient[int(len(lenient)/2) + 1:]:
            #  relevance[adv] = 1
        #  else:
            #  relevance[adv] = 0
    #  relevance = {k: v for k, v in sorted(
                                    #  relevance.items(),
                                    #  key=lambda x: x[1], reverse=True)}
    #  return relevance


def integer_relevance_rank(actual, lenient, full, relevance_scores):

    advs = [*actual, *lenient]
    num = len(advs)
    relevance = {k: v for k, v in zip([a for a in sorted(
                                    advs, key=lambda x: relevance_scores[x],
                                    reverse=True)], range(num, 0, -1))}
    relevance.update({k: 0.0 for k in full if k not in advs})
    print(relevance)

    return relevance


def cont_relevance_rank(actual, lenient, full, relevance_scores):
    relevance = {}
    print(relevance_scores)
    for adv in full:
        if adv in actual or adv in lenient:
            relevance[adv] = relevance_scores[adv]
        else:
            relevance[adv] = 0
    return relevance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--charge_adv_win_ratios",
                        help="Win ratios of advocates for each charge.")
    parser.add_argument("--case_charges",
                        help="Charges/offences of each case.")
    parser.add_argument("--charge_cases",
                        help="Cases of each charge.")
    parser.add_argument("--advocate_charges",
                        help="Charges of each advocate.")
    parser.add_argument("--charge_targets",
                        help="Target charges to consider.")
    parser.add_argument("--targets",
                        help="Target advocates of each case.")
    parser.add_argument("--relevant_advocates",
                        help="Advocates to consider.")
    parser.add_argument("--relevant_cases", type=str, default=None,
                        help="Cases to consider in evaluation.")
    parser.add_argument("--scores",
                        help="Ranked scores of advocates.")
    parser.add_argument("--strategy", type=str, default='equal',
                        help="Score combination strategy.")
    parser.add_argument("--output_path",
                        help="Path to save generated ndcg scores.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for lenient targets.")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Charge advocate win ratios
    with open(args.charge_adv_win_ratios, 'r') as f:
        charge_adv_win_ratios = json.load(f)

    # Case offences
    with open(args.case_charges, 'r') as f:
        case_charges = json.load(f)

    # Charge cases
    with open(args.charge_cases, 'r') as f:
        charge_cases = json.load(f)

    # Advocate charges
    with open(args.advocate_charges, 'r') as f:
        adv_charges = json.load(f)

    # Charge targets
    with open(args.charge_targets, 'r') as f:
        charge_targets = f.readlines()
    charge_targets = list(filter(None, map(lambda x: x.strip("\n"),
                                           charge_targets)))

    # Case advocates
    with open(args.targets, 'r') as f:
        case_advs = json.load(f)

    # Relevant advocates
    with open(args.relevant_advocates, 'r') as f:
        advs = json.load(f)
    advs = {k: i for i, k in advs.items()}

    # Relevant cases
    if args.relevant_cases is not None:
        with open(args.relevant_cases, 'r') as f:
            rel_cases = f.readlines()
        rel_cases = list(filter(None, map(lambda x: x.strip("\n"),
                                          rel_cases)))
    else:
        rel_cases = None

    # Scores
    with open(args.scores, 'r') as f:
        scores = json.load(f)

    # Getting scores of charges
    result_scores = {}
    ndcg = {}

    if args.strategy == 'equal':
        weights = None
    else:
        total_cases = set([value
                           for values in charge_cases.values()
                           for value in values])

        weights = {}
        for charge, cases in charge_cases.items():
            weights[charge] = len(cases) * 1./len(total_cases)

        print(weights)

    for case, ranks in scores.items():
        if rel_cases is not None:
            if case not in rel_cases:
                continue

        if case_advs.get(case, -1) == -1:
            print(f"{case} not found.")
            continue

        if set(case_advs[case]).intersection(set(advs)) == set():
            print(f"{case} not found.")
            continue

        rel_charges = set(case_charges[case]).intersection(set(charge_targets))

        if rel_charges == set():
            continue

        result_scores[case] = create_adv_scores(rel_charges,
                                                charge_adv_win_ratios,
                                                strategy=args.strategy,
                                                weights=weights)
        actual, lenient = create_targets(targets_dict=case_advs,
                                         adv_index=advs,
                                         case_charges=case_charges,
                                         adv_charges=adv_charges,
                                         threshold=args.threshold,
                                         case=case)

        # Continuous Scale Relevance Ranking
        #  relevance = cont_relevance_rank(actual=actual,
                                        #  lenient=lenient,
                                        #  full=advs,
                                        #  relevance_scores=result_scores[case])

        relevance = integer_relevance_rank(
                                    actual=actual,
                                    lenient=[],
                                    full=advs,
                                    relevance_scores=result_scores[case])
        predicted = list(ranks.keys())
        print(predicted)
        ndcg_score = dcg(relevance, predicted) * 1./idcg(relevance, predicted)
        ndcg[case] = ndcg_score

    print(len(ndcg.keys()))

    avg = sum(ndcg.values()) * 1./len(ndcg.keys())

    # Saving
    with open(os.path.join(args.output_path, "ndcg.json"), 'w') as f:
        json.dump(ndcg, f, indent=4)

    with open(os.path.join(args.output_path, "avg_ndcg.txt"), 'w') as f:
        print(avg, file=f)

    with open(os.path.join(args.output_path, "result_scores.json"), 'w') as f:
        json.dump(result_scores, f, indent=4)


if __name__ == "__main__":
    main()
