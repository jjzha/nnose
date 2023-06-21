#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time:        14/04/2022 09.18

import argparse

from seqeval.metrics import *


def read_conll(path):
    pred_tags = []
    gold_tags = []
    tokens = []
    # current_tags = []
    for line in open(path):
        line = line.strip()
        if line:
            token, gold, pred = line.split("\t")
            tokens.append(token)
            pred_tags.append(pred)
            gold_tags.append(gold)
    return tokens, pred_tags, gold_tags


def read_conll_seqeval(path):
    pred_tags = []
    gold_tags = []

    current_pred = []
    current_gold = []
    # current_tags = []
    for line in open(path):
        line = line.strip()
        if line:
            _, gold, pred = line.split("\t")
            # tokens.append(token)
            current_pred.append(pred)
            current_gold.append(gold)
        else:
            gold_tags.append(current_gold)
            pred_tags.append(current_pred)
            current_pred = []
            current_gold = []
    return pred_tags, gold_tags


def to_spans(tokens, tags):
    spans = []
    for beg in range(len(tags)):
        if tags[beg][0] == "B":
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[end][0] != "I":
                    break
            spans.append(str(beg) + "-" + str(end) + ":" + " ".join(tokens[beg:end]))

    return spans


def getBegEnd(span):
    return [int(x) for x in span.split(":")[0].split("-")]


def get_false_positives():
    args = parse_args()

    tokens, pred_tags, gold_tags = read_conll(args.prediction_dir)
    tokens_knn, pred_tags_knn, gold_tags_knn = read_conll(args.prediction_dir_knn)
    pred_tags_seqeval, gold_tags_seqeval = read_conll_seqeval(args.prediction_dir)
    pred_tags_knn_seqeval, gold_tags_knn_seqeval = read_conll_seqeval(
        args.prediction_dir_knn
    )

    gold_spans = to_spans(tokens, gold_tags)
    pred_spans = to_spans(tokens, pred_tags)
    pred_spans_knn = to_spans(tokens_knn, pred_tags_knn)
    print(len(pred_spans), len(pred_spans_knn))

    fp_knn_set = []
    fp_vanilla_set = []
    for span in pred_spans_knn:
        if span not in gold_spans:
            fp_knn_set.append(span)
    for span in pred_spans:
        if span not in gold_spans:
            fp_vanilla_set.append(span)

    fn_knn_set = []
    fn_vanilla_set = []
    for span in gold_spans:
        if span not in pred_spans_knn:
            fn_knn_set.append(span)
        if span not in pred_spans:
            fn_vanilla_set.append(span)

    print("SeqEval Implementation Huggingface")
    print(
        precision_score(gold_tags_seqeval, pred_tags_seqeval),
        recall_score(gold_tags_seqeval, pred_tags_seqeval),
        f1_score(gold_tags_seqeval, pred_tags_seqeval),
    )
    print(
        precision_score(gold_tags_seqeval, pred_tags_knn_seqeval),
        recall_score(gold_tags_seqeval, pred_tags_knn_seqeval),
        f1_score(gold_tags_seqeval, pred_tags_knn_seqeval),
    )

    # fn
    print(len(fn_vanilla_set))
    print(len(fn_knn_set))

    # fp
    print(len(fp_vanilla_set))
    print(len(fp_knn_set))


def parse_args():
    parser = argparse.ArgumentParser(description="Investigate false positives.")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="the directory of the generated predictions.",
    )
    parser.add_argument(
        "--prediction_dir_knn",
        type=str,
        required=True,
        help="the directory of the generated predictions of the knn.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    get_false_positives()
