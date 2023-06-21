#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
)


def main(args):
    """Very specific file that predicts sentences from a json file with tokenized sentences"""
    tokenizer = AutoTokenizer.from_pretrained(
        args.trained_model, use_fast=True, add_prefix_space=True
    )
    model = AutoModelForTokenClassification.from_pretrained(args.trained_model)
    token_skill_classifier = TokenClassificationPipeline(
        model=model, tokenizer=tokenizer, aggregation_strategy="first"
    )

    # load in data
    with open(args.predict_file) as f:
        for obj in f:
            obj = json.loads(obj)
            sentence = " ".join(obj["tokens"])
            output = token_skill_classifier(sentence)

            tokens = obj["tokens"]
            tags = []

            for token in tokens:
                tag = "O"
                for entity in output:
                    if entity["start"] <= sentence.index(token) < entity["end"]:
                        tag = entity["entity_group"]
                tags.append(tag)

                if len(tags) > 1:
                    if tags[-2] == "O" and tag == "I":
                        tags[-1] = "B"

            assert len(tokens) == len(tags)

            with open(args.predict_file[:-5] + "_silver.json", "a") as fout:
                obj_out = {"tokens": tokens, "tags_skill": tags}
                fout.write(f"{obj_out}")
                fout.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict on a file given a trained model."
    )
    parser.add_argument(
        "--predict_file",
        type=str,
        required=True,
        help="the path of the prediction file",
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        required=True,
        help="the path of the trained model",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
