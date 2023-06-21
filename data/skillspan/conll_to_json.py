#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    conll_to_json.py
# @Time:        05/04/2023 16.18

import argparse
import json


def main(args):
    """Function to convert conll to json format."""
    cnt = 0
    json_doc = {
        "idx": cnt,
        "tokens": [],
        "tags_skill": [],
        # "tags_knowledge": []
    }
    with open(args.data_dir, "r", encoding="utf-8") as f, open(
        f"{args.data_dir.split('.')[0]}.json", "w"
    ) as out:
        current_doc = None
        for line in f:
            if line.startswith("###"):
                # current_doc = json.loads(line.split("### ")[1])
                cnt += 1
                continue

            line = line.split()
            print(line)
            if line:
                token, skill, knowledge = line
                json_doc["idx"] = cnt
                json_doc["tokens"].append(token)
                if skill[0] in ["B", "I"]:
                    json_doc["tags_skill"].append(skill[0])
                else:
                    json_doc["tags_skill"].append(knowledge[0])
                # json_doc["meta"] = current_doc["meta"]
            elif json_doc["tokens"]:
                out.write(json.dumps(json_doc))
                out.write("\n")
                json_doc = {
                    "idx": cnt,
                    "tokens": [],
                    "tags_skill": [],
                    # "tags_knowledge": [],
                    # "meta": current_doc["meta"]
                }


def parse_args():
    parser = argparse.ArgumentParser(description="Read data to convert conll to json")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="the directory of the data",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
