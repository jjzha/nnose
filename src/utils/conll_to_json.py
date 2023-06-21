#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    convert_to_conll.py
# @Time:        20/04/2022 11.16
import argparse
import json


def main(args):
    """Function to convert conll to json format."""
    cnt = 0
    json_doc = {"idx": cnt, "tokens": [], "tags_skill": []}
    with open(args.data_dir, "r", encoding="utf-8") as f, open(
        f"{args.data_dir.split('.')[0]}.json", "w"
    ) as out:
        for line in f:
            line = line.split()
            if len(line) > 2:
                token, skill, knowledge = line
                json_doc["idx"] = cnt
                json_doc["tokens"].append(token)
                json_doc["tags_skill"].append(knowledge[0])
            elif line:
                token, skill = line
                json_doc["idx"] = cnt
                json_doc["tokens"].append(token)
                json_doc["tags_skill"].append(skill[0])
            elif json_doc["tokens"]:
                cnt += 1
                out.write(json.dumps(json_doc))
                out.write("\n")
                json_doc = {"idx": cnt, "tokens": [], "tags_skill": []}


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
