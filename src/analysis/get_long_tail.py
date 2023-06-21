#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time:        14/04/2022 09.18
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def plot(results=None):
    skillspan = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.5719498910675381,
            0.7709750566893424,
            0.8365079365079366,
            0.5714285714285715,
        ],
        "count": [75, 15, 6, 1],
    }
    skillspan_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.5850451291627763,
            0.7709750566893424,
            0.8365079365079366,
            0.7499999999999999,
        ],
        "count": [75, 15, 6, 1],
    }
    #
    sayfullina = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.6967091295116773, 0.9404987297373295, 0.9013975155279503, 0.65],
        "count": [158, 190, 41, 0],
    }
    sayfullina_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.6967091295116773, 0.942124739915189, 0.9013975155279503, 0.65],
        "count": [158, 190, 41, 0],
    }
    #
    green = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.3914901523597176,
            0.508848857235954,
            0.5774936061381074,
            0.5536603493125232,
        ],
        "count": [133, 27, 17, 13],
    }
    green_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.40152727647346675,
            0.5122549019607844,
            0.5762796027501911,
            0.5647714604236344,
        ],
        "count": [133, 27, 17, 13],
    }

    fig, ax = plt.subplots(figsize=(9, 3), ncols=3, nrows=1)
    for i in range(len(ax)):
        ax[i].grid(
            visible=True, axis="both", which="major", linestyle=":", color="grey"
        )

    width = 0.15
    labels = skillspan["i"]
    x = np.arange(len(labels))

    colors = ["lightsalmon", "mediumturquoise"]
    edgecolors = ["orangered", "teal"]
    hatches = ["//", "\\\\"]

    ax[0].bar(
        x - width,
        skillspan["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[0].bar(
        x + width,
        skillspan_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[1].bar(
        x - width,
        sayfullina["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[1].bar(
        x + width,
        sayfullina_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[2].bar(
        x - width,
        green["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[2].bar(
        x + width,
        green_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )
    # ax.errorbar(
    #         x - width / 2, en_dev["f1_macro"], yerr=en_dev["std"],
    #         color="black", fmt="_", alpha=1., linestyle='', linewidth=1,
    #         solid_capstyle="projecting", capsize=3.5, capthick=1
    #         )
    for i, v in enumerate(skillspan_knn["f1"]):
        ax[0].text(i, v, skillspan_knn["count"][i], ha="center")
    for i, v in enumerate(sayfullina_knn["f1"]):
        ax[1].text(i, v, sayfullina_knn["count"][i], ha="center")
    for i, v in enumerate(green_knn["f1"]):
        ax[2].text(i, v, green_knn["count"][i], ha="center")

    for i in range(len(ax)):
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels, alpha=0.6, fontsize=8)
        ax[i].set_ylabel("Span-F1", alpha=0.6)
        ax[i].legend(["¬KNN", "+KNN"], fontsize=8)
    # ax[0].legend(labels=perf_labels, prop={'size': 9})
    ax[0].set_ylim(bottom=0.5)
    ax[1].set_ylim(bottom=0.65)
    ax[2].set_ylim(bottom=0.35)
    ax[0].set_title("SkillSpan Dev.", alpha=0.6, fontsize=10)
    ax[1].set_title("Sayfullina Dev.", alpha=0.6, fontsize=10)
    ax[2].set_title("Green Dev.", alpha=0.6, fontsize=10)

    fig.tight_layout()
    # plt.show()
    plt.savefig("plots/longtail_all.pdf", dpi=300, bbox_inches="tight")


def plot_cross(results=None):
    sayfullina_skillspan = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.16210539546452857, 0.3557692307692308, 0.5666666666666667, 0.5],
        "count": [75, 12, 2, 1],
    }
    sayfullina_skillspan_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.32054945054945055, 0.7156462585034014, 0.6851851851851851, 0.5],
        "count": [75, 15, 5, 1],
    }
    green_skillspan = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.28914935953239645,
            0.6676328502415458,
            0.5097402597402597,
            0.8571428571428571,
        ],
        "count": [75, 15, 6, 1],
    }
    green_skillspan_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.37407268170426067,
            0.8096969696969697,
            0.6349206349206349,
            0.8571428571428571,
        ],
        "count": [75, 15, 5, 1],
    }
    #
    skillspan_sayfullina = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.33663304402220656, 0.4056281771968046, 0.17716036772216548, 0],
        "count": [158, 190, 40, 0],
    }
    skillspan_sayfullina_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.4580676754729126, 0.554589963141056, 0.41690821256038646, 0],
        "count": [158, 190, 37, 0],
    }
    green_sayfullina = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.3062761138232836, 0.33271215730232123, 0.23581232492997198, 0],
        "count": [158, 190, 40, 0],
    }
    green_sayfullina_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.48146328403307326, 0.556154066261982, 0.3394141666982718, 0],
        "count": [158, 190, 40, 0],
    }
    #
    skillspan_green = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.3290946949978704, 0.5164750957854406, 0.4375, 0.5101540616246498],
        "count": [133, 27, 17, 11],
    }
    skillspan_green_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.34890943575154104,
            0.5693627450980392,
            0.487671384343211,
            0.5555555555555556,
        ],
        "count": [133, 27, 17, 13],
    }
    sayfullina_green = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [0.10861287834972046, 0.1956075373619233, 0, 0.28571428571428575],
        "count": [133, 27, 0, 3],
    }
    sayfullina_green_knn = {
        "i": ["low", "mid-low", "mid-high", "high"],
        "f1": [
            0.21210946676062956,
            0.3378288378288379,
            0.4621396147711937,
            0.4184873949579832,
        ],
        "count": [133, 27, 17, 12],
    }

    fig, ax = plt.subplots(figsize=(9, 6), ncols=3, nrows=2)
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].grid(
                visible=True, axis="both", which="major", linestyle=":", color="grey"
            )

    width = 0.15
    labels = sayfullina_skillspan["i"]
    x = np.arange(len(labels))

    colors = ["lightsalmon", "mediumturquoise"]
    edgecolors = ["orangered", "teal"]
    hatches = ["//", "\\\\"]

    ax[0][0].bar(
        x - width,
        sayfullina_skillspan["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[0][0].bar(
        x + width,
        sayfullina_skillspan_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[1][0].bar(
        x - width,
        green_skillspan["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[1][0].bar(
        x + width,
        green_skillspan_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[0][1].bar(
        x - width,
        skillspan_sayfullina["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[0][1].bar(
        x + width,
        skillspan_sayfullina_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[1][1].bar(
        x - width,
        green_sayfullina["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[1][1].bar(
        x + width,
        green_sayfullina_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[0][2].bar(
        x - width,
        skillspan_green["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[0][2].bar(
        x + width,
        skillspan_green_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    ax[1][2].bar(
        x - width,
        sayfullina_green["f1"],
        label="dev",
        width=0.3,
        color=colors[0],
        hatch=hatches[0],
        edgecolor=edgecolors[0],
        linewidth=0.5,
    )
    ax[1][2].bar(
        x + width,
        sayfullina_green_knn["f1"],
        label="dev",
        width=0.3,
        color=colors[1],
        hatch=hatches[1],
        edgecolor=edgecolors[1],
        linewidth=0.5,
    )

    for i, v in enumerate(sayfullina_skillspan["f1"]):
        ax[0][0].text(
            i - width, v, sayfullina_skillspan["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(sayfullina_skillspan_knn["f1"]):
        ax[0][0].text(
            i + width, v, sayfullina_skillspan_knn["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(green_skillspan["f1"]):
        ax[1][0].text(
            i - width, v, green_skillspan["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(green_skillspan_knn["f1"]):
        ax[1][0].text(
            i + width, v, green_skillspan_knn["count"][i], ha="center", fontsize=8
        )

    for i, v in enumerate(skillspan_sayfullina["f1"]):
        ax[0][1].text(
            i - width, v, skillspan_sayfullina["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(skillspan_sayfullina_knn["f1"]):
        ax[0][1].text(
            i + width, v, skillspan_sayfullina_knn["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(green_sayfullina["f1"]):
        ax[1][1].text(
            i - width, v, green_sayfullina["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(green_sayfullina_knn["f1"]):
        ax[1][1].text(
            i + width, v, green_sayfullina_knn["count"][i], ha="center", fontsize=8
        )

    for i, v in enumerate(skillspan_green["f1"]):
        ax[0][2].text(
            i - width, v, skillspan_green["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(skillspan_green_knn["f1"]):
        ax[0][2].text(
            i + width, v, skillspan_green_knn["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(sayfullina_green["f1"]):
        ax[1][2].text(
            i - width, v, sayfullina_green["count"][i], ha="center", fontsize=8
        )
    for i, v in enumerate(sayfullina_green_knn["f1"]):
        ax[1][2].text(
            i + width, v, sayfullina_green_knn["count"][i], ha="center", fontsize=8
        )

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].set_xticks(x)
            ax[i][j].set_xticklabels(labels, alpha=0.6, fontsize=8)
            ax[i][j].set_ylabel("Span-F1", alpha=0.6)
            ax[i][j].legend(["¬KNN", "+KNN"], fontsize=8)

    ax[0][0].set_title("Sayfullina → SkillSpan", alpha=0.6, fontsize=10)
    ax[1][0].set_title("Green → SkillSpan", alpha=0.6, fontsize=10)

    ax[0][1].set_title("SkillSpan → Sayfullina", alpha=0.6, fontsize=10)
    ax[1][1].set_title("Green → Sayfullina", alpha=0.6, fontsize=10)

    ax[0][2].set_title("SkillSpan → Green", alpha=0.6, fontsize=10)
    ax[1][2].set_title("Sayfullina → Green", alpha=0.6, fontsize=10)

    fig.tight_layout()
    plt.savefig("plots/crossdata.pdf", dpi=300, bbox_inches="tight")


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


def read_conll_train(path):
    gold_tags = []
    tokens = []
    # current_tags = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split()
            if len(line) > 2:
                token, gold, _ = line
            else:
                token, gold = line

            tokens.append(token)
            gold_tags.append(gold[0])

    return tokens, gold_tags


def to_spans(tokens, tags):
    spans = []
    for beg in range(len(tags)):
        if tags[beg][0] == "B":
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[end][0] != "I":
                    break
            # spans.append(str(beg) + "-" + str(end) + ":" + tags[beg][2:] + ":" + " ".join(tokens[beg:end]))
            spans.append(str(beg) + "-" + str(end) + ":" + " ".join(tokens[beg:end]))

    return spans


def to_spans_train(tokens, tags):
    spans = []
    for beg in range(len(tags)):
        if tags[beg][0] == "B":
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[end][0] != "I":
                    break
            spans.append(" ".join(tokens[beg:end]))

    return spans


def getBegEnd(span):
    return [int(x) for x in span.split(":")[0].split("-")]


def calculate_f1(tokens, pred, gold):
    tp = 0
    fp = 0
    fn = 0

    gold_spans = to_spans(tokens, gold)
    pred_spans = to_spans(tokens, pred)
    overlap_nums = len(set(gold_spans).intersection(set(pred_spans)))

    tp += overlap_nums
    fp += len(pred_spans) - overlap_nums
    fn += len(gold_spans) - overlap_nums

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)

    return prec, rec, f1


def generate_data_plot():
    args = parse_args()
    train_path = args.train_dir

    train_tokens, train_gold = read_conll_train(train_path)
    train_spans = to_spans_train(train_tokens, train_gold)
    skill_count_train = Counter(train_spans)

    pred_paths = [args.prediction_dir]
    result_dict = {"i": [], "f1": [], "count": []}
    final_dict = {"i": ["low", "mid-low", "mid-high", "high"], "f1": [], "count": []}
    low_counter = 3
    mid_counter = 6
    mid_high_counter = 10
    high_counter = 15

    for pred_path in pred_paths:
        tokens, pred_tags, gold_tags = read_conll(pred_path)
        print(calculate_f1(tokens, pred_tags, gold_tags))

        intervals = range(0, 16, 1)

        for i in intervals:
            tp = 0
            fp = 0
            fn = 0
            total_sum = 0
            current_support = []
            current_preds = []
            gold_spans = to_spans(tokens, gold_tags)
            pred_spans = to_spans(tokens, pred_tags)

            gold_span_count = []
            pred_span_count = []

            for gold_span in gold_spans:
                current_support.append(gold_span.split(":")[-1])
            skill_count = Counter(current_support)
            current_skill_count = {
                k: skill_count_train.get(k)
                for k, _ in skill_count.items()
                if skill_count_train.get(k) and skill_count_train.get(k) == i
            }
            total_sum += len(current_skill_count.keys())

            for pred_span in pred_spans:
                current_preds.append(pred_span.split(":")[-1])
            pred_count = Counter(current_preds)
            current_pred_count = {
                k: skill_count_train.get(k)
                for k, _ in pred_count.items()
                if skill_count_train.get(k) and skill_count_train.get(k) == i
            }

            for gold_span in gold_spans:
                if current_skill_count.get(gold_span.split(":")[-1]):
                    gold_span_count.append(gold_span)

            for pred_span in pred_spans:
                if current_pred_count.get(pred_span.split(":")[-1]):
                    pred_span_count.append(pred_span)

            overlap_nums = len(set(gold_span_count).intersection(set(pred_span_count)))

            tp += overlap_nums
            fp += len(pred_span_count) - overlap_nums
            fn += len(gold_span_count) - overlap_nums

            prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
            rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)


            result_dict["f1"].append(f1)
            result_dict["count"].append(total_sum)

            print(f"{total_sum} skill(s) appear {i}x, f1: {f1}")

    print(result_dict)

    avg_low_f1, avg_low_count = [], []
    avg_midlow_f1, avg_midlow_count = [], []
    avg_midhigh_f1, avg_midhigh_count = [], []
    avg_high_f1, avg_high_count = [], []
    for idx, (f1, count) in enumerate(zip(result_dict["f1"], result_dict["count"])):
        if idx <= low_counter:
            avg_low_f1.append(f1)
            avg_low_count.append(count)
        elif low_counter < idx <= mid_counter and (count != 0 and f1 != 0.0):
            avg_midlow_f1.append(f1)
            avg_midlow_count.append(count)
        elif mid_counter < idx <= mid_high_counter and (count != 0 and f1 != 0.0):
            avg_midhigh_f1.append(f1)
            avg_midhigh_count.append(count)
        elif mid_high_counter < idx <= high_counter and (count != 0 and f1 != 0.0):
            avg_high_f1.append(f1)
            avg_high_count.append(count)

    print(avg_low_f1)
    print(avg_midlow_f1)
    print(avg_midhigh_f1)
    print(avg_high_f1)

    final_dict["f1"].append(np.mean(avg_low_f1))
    final_dict["f1"].append(np.mean(avg_midlow_f1))
    final_dict["f1"].append(np.mean(avg_midhigh_f1))
    final_dict["f1"].append(np.mean(avg_high_f1))

    final_dict["count"].append(sum(avg_low_count))
    final_dict["count"].append(sum(avg_midlow_count))
    final_dict["count"].append(sum(avg_midhigh_count))
    final_dict["count"].append(sum(avg_high_count))

    print(final_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create plot for skill distribution performance."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="the directory of the gold training data.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="the directory of the generated predictions.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    generate_data_plot()
    plot_cross()
    plot()
