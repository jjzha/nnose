#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    skill_distribution.py
# @Time:        17/03/2023 11.21

import json
from collections import Counter

import matplotlib.pyplot as plt


def plot(skillspan, sayfullina, green):
    fig, ax = plt.subplots(figsize=(9, 3), ncols=3, nrows=1)
    for i in range(len(ax)):
        ax[i].grid(
            visible=True, axis="both", which="major", linestyle=":", color="grey"
        )

    ax[0].bar(list(skillspan.keys()), list(skillspan.values()))
    ax[1].bar(list(sayfullina.keys()), list(sayfullina.values()))
    ax[2].bar(list(green.keys()), list(green.values()))

    ax[0].set_title("Distribution Skills in Skillspan", fontsize=10, alpha=0.6)
    ax[1].set_title("Distribution Skills in Sayfullina", fontsize=10, alpha=0.6)
    ax[2].set_title("Distribution Skills in Green", fontsize=10, alpha=0.6)

    for i in range(len(ax)):
        ax[i].set_ylabel("Frequency", alpha=0.6)
        ax[i].set_xlabel("Skill Count", alpha=0.6)
        ax[i].set_xlim(0, 16)

    fig.tight_layout()
    # Show the plot
    plt.savefig("plots/skill_distribution.pdf", dpi=300, bbox_inches="tight")


def count_skills(path):
    length_list = []
    competences = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)

            len_skill = 0
            current_skill = []
            for token, tag_skill in zip(data["tokens"], data["tags_skill"]):
                if tag_skill in ["B"]:
                    if current_skill and ">" not in current_skill[0]:
                        competences.append(" ".join(current_skill).lower())
                        length_list.append(len_skill)
                        len_skill = 0
                        current_skill = []
                    len_skill += 1
                    current_skill.append(token)
                elif tag_skill in ["I"]:
                    len_skill += 1
                    current_skill.append(token)
                elif tag_skill == "O":
                    if current_skill and ">" not in current_skill[0]:
                        competences.append(" ".join(current_skill).lower())
                        length_list.append(len_skill)
                    len_skill = 0
                    current_skill = []

                if token == data["tokens"][-1]:
                    if current_skill and ">" not in current_skill[0]:
                        competences.append(" ".join(current_skill).lower())
                        length_list.append(len_skill)
                    len_skill = 0
                    current_skill = []

    counts = Counter(length_list)
    return counts


def main():
    path_skillspan = "data/skillspan/train.json"
    path_sayfullina = "data/sayfullina/train.json"
    path_green = "data/green/train.json"

    skillspan = count_skills(path_skillspan)
    sayfullina = count_skills(path_sayfullina)
    green = count_skills(path_green)

    # overlap skillspan + sayfullina
    print("SkillSpan -- Sayfullina overlap")
    print(
        len(set(list(skillspan.keys())) & set(list(sayfullina.keys())))
        / float(len(set(list(skillspan.keys())) | set(list(sayfullina.keys()))))
        * 100
    )

    # overlap sayfullina + green
    print("Sayfullina -- Green overlap")
    print(
        len(set(list(green.keys())) & set(list(sayfullina.keys())))
        / float(len(set(list(green.keys())) | set(list(sayfullina.keys()))))
        * 100
    )

    # overlap green + skillspan
    print("Green -- SkillSpan overlap")
    print(
        len(set(list(green.keys())) & set(list(skillspan.keys())))
        / float(len(set(list(green.keys())) | set(list(skillspan.keys()))))
        * 100
    )

    plot(skillspan, sayfullina, green)


if __name__ == "__main__":
    main()
