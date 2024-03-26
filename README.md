# Repository for NNOSE

This repository accompanies the paper:

**NNOSE: Nearest Neighbor Occupational Skill Extraction**

Mike Zhang, Rob van der Goot, Min-Yen Kan, and Barbara Plank. In EACL 2024.

<image src="img/figure1.png"></image>

# Getting Started

## Requirements

Clone the repository. If you use `conda`, please install the accompanying environment by:

```
# create the environment
conda env create -f environment.yml

# activate environment
conda activate nnose

# install torch separately
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

There is a separate environment for generating the UMAP plot.

```
# create the environment
conda env create -f environment_umap.yml

# activate environment
conda activate nnose_umap

# install torch separately
pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

! The UMAP plot can only be created once we obtained the embeddings using `run_scripts/get_representations.sh`.

All experiments are ran on `python=3.9` and `torch==1.10.1`.

## Getting JobBERTa

JobBERTa will be released when the paper is accepted. You can check how JobBERTa is trained using: 
`run_scripts/run_mlm.sh`.

The MLM script is derived from HuggingFace and can be found in `src/utils/run_mlm.py`.

# Running Experiments

‼️ It is extremely important that the experiments are ran in the right order.

## 1. Training the Language Models

To fine-tune the models used in the paper, run the following script:

```
bash run_scripts/run_trainer.sh
```

## 2. Obtaining Embeddings + Creating the Datastore

We have put the extraction of embeddings from the training datasets and training the datastore in one file.
We have two types of datastores in our experiments, an in-dataset datastore ({D}) and an 'all' datastore ($\forall$ D).

To create the in-dataset datastore:
```
bash run_scripts/get_representations_dataset.sh
```

To create the \forall datastore:
```
bash run_scripts/get_representations.sh
```

## 3. Sweeping Hyperparameters

To do a hyperparameter sweep for NNOSE to get the best working k neighbors, lambda, and temperature run:
```
bash run_scripts/run_inference_sweep.sh
```

## 4. Running Inference

We have two scripts to run test/inference, which also outputs the predictions in a separate output file.
To do this, please _look_ at the following scripts:

```
run_scripts/run_test.sh
```
or to do test/inference with NNOSE:
```
run_scripts/run_test_index.sh
```

## 5. Doing the Analysis

### Skill Distribution (Figure 8) + Jaccard Overlap
This analysis doesn't need any already ran experiments, to do this run:

```
python3 src/analysis/skill_distribution.py
```

### Long Tail Analysis (Figure 2 and Figure 4)
This analysis _requires_ the output of the models from step (4) above:

Example
```
python3 src/analysis/get_long_tail.py \
    --train_dir data/skillspan/train.json \
    --prediction_dir results/<name_of_file> \
```

### False Positive Analysis (Table 11 + Table 12)
This analysis also _requires_ the output of the models from step (4) above:

```
python3 src/analysis/skill_distribution.py \
    --prediction_dir results/<name_of_file_predictions> \
    --prediction_dir_knn results/<name_of_file_predictions_knn> \
```

### Cross-dataset Analysis (Table 3)
This analysis requires you to have trained the models. Note that there is also an "_all_" model, 
which is the concatenation of all datasets.

This one is done manually with the `run_scripts/run_test.sh` script:

Example
```
MODEL="jobbert"     # "roberta" "jobberta"
DATASET="skillspan" # "sayfullina" "green"
TIMESTAMP=$(date +%F_%T)

python3 src/run_inference.py \
  --model_name_or_path "tmp-5e-5/"$DATASET"/$MODEL/"* \
  --train_file data/"$DATASET"/train.json \
  --validation_file data/sayfullina/dev.json \
  --text_column_name tokens \
  --label_column_name tags_skill \
  --seed 113412 \
  --write_output "results/run_test_$TIMESTAMP/" \

```

You can change the `--validation_file` flag to the dataset you want to apply it on. In this case, we use `sayfullina`.

### UMAP plot (Figure 3)
Please change to the UMAP environment as stated in the `Getting Started` section. The only difference is the pytorch version.

This analysis further requires you to have ran step (2): Obtaining Embeddings + Creating the Datastore.

Example:
```
python3 src/analysis/plot_umap.py --output_dir plots/
```
**WARNING**: This script takes a lot of time if run for the first time (around 45-60 minutes on a good machine).

# Questions

If there's any questions, please reach out to \<email\>.

# Citation

If you have been using this work in your cool work, consider citing it:

```
@inproceedings{zhang-etal-2024-nnose,
    title = "{NNOSE}: Nearest Neighbor Occupational Skill Extraction",
    author = "Zhang, Mike  and
      Goot, Rob  and
      Kan, Min-Yen  and
      Plank, Barbara",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.35",
    pages = "589--608",
    abstract = "The labor market is changing rapidly, prompting increased interest in the automatic extraction of occupational skills from text. With the advent of English benchmark job description datasets, there is a need for systems that handle their diversity well. We tackle the complexity in occupational skill datasets tasks{---}combining and leveraging multiple datasets for skill extraction, to identify rarely observed skills within a dataset, and overcoming the scarcity of skills across datasets. In particular, we investigate the retrieval-augmentation of language models, employing an external datastore for retrieving similar skills in a dataset-unifying manner. Our proposed method, \textbf{N}earest \textbf{N}eighbor \textbf{O}ccupational \textbf{S}kill \textbf{E}xtraction (NNOSE) effectively leverages multiple datasets by retrieving neighboring skills from other datasets in the datastore. This improves skill extraction \textit{without} additional fine-tuning. Crucially, we observe a performance gain in predicting infrequent patterns, with substantial gains of up to 30{\%} span-F1 in cross-dataset settings.",
}

```
