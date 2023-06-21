MODEL=("jobbert") # "roberta" "jobberta"
DATASET=("green" "skillspan" "sayfullina")
PERCENTAGE=(100)

for i in "${!MODEL[@]}"; do
  for dataset in "${!DATASET[@]}"; do
    for p in "${!PERCENTAGE[@]}"; do

      echo "Getting representations for dataset: ${DATASET[$dataset]} with model: ${MODEL[$i]}."

      python3 src/get_representations.py \
        --model_name_or_path "tmp-5e-5/${DATASET[$dataset]}/${MODEL[$i]}/"* \
        --train_file data/"${DATASET[$dataset]}"/train.json \
        --text_column_name tokens \
        --label_column_name tags_skill \
        --max_length 128 \
        --seed 113412 \
        --save_path datastore_"${PERCENTAGE[$p]}"_"${DATASET[$dataset]}"/saved_embedding_"${MODEL[$i]}"

    done

    echo "Creating Datastore for each dataset separately..."

    python3 src/create_datastore.py \
      --feature_dir datastore_"${PERCENTAGE[$p]}"_"${DATASET[$dataset]}"/saved_embedding_"${MODEL[$i]}" \
      --output_dir datastore_"${PERCENTAGE[$p]}"_"${DATASET[$dataset]}"/index_"${MODEL[$i]}"/ \
      --sample_percentage ${PERCENTAGE[$p]} \
      --seed 113412 \
      --whitening \
      --dim_reduction 768 \

  done
done
