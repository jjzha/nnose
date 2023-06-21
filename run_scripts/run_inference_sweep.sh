K=(4 8 16 32 64 128)
LAMBDA=("0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.50" "0.55" "0.60" "0.65" "0.70" "0.75" "0.80" "0.85" "0.90")
T=("0.1" "0.5" "1.0" "2.0" "3.0" "5.0" "10.0")
MODEL=("jobbert") # "roberta" "jobberta"
DATASET=("skillspan" "sayfullina" "green")
DATASTORE="AD" # "skillspan" "sayfullina" "green"
TIMESTAMP=$(date +%F_%T)

# iterate over models
for model in "${!MODEL[@]}"; do
  # iterate over datasets
  for dataset in "${!DATASET[@]}"; do
    # iterate over k neighbors
    for k in "${!K[@]}"; do
      # iterate over lambda values
      for l in "${!LAMBDA[@]}"; do
        # iterate over temperature
        for t in "${!T[@]}"; do

          echo "Parameter Sweep with K: '${K[$k]}', Lambda: '${LAMBDA[$l]}', Temperature: '${T[$t]}'."

          OUTPUT_DIR=sweep/"$TIMESTAMP"/"${MODEL[$model]}"_"${DATASET[$dataset]}".out

          python3 src/run_inference.py \
            --model_name_or_path "tmp-5e-5/${DATASET[$dataset]}/${MODEL[$model]}/"* \
            --train_file data/"${DATASET[$dataset]}"/train.json \
            --validation_file data/"${DATASET[$dataset]}"/dev.json \
            --text_column_name tokens \
            --label_column_name tags_skill \
            --max_length 128 \
            --seed 113412 \
            --datastore_path datastore_*_"$DATASTORE"/index_"${MODEL[$model]}" \
            --projection \
            --k "${K[$k]}" \
            --lambda_value "${LAMBDA[$l]}" \
            --temperature "${T[$t]}" \
            --knn \
            --sweep \
            --output_sweep "$OUTPUT_DIR" \

        done
      done
    done
  done
done
