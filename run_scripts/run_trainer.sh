MODEL=("jjzha/jobbert-base-cased") # "roberta-base" "jobberta-base" not released yet.
MODEL_NAME=("jobbert")             # "roberta" "jobberta"
DATASET=("skillspan" "sayfullina" "green" "all")

for model in "${!MODEL[@]}"; do
  for dataset in "${!DATASET[@]}"; do
    echo "Running NER on dataset ${DATASET[$dataset]} with "${MODEL[$model]}"."

    python3 src/run_ner_no_trainer.py \
      --model_name_or_path "${MODEL[$model]}" \
      --train_file data/${DATASET[$dataset]}/train.json \
      --validation_file data/${DATASET[$dataset]}/dev.json \
      --text_column_name tokens \
      --label_column_name tags_skill \
      --output_dir tmp-5e-5/${DATASET[$dataset]}/"${MODEL_NAME[$model]}" \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --learning_rate 5e-5 \
      --checkpointing_steps epoch \
      --num_train_epochs 20 \
      --patience 5 \
      --seed 113412 \

  done
done
