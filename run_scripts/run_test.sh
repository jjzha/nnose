MODEL="jobbert"     # "roberta" "jobberta"
DATASET="skillspan" # "sayfullina" "green"
TIMESTAMP=$(date +%F_%T)

python3 src/run_inference.py \
  --model_name_or_path "tmp-5e-5/"$DATASET"/$MODEL/"* \
  --train_file data/"$DATASET"/train.json \
  --validation_file data/"$DATASET"/dev.json \
  --text_column_name tokens \
  --label_column_name tags_skill \
  --seed 113412 \
  --write_output "results/run_test_$TIMESTAMP/" \
