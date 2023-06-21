MODEL="jobberta" # "jobbert" "roberta"
DATASET="green"  # "skillspan" "sayfullina"
DATASTORE="AD"   # "skillspan" "sayfullina" "green"
K=32             # any int
LAMBDA=0.9       # any float 0-1
T=0.1            # any float
TIMESTAMP=$(date +%F_%T)

python3 src/run_inference.py \
  --model_name_or_path "tmp-5e-5/"$DATASET"/$MODEL/"* \
  --train_file data/"$DATASET"/train.json \
  --validation_file data/"$DATASET"/dev.json \
  --text_column_name tokens \
  --label_column_name tags_skill \
  --seed 113412 \
  --datastore_path "datastore_*_"$DATASTORE"/index_$MODEL/" \
  --knn \
  --k $K \
  --lambda_value $LAMBDA \
  --temperature $T \
  --projection \
  --write_output "results/run_test_index_$TIMESTAMP/" \
