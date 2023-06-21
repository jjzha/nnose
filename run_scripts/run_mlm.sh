# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Running MLM..."

python3 src/utils/run_mlm.py \
  --model_name_or_path roberta-base \
  --train_file data/mlm/en_mlm_train.txt \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --do_train \
  --do_eval \
  --output_dir tmp_mlm/jobberta/ \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --save_steps 10000 \
  --save_total_limit 2 \
  --load_best_model_at_end \
