DATA_PATH="dataset"
CODE_PATH="."
MODEL_PATH="./checkpoint/"

code_path=$CODE_PATH
model_path=meta-llama/Llama-2-7b-hf
dataset_path=$DATA_PATH/ChatTime-1-Pretrain-1M/
log_path=$MODEL_PATH/log_pretrain/
output_path=$MODEL_PATH/ChatTime-1-7B-Base/

lora_rank=8
lora_alpha=16
lora_dropout=0.00
num_train_epochs=2

# DDP 환경에 맞춘 파라미터 조정 (GPU 4대 기준)
per_device_train_batch_size=16
gradient_accumulation_steps=8
max_seq_length=1024

save_steps=200
logging_steps=20
max_steps=-1

# 사용할 GPU 번호 할당 (예: 0, 1, 2, 3번 GPU 사용)
export CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2

# python 대신 torchrun을 사용하여 분산 학습(DDP) 실행
torchrun --nproc_per_node=$NUM_GPUS "$code_path/training/pretrain.py" \
  --code_path "$code_path" \
  --model_path "$model_path" \
  --dataset_path "$dataset_path" \
  --log_path "$log_path" \
  --output_path "$output_path" \
  --max_seq_length $max_seq_length \
  --lora_rank $lora_rank \
  --lora_alpha $lora_alpha \
  --lora_dropout $lora_dropout \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --save_steps $save_steps \
  --logging_steps $logging_steps \
  --max_steps $max_steps \
  --load_in_4bit 2>&1 | tee multi_gpu_pretrain3.log