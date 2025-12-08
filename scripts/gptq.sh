dataset=c4  # Learning on C4 because Xu et al., 2024 says it yields the most transferrable prompts
model_name_or_path=MiniLLM/MiniLLM-gpt2-120M # Distilled student
model=MiniLLM/teacher-gpt2-1.5B # Undistilled teacher
per_device_train_batch_size=1
per_device_eval_batch_size=1
soft_token_num=25

for opt in adamw; do
for lr in 0.001; do
for steps in 1000; do
# original num steps = 30000 induces 5 hrs training time on 4 48GB GPUs.
# trying num_steps = 300 for 1.2 hr training on 1 8GB GPU.

LOG_FILE_NAME=./logs/prompt_generation/log_kd_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}.txt
touch $LOG_FILE_NAME && \
python soft_prompt_learning.py \
    --model_name_or_path ${model_name_or_path} \
    --model ${model} \
    --dataset ${dataset} \
    --eval_every_steps 100 \
    --seqlen 1024 \
    --soft_token_num ${soft_token_num} \
    --prompt_lr ${lr} \
    --max_steps ${steps} \
    --optimizer ${opt} \
    --output_dir ./gptq/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${dataset} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    --dataset_cache_dir ./.huggingface_cache/dataset/ \
    2>&1 | tee $LOG_FILE_NAME
done 
done
done

# | tee ./logs/log_${opt}_lr${lr}_${dataset}_steps${steps}.txt
# --per_device_eval_batch_size 1 \