model=MiniLLM/teacher-gpt2-1.5B # Model from which to instantiate tokenizer
prompt_tuning_dataset=c4 # Dataset used to train prompts

for dataset in wikitext2 ptb c4; do

# Evaluate prompt-aided distills
model_name_or_path=MiniLLM/MiniLLM-gpt2-120M
for opt in adamw; do
for lr in 0.001; do
for steps in 300 1000; do
for soft_token_num in 25; do
for ckpt in ./generated_prompts/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${prompt_tuning_dataset}/best.pth; do

echo "Evaluating ${model_name_or_path} model with ${soft_token_num}-token prompt from ${ckpt} generated with ${steps} steps, ${lr} learning rate, and ${opt} optimizer."
LOG_FILE_NAME=./logs/evaluation/kd_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}_dataset${dataset}.txt
touch $LOG_FILE_NAME && \
python evaluate.py \
    --model ${model} \
    --model_name_or_path ${model_name_or_path} \
    --ckpt ${ckpt} \
    --dataset ${dataset} \
    --ntoken ${soft_token_num} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    --dataset_cache_dir ./.huggingface_cache/dataset/ \
    2>&1 | tee $LOG_FILE_NAME

done
done
done
done
done

# Evaluate unprompted student and unprompted teacher as well
model_name_or_path=MiniLLM/MiniLLM-gpt2-120M
echo "Evaluating ${model_name_or_path} student model with no prompt."
LOG_FILE_NAME=./logs/evaluation/unprompted_student_dataset${dataset}.txt
touch $LOG_FILE_NAME && \
python evaluate.py \
    --model ${model} \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    --dataset_cache_dir ./.huggingface_cache/dataset/ \
    2>&1 | tee $LOG_FILE_NAME

model_name_or_path=MiniLLM/teacher-gpt2-1.5B
echo "Evaluating ${model_name_or_path} teacher model with no prompt."
LOG_FILE_NAME=./logs/evaluation/unprompted_teacher_dataset${dataset}.txt
touch $LOG_FILE_NAME && \
python evaluate.py \
    --model ${model} \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    --dataset_cache_dir ./.huggingface_cache/dataset/ \
    2>&1 | tee $LOG_FILE_NAME

done