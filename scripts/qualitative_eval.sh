model=MiniLLM/teacher-gpt2-1.5B # Model from which to instantiate tokenizer
prompt_tuning_dataset=c4 # Dataset used to train prompts

# Evaluate prompt-aided distills
model_name_or_path=MiniLLM/MiniLLM-gpt2-120M
for opt in adamw; do
for lr in 0.001; do
for steps in 300 1000; do
for soft_token_num in 25; do
for ckpt in ./generated_prompts/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${prompt_tuning_dataset}/best.pth; do

echo "Generating responses with ${model_name_or_path} model with ${soft_token_num}-token prompt from ${ckpt} generated with ${steps} steps, ${lr} learning rate, and ${opt} optimizer."
LOG_FILE_NAME=./logs/qualitative_evaluation/kd_${opt}_lr${lr}_${prompt_tuning_dataset}_steps${steps}_token${soft_token_num}_qualitative_evaluation.txt
touch $LOG_FILE_NAME && \
python qualitative_evaluate.py \
    --model ${model} \
    --model_name_or_path ${model_name_or_path} \
    --ckpt ${ckpt} \
    --ntoken ${soft_token_num} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    2>&1 | tee $LOG_FILE_NAME

done
done
done
done
done

# Evaluate unprompted student
model_name_or_path=MiniLLM/MiniLLM-gpt2-120M
echo "Generating responses with ${model_name_or_path} student model with no prompt."
LOG_FILE_NAME=./logs/qualitative_evaluation/unprompted_student_qualitative_evaluation.txt
touch $LOG_FILE_NAME && \
python qualitative_evaluate.py \
    --model ${model} \
    --model_name_or_path ${model_name_or_path} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    2>&1 | tee $LOG_FILE_NAME

# Evaluated unprompted teacher
model_name_or_path=MiniLLM/teacher-gpt2-1.5B
echo "Generating responses with ${model_name_or_path} teacher model with no prompt."
LOG_FILE_NAME=./logs/qualitative_evaluation/unprompted_teacher_qualitative_evaluation.txt
touch $LOG_FILE_NAME && \
python qualitative_evaluate.py \
    --model ${model} \
    --model_name_or_path ${model_name_or_path} \
    --llm_cache_dir ./.huggingface_cache/llm/ \
    2>&1 | tee $LOG_FILE_NAME