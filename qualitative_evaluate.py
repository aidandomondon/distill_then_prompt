import argparse
from transformers import AutoTokenizer, LlamaForCausalLM, OPTForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
from torch.accelerator import current_accelerator
from prompt import LLamaPromptTuningLM, OPTPromptTuningLM, GPTPromptTuningLM
from transformers.models import llama as llama_loader
from prompt.modelutils import get_llama
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', type=str, default= "MiniLLM/teacher-gpt2-1.5B")
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt', type=str, default=None)
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/baseline/ptb/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/adamw_lr0.001_steps30000/c4/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/unpruned/ptb/best.ckpt")
parser.add_argument('--dtype', type = str, default = "auto")
parser.add_argument('--ntoken', type = int, default = 50)
parser.add_argument('--llm_cache_dir', type = str)
args = parser.parse_args()


def prepare_input_and_label(model, inputs_ids):
    # shift right
    if hasattr(model, 'n_tokens'):
        padded_input_tokens = model._extend_labels(inputs_ids)
    else:
        padded_input_tokens = inputs_ids
    labels = padded_input_tokens[..., 1:].contiguous()
    input_tokens = padded_input_tokens[..., :-1].contiguous()
    labels[input_tokens<0] = -100
    return labels

@torch.no_grad()
def evaluate(prompt_model, queries):
    prompt_model.eval()
    
    results = []
    for query in queries:
        inputs_ids = tokenizer.encode(query)
        if not isinstance(inputs_ids, torch.Tensor):
            inputs_ids = torch.Tensor(inputs_ids)
        inputs_ids = inputs_ids.to(torch.int32) # enforce token IDs to be stored as integers
        inputs_ids = inputs_ids.to(current_accelerator())
        inputs_ids.unsqueeze_(0)
        # if isinstance(prompt_model, LLamaPromptTuningLM):
        #     output = prompt_model.forward_with_soft_prompt(inputs_ids)
        # else:
        #     output = prompt_model(inputs_ids)
        # Simple greedy generation loop
        generated_tokens = inputs_ids.clone()
        for _ in range(50):  # max_new_tokens
            output = prompt_model(generated_tokens)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        response_text = tokenizer.decode(generated_tokens[0][inputs_ids.shape[1]:], skip_special_tokens=True)
        results.append({
            'query': query,
            'response': response_text
        })
    return results


if args.dtype == 'auto':
    dtype = 'auto'
elif args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
else:
    raise NotImplementedError

if 'llama' in args.model:
    if args.ckpt is None:
        model = get_llama(args.model_name_or_path)
    else:
        model = LLamaPromptTuningLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, n_tokens=args.ntoken)
    tokenizer = llama_loader.LlamaTokenizer.from_pretrained(args.model, use_fast=False)
elif 'opt' in args.model:
    if args.ckpt is None:
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = OPTPromptTuningLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, n_tokens=args.ntoken)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
elif 'gpt' in args.model.lower():
    # if len(os.listdir(args.llm_cache_dir)) > 0:
    #     local_files_only = True
    # else:
    #     fetch_remote = input("Local files dir empty. Attempt to download? [Y/n]")
    #     if fetch_remote == 'Y':
    #         local_files_only = False
    #     else:
    #         print('Aborting.')
    #         exit()
    local_files_only = False
    if args.ckpt:   # If we have a pretrained soft prompt available
        model = GPTPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            n_tokens=args.ntoken,
            dtype=torch.float32,
            device_map=current_accelerator().type,
            cache_dir=args.llm_cache_dir,
            local_files_only=local_files_only
        )
    else:   
        # Otherwise, this is a model with no accompanying soft prompt 
        # (likely an uncompressed model)
        model = GPT2LMHeadModel.from_pretrained(
            args.model_name_or_path,
            dtype=torch.float32,
            device_map=current_accelerator().type,
            cache_dir=args.llm_cache_dir,
            local_files_only=local_files_only,
        )
    tokenizer = GPT2Tokenizer.from_pretrained(
        args.model, 
        use_fast=False, 
        local_files_only=local_files_only, 
        cache_dir=args.llm_cache_dir
    )

print(model.dtype)

if args.ckpt is not None:
    state_dicts = torch.load(args.ckpt)
    soft_prompt_state_dict = state_dicts['model']
    model.soft_prompt.load_state_dict(soft_prompt_state_dict)
model.seqlen = model.config.max_position_embeddings
model.seqlen = 1024
model.to(current_accelerator().type)
model.eval()

print(f'ckpt: {args.ckpt}\nmodel_name_or_path: {args.model_name_or_path}\n')

print("Responses:\n")
results = evaluate(model, queries = [
    "Please give answers to this question: Where is Long Beach?",
    "Please give answers to this question: Where is Tulsa, Oklahoma?",
    "Please give answers to this question: What is Asparagus?"
])
for result in results:
    print(f"Query: {result['query']}\nResponse: {result['response']}\n")