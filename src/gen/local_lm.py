#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from alignment.model_utils import get_tokenizer

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from alignment.model_utils import get_tokenizer

def get_model(model_id, model_revision, load_in_8bit, load_in_4bit, model_args, data_args, sft_args):
    model_id = sft_args.hub_model_id if model_id is None else model_id
    tokenizer = get_tokenizer(model_args, data_args)
    
    # Set up quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit, 
        load_in_4bit=load_in_4bit,
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
    )
    
    # Define model arguments
    model_kwargs = dict(
        revision=model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    
    # Define custom device map including all necessary components
    custom_device_map = {
        'model.embed_tokens': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model.layers.0': 'cuda',
        'model.layers.1': 'cuda',
        'model.layers.2': 'cuda',
        'model.layers.3': 'cuda',
        'model.layers.4': 'cuda',
        'model.layers.5': 'cuda',
        'model.layers.6': 'cuda',
        'model.layers.7': 'cuda',
        'model.layers.8': 'cuda',
        'model.layers.9': 'cuda',
        'model.layers.10': 'cuda',
        'model.layers.11': 'cuda',
        'model.layers.12': 'cuda',
        'model.layers.13': 'cuda',
        'model.layers.14': 'cuda',
        'model.layers.15': 'cuda',
        'model.layers.16': 'cpu',
        'model.layers.17': 'cpu',
        'model.layers.18': 'cpu',
        'model.layers.19': 'cpu',
        'model.layers.20': 'cpu',
        'model.layers.21': 'cpu',
        'model.layers.22': 'cpu',
        'model.layers.23': 'cpu',
        'model.layers.24': 'cpu',
        'model.layers.25': 'cpu',
        'model.layers.26': 'cpu',
        'model.layers.27': 'cpu',
        'model.layers.28': 'cpu',
        'model.layers.29': 'cpu',
        'model.layers.30': 'cpu',
        'model.layers.31': 'cpu',
        'model.layers.32': 'cpu',
        'model.layers.33': 'cpu',
        'model.layers.34': 'cpu',
        'model.layers.35': 'cpu',
        'model.layers.36': 'cpu',
        'model.layers.37': 'cpu',
        'model.layers.38': 'cpu',
        'model.layers.39': 'cpu',
        'model.layers.40': 'cpu',
        'model.layers.41': 'cpu',
        'model.layers.42': 'cpu',
        'model.layers.43': 'cpu',
        'model.layers.44': 'cpu',
        'model.layers.45': 'cpu',
        'model.layers.46': 'cpu',
        'model.layers.47': 'cpu',
        'model.layers.48': 'cpu',
        'model.layers.49': 'cpu',
        'model.layers.50': 'cuda',
        'model.layers.51': 'cuda',
        'model.layers.52': 'cuda',
        'model.layers.53': 'cuda',
        'model.layers.54': 'cuda',
        'model.layers.55': 'cuda',
        'model.layers.56': 'cuda',
        'model.layers.57': 'cuda',
        'model.layers.58': 'cuda',
        'model.layers.59': 'cuda',
        'model.layers.60': 'cuda',
        'model.layers.61': 'cuda',
        'model.layers.62': 'cuda',
        'model.layers.63': 'cuda',
        'model.layers.64': 'cuda',
        'model.layers.65': 'cuda',
        'model.layers.66': 'cuda',
        'model.layers.67': 'cuda',
        'model.layers.68': 'cuda',
        'model.layers.69': 'cuda',
        'model.layers.70': 'cuda',
        'model.layers.71': 'cuda',
        'model.layers.72': 'cuda',
        'model.layers.73': 'cuda',
        'model.layers.74': 'cuda',
        'model.layers.75': 'cuda',
        'model.layers.76': 'cuda',
        'model.layers.77': 'cuda',
        'model.layers.78': 'cuda',
        'model.layers.79': 'cuda',
        'model.norm': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lm_head': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Load the model with custom device map
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=custom_device_map, **model_kwargs)

    return tokenizer, model_id, model




def gen_model_outputs(model, tokenizer, batch_data, temperature=0.7, max_new_tokens=512, delimiter="assistant\n"):
    """
    gen_model_output generates and returns response (output) from a given model.
    """
    input_ids = tokenizer(
        batch_data["input_ids"], 
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(
        **input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1
    )
    
    outputs = []
    raw_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for raw_output in raw_outputs:
        
        if delimiter in raw_output:
            outputs.append(raw_output.split(delimiter)[1])
        else:
            print("Delimiter not found in output.")
            outputs.append(raw_output)  # Or handle appropriately

    return outputs


# %%
