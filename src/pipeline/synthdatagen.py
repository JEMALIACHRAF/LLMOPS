import os
import json
import toml
import numpy as np
import asyncio
from tqdm import tqdm
from string import Template
from datasets import load_from_disk, Dataset, DatasetDict
from collections import deque
from typing import List, Dict
from ..gen.gpt import get_model as get_service_model
from ..gen.utils import call_service_llm, _calculate_job_distribution

# JSON_KEYS_TO_CHECK is used for validating specific keys in the JSON response (can remain empty)
JSON_KEYS_TO_CHECK = {}

def _load_all_json_files(filenames):
    """
    Loads JSON files and returns them as Python dictionaries.
    """
    all_json_dicts = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                all_json_dicts.append(data)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON in file: {filename}")
    return all_json_dicts

def _format_response(responses: List[Dict[str, str]]):
    """
    Formats the 'prompt' and 'response' into a list of dictionaries with
    user and assistant roles.
    """
    final_instruction_answer_pairs = []
    for response in responses:
        user_response_dict = {}
        assistant_response_dict = {}
        user_response_dict["content"] = response["prompt"]
        user_response_dict["role"] = "user"
        assistant_response_dict["content"] = response["response"]
        assistant_response_dict["role"] = "assistant"
        final_instruction_answer_pairs.append([user_response_dict, assistant_response_dict])
    return final_instruction_answer_pairs

def _sampling(loaded_dataset, num_sample, seed):
    """
    Randomly samples a specified number of examples from the dataset.
    """
    np.random.seed(seed)
    total_samples = len(loaded_dataset)
    random_indices = np.random.randint(0, total_samples, size=num_sample)
    
    # Select random samples
    sampled_data = loaded_dataset.select(random_indices)
    return sampled_data

def _get_synth_data_gen_prompt_tmpl(prompt_tmpl_path):
    """
    Loads the prompt template from a TOML file.
    """
    prompts = toml.load(prompt_tmpl_path)
    return prompts['synth_data_gen']['prompt']

def _craft_prompt(prompt_tmpl, prompt, response):
    """
    Uses a template to craft a prompt with a given instruction (prompt) and response.
    """
    return Template(prompt_tmpl).substitute(
        instruction=prompt,
        response=response
    )

def _craft_prompts(samples, prompt_tmpl):
    """
    Crafts a list of prompts from the sampled dataset and includes model_name.
    """
    prompts = [
        {
            'crafted_prompt': _craft_prompt(
                prompt_tmpl,
                sample["prompt"], 
                sample["response"]
            ),
            'model_name': sample["model_name"]
        }
        for sample in samples
    ]
    return prompts

async def _gen_synth_data(prompts, model, eval_workers, rate_limit_per_minute):
    """
    Asynchronously generates synthetic data by calling the service LLM with prompts
    """
    generated_data = []
    jobs_at_once, sleep_interval = _calculate_job_distribution(rate_limit_per_minute, num_workers=eval_workers)
    prompt_queue = deque(prompts)

    with tqdm(total=len(prompts), desc="Generating batches") as pbar:
        while prompt_queue:
            tasks = []
            for _ in range(min(jobs_at_once, len(prompt_queue))):
                eval_item = prompt_queue.popleft()
                eval_prompt = eval_item['crafted_prompt']
                model_name = eval_item['model_name']
                
                task = asyncio.create_task(
                    call_service_llm(model, eval_prompt, JSON_KEYS_TO_CHECK, retry_num=10, job_num=len(generated_data))
                )
                task.model_name = model_name
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Log the raw responses to inspect them
            for result in results:
                print(f"Raw response from model for prompt {eval_prompt}: {result}")
            
            # Filter out any invalid results
            # Adjust validation: Add logging to see what is skipped
            for i, result in enumerate(results):
                if result and isinstance(result[1], dict):
                    valid_results = result[1]  # Assuming result[1] is the valid part
                    generated_data.append(valid_results)
                else:
                    print(f"Invalid result for task {i}: {result}")  # Log invalid data
            
            pbar.update(len(results))
            
            # Rate limiting
            await asyncio.sleep(sleep_interval)
        
    return generated_data


import os
import json
from tqdm import tqdm

async def synth_data_generation(
    loaded_dataset, seed, num_sample, prompt_tmpl_path,
    service_model_name, gen_workers, rate_limit_per_minute
):
    """
    Main function for generating synthetic data with model_name.
    """
    # Sample the dataset
    samples = _sampling(loaded_dataset, num_sample, seed)
    
    # Load the prompt template
    prompt_tmpl = _get_synth_data_gen_prompt_tmpl(prompt_tmpl_path)
    
    # Craft prompts including model_name
    prompts = _craft_prompts(samples, prompt_tmpl)

    # Get the model to generate synthetic data
    gen_model = get_service_model(service_model_name)
    print("Generating synthetic data")
    generated_data = await _gen_synth_data(prompts, gen_model, gen_workers, rate_limit_per_minute)

    # Save the generated data into JSON files
    save_dir_path = "Generated_Data"
    os.makedirs(save_dir_path, exist_ok=True)
    filenames = []
    print("Exporting generated data to JSON files")

    for i, (prompt_data, data) in tqdm(enumerate(zip(prompts, generated_data)), total=len(generated_data), desc="Saving to JSON"):
        # Ensure data is not empty and in the expected format
        if data and isinstance(data, dict):  # Adjusted to check for dict directly
            try:
                # Attach model_name and seed_prompt to the saved data
                data["seed_prompt"] = prompt_data['crafted_prompt']
                data["model_name"] = prompt_data['model_name']  # Include model_name
                filename = f"{save_dir_path}/generated_data_{i}.json"
                filenames.append(filename)

                # Save to JSON
                with open(filename, "w", encoding='utf-8') as f:
                    json.dump(data, f)
                print(f"File saved: {filename}")

            except KeyError as e:
                print(f"KeyError occurred while processing prompt {i}: {e}")
            except Exception as e:
                print(f"Failed to save file for prompt {i}: {e}")
        else:
            print(f"No valid data generated for prompt {i}. Skipping.")
            # Print the raw data that was skipped for debugging
            print(f"Skipped data for prompt {i}: {data}")

    return filenames

def collage_as_dataset(filenames, service_model_name, topic, split):
    """
    Loads JSON files and organizes them into a Hugging Face DatasetDict
    with columns: 'prompt', 'response', and 'model_name'.
    """
    all_json_dicts = _load_all_json_files(filenames)
    
    prompts = []
    responses = []
    model_names = []

    for json_dict in all_json_dicts:
        # Extract the 'contents' list
        if "contents" in json_dict and isinstance(json_dict["contents"], list):
            for entry in json_dict["contents"]:
                # Extract the instruction and response from each entry inside 'contents'
                prompts.append(entry["instruction"])  # Extract instruction (which is treated as the prompt)
                responses.append(entry["response"])  # Extract response
                # Extract model_name from the top level of json_dict
                model_names.append(json_dict.get("model_name", service_model_name))  # Use service_model_name if not found

    # Construct the dataset in the same format as the input data
    dataset_train = Dataset.from_dict(
        {
            "prompt": prompts,
            "response": responses,
            "model_name": model_names
        }
    )
    
    return DatasetDict(
        {split: dataset_train}
    )
