import datasets
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from datasets import load_from_disk
from elasticsearch import Elasticsearch
from ..gen.local_lm import get_model, gen_model_outputs
from ..pipeline.utils import get_args, get_sha

def _get_test_dataset(dataset_id, split, tokenizer, batch_size):
    """
    _get_test_dataset returns the target test dataset
    """
    def __batch_process(batch):
        batch["input_ids"] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False
            )
            for prompt in batch["prompt"]
        ]
        return batch
    
    dataset_dict = load_from_disk(dataset_id)
        
    return dataset_dict[split].map(
        __batch_process, batched=True, batch_size=batch_size
    )

def save_to_elasticsearch(es, index_name, results):
    """
    Save results to Elasticsearch index
    """
    for i in range(len(results['prompt'])):
        doc = {
            "prompt": results['prompt'][i],
            "response": results['response'][i],
            "model_id": results['model_id'][i],
            "model_sha": results['model_sha'][i]
        }
        es.index(index=index_name, body=doc)

def gen_local_lm_responses(
    model_id, model_revision, 
    load_in_8bit, load_in_4bit,
    test_dataset_id, test_dataset_split, 
    data_preprocess_bs, inference_bs, repeat,
    lm_response_dataset_split, config_path, 
    es_params=None
):
    model_args, data_args, sft_args = get_args(config_path)
    tokenizer, model_id, model = get_model(
        model_id, model_revision, load_in_8bit, load_in_4bit, model_args, data_args, sft_args,
    )
    model_sha = get_sha(model_id, model_revision)
    ds = _get_test_dataset(test_dataset_id, test_dataset_split, tokenizer, data_preprocess_bs)

    results = {"prompt": [], "response": []}

    for idx in tqdm(range(0, len(ds), inference_bs), desc="batches"):
        for repeat_idx in tqdm(range(repeat), desc="repeat"):
            batch_data = ds[idx:idx+inference_bs]
            lm_responses = gen_model_outputs(model, tokenizer, batch_data)

            for prompt, lm_response in zip(batch_data["prompt"], lm_responses):
                instruction = prompt 
                results["prompt"].append(instruction)
                results["response"].append(lm_response)

    results['model_id'] = [model_id] * len(results["prompt"])
    results['model_sha'] = [model_sha] * len(results["response"])
    
    ds = Dataset.from_dict(results)
    
    # Save results to Elasticsearch if es_params is provided
    if es_params:
        es = Elasticsearch(es_params["es_url"], basic_auth=(es_params["username"], es_params["password"]))
        save_to_elasticsearch(es, es_params["index_name"], results)

    return DatasetDict(
        {lm_response_dataset_split: ds}
    )

