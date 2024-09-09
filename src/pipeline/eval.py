import toml
import asyncio
from tqdm import tqdm
from string import Template
from datetime import datetime
from datasets import load_dataset, DatasetDict
from datasets import load_from_disk
from collections import deque
from langfuse import Langfuse  # Import Langfuse

from ..gen.gpt import get_model as get_service_model
from ..gen.utils import call_service_llm, _calculate_job_distribution

JSON_KEYS_TO_CHECK = []

def _get_eval_prompt_tmpl(eval_prompt_tmpl_path, use_langfuse, langfuse_instance=None):
    """
    _get_eval_prompt_tmpl returns pre-defined prompt template for 
    evaluation of language model's generated output. 
    It uses Langfuse if use_langfuse is True.
    """
    if use_langfuse and langfuse_instance:
        prompt = langfuse_instance.get_prompt("Report_eval")
        return prompt  # Return Langfuse prompt object
    else:
        prompts = toml.load(eval_prompt_tmpl_path)
        return prompts['Report_eval']['prompt']

def _get_lm_response_dataset(lm_response_dataset_id, eval_prompt_tmpl, batch_size, use_langfuse):
    """
    _get_lm_response_dataset loads the dataset from a local directory and processes it to add evaluation prompts.
    """
    # Load dataset from disk
    dataset_dict = load_from_disk(lm_response_dataset_id)
    dataset = dataset_dict["test"]  # Adjust this if using a different split

    # Batch process the dataset to include eval prompts
    def __batch_process(batch):
        eval_prompts = []
        for (instruction, generated_section) in zip(
            batch['prompt'], batch['response']
        ):
            if use_langfuse:
                prompt = eval_prompt_tmpl.compile(
                    instruction=instruction,
                    generated_section=generated_section,
                )
            else:
                prompt = Template(eval_prompt_tmpl).substitute(
                    instruction=instruction,
                    generated_section=generated_section,
                )
            eval_prompts.append(prompt)
        batch["eval_prompts"] = eval_prompts
        return batch

    return dataset.map(__batch_process, batched=True, batch_size=batch_size)

async def _gen_eval_on_records(eval_prompts, eval_model, eval_workers, rate_limit_per_minute):
    """
    _gen_eval_on_records simultaneously generates evaluations on the eval_prompts,
    respecting rate limits and scheduling constraints.
    """
    assessments = []
    jobs_at_once, sleep_interval = _calculate_job_distribution(rate_limit_per_minute, num_workers=eval_workers)
    prompt_queue = deque(eval_prompts)  # Use a deque to efficiently manage the queue of prompts

    while prompt_queue:
        tasks = []
        for _ in range(min(jobs_at_once, len(prompt_queue))):
            eval_prompt = prompt_queue.popleft()  # Take the prompt from the front of the queue
            task = asyncio.create_task(
                call_service_llm(eval_model, eval_prompt, JSON_KEYS_TO_CHECK, retry_num=10, job_num=len(assessments))
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        results = sorted(results, key=lambda item: item[0])
        print("Results received:", results)
        assessments.extend(result[1] for result in results)  # Simplify processing and appending

        # Implement rate limiting
        await asyncio.sleep(sleep_interval)

    return assessments

def _iterate_inner_lists(outer_list):
    num_items = len(outer_list[0])  # Get number of items from any inner list
    for i in range(num_items):
        yield tuple(inner_list[i] for inner_list in outer_list)

async def eval_on_records(
    lm_response_dataset_id,
    eval_prompt_tmpl_path,
    service_model_name,
    eval_workers,
    eval_repeat,
    batch_size,
    eval_dataset_split,
    rate_limit_per_minute,
    prompt_langfuse=False  # Add this parameter to control prompt source
):
    eval_model = get_service_model(service_model_name)

    langfuse_instance = None
    if prompt_langfuse:
        langfuse_instance = Langfuse()  # Initialize Langfuse if needed

    eval_prompt_tmpl = _get_eval_prompt_tmpl(eval_prompt_tmpl_path, prompt_langfuse, langfuse_instance)
    lm_response_ds = _get_lm_response_dataset(lm_response_dataset_id, eval_prompt_tmpl, batch_size, prompt_langfuse)

    criteria = ["relevance", "completeness", "grammar_and_style", "technical_accuracy", "openxml_structure"]
    results = {criterion: [] for criterion in criteria}
    reasons = []  # Add a list for combined reasons

    for idx in tqdm(range(0, len(lm_response_ds), eval_workers), desc="batches"):
        batch_data = lm_response_ds[idx:idx+eval_workers]
        
        partial_assessments = []
        
        for _ in tqdm(range(eval_repeat), desc="repeat"):
            assessments = await _gen_eval_on_records(batch_data["eval_prompts"], eval_model, eval_workers, rate_limit_per_minute)
            partial_assessments.append(assessments)
            
        for partial_idx, each_assessments in enumerate(_iterate_inner_lists(partial_assessments)):
            scores = {criterion: [] for criterion in criteria}
            reasons_batch = {criterion: [] for criterion in criteria}  # Add a dictionary for batch reasons
            
            for each_assessment in each_assessments:
                for criterion in criteria:
                    scores[criterion].append(int(each_assessment[criterion]['score']))
                    reasons_batch[criterion].append(each_assessment[criterion]['reason'])  # Add the reason

            avg_scores = {criterion: sum(scores[criterion]) / len(scores[criterion]) for criterion in criteria}
            avg_reasons = {criterion: reasons_batch[criterion] for criterion in criteria}  # Keep the reasons
            
            for criterion in criteria:
                results[criterion].append(avg_scores[criterion])
            
            reasons.append(avg_reasons)  # Add the reasons to the final result
            
            print(f"Eval on (sample_num={idx+partial_idx}): " + ", ".join(f"{criterion} score: {avg_scores[criterion]}" for criterion in criteria))

    ds_with_scores = lm_response_ds
    for criterion in criteria:
        ds_with_scores = ds_with_scores.add_column(f"{criterion}_scores", results[criterion])
    
    ds_with_scores = ds_with_scores.add_column("reasons", reasons)  # Add a single column for reasons
    ds_with_scores = ds_with_scores.add_column("evaluators", [service_model_name] * len(lm_response_ds))

    ds_with_scores = DatasetDict({eval_dataset_split: ds_with_scores})

    return {
        "ds_with_scores": ds_with_scores
    }
