import os
import asyncio
import argparse
import openai
import yaml
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from langfuse import Langfuse

from utils import is_push_to_hf_hub_enabled, update_args
from src.pipeline.eval import eval_on_records
from src.pipeline.utils import push_to_hf_hub


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def set_langfuse_env_vars(config):
    """Set Langfuse environment variables based on the YAML configuration."""
    os.environ["LANGFUSE_PUBLIC_KEY"] = config['langfuse_public_key']
    os.environ["LANGFUSE_SECRET_KEY"] = config['langfuse_secret_key']
    os.environ["LANGFUSE_HOST"] = config['langfuse_host']


def log_to_langfuse(dataset_dict, session_id, split_name="eval"):
    """Log results to Langfuse for each record in the dataset."""
    dataset = dataset_dict[split_name]
    df = dataset.to_pandas()

    langfuse = Langfuse()

    # Iterate through each row in the DataFrame to log traces and scores
    for i, row in df.iterrows():
        model_name = row['model_name']
        
        # Convert reasons from numpy array to plain text
        reasons = {key: value[0] if isinstance(value, (np.ndarray, pd.Series)) else value for key, value in row['reasons'].items()}
        
        # Create a trace for each interaction with dynamic session_id
        trace = langfuse.trace(name=model_name, session_id=session_id,
                               input={'prompt': row['prompt']},
                               output={'response': row['response']})
        trace.span(
            name=f"evaluation_number_{i}",
            input={'eval_prompts': row['eval_prompts']},
            output={'reasons': reasons}
        )
        # Log scores associated with each trace
        for metric_name in ['relevance_scores', 'completeness_scores', 'grammar_and_style_scores', 'technical_accuracy_scores', 'openxml_structure_scores']:
            langfuse.score(
                name=metric_name,
                value=row[metric_name],
                trace_id=trace.trace_id  # assuming trace_id is accessible from the trace object
            )

        # Ensure all events are processed before moving to the next interaction
        langfuse.flush()


def push_to_elasticsearch(config, dataset_dict):
    """Push evaluation results to Elasticsearch based on the configuration."""
    if not config.get('push_to_elasticsearch', False):
        print("Elasticsearch push is disabled.")
        return
    
    es_url = config.get('es_url')
    es_username = config.get('es_username')
    es_password = config.get('es_password')
    
    if not es_url or not es_username or not es_password:
        print("Elasticsearch configuration is missing.")
        return

    es = Elasticsearch(es_url, basic_auth=(es_username, es_password))
    
    # Convert dataset to a Pandas DataFrame for easy manipulation
    df = dataset_dict['eval'].to_pandas()

    for i, row in df.iterrows():
        document = {
            "prompt": row['prompt'],
            "response": row['response'],
            "model_name": row['model_name'],
            "relevance_scores": row['relevance_scores'],
            "completeness_scores": row['completeness_scores'],
            "grammar_and_style_scores": row['grammar_and_style_scores'],
            "technical_accuracy_scores": row['technical_accuracy_scores'],
            "openxml_structure_scores": row['openxml_structure_scores'],
            "reasons": row['reasons'],
            "evaluators": row['evaluators']
        }

        try:
            # Index the document into Elasticsearch
            res = es.index(index="evaluation-results", document=document)
            print(f"Document {i} indexed successfully. ID: {res['_id']}")
        except Exception as e:
            print(f"Error indexing document {i}: {str(e)}")


async def evaluate(args, config):
    """Run the evaluation and save or log results."""
    eval_results = await eval_on_records(
        args.lm_response_ds_id,
        args.prompt_tmpl_path, 
        args.service_model_name, 
        args.eval_workers, 
        args.eval_repeat,
        args.eval_data_preprocess_bs, 
        args.eval_ds_split, 
        args.rate_limit_per_minute
    )

    # Save the results to disk
    eval_results["ds_with_scores"].save_to_disk(args.eval_ds_id)
    
    # Log results to Langfuse
    log_to_langfuse(eval_results["ds_with_scores"], config.get('session_id', "DefaultSessionID"), args.eval_ds_split)
    
    # Push to Elasticsearch
    push_to_elasticsearch(config, eval_results["ds_with_scores"])
    
    print(eval_results)
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for batch inference step")

    # OpenAI API Key
    parser.add_argument("--openai-api-key", type=str, default="****************************",
                        help="OpenAI API key for authentication.")
    # Path to the configuration file
    parser.add_argument("--from-config", type=str, default="config/evaluation.yaml",
                        help="Set CLI options from YAML config")
    # Service model name
    parser.add_argument("--service-model-name", type=str, default="gpt-3.5-turbo",
                        help="Which service LLM to use for evaluation of the local fine-tuned model")
    # Rate limit for the service model
    parser.add_argument("--rate-limit-per-minute", type=int, default=60,
                        help="Rate-limit per minute for the service LLM.")
    # Path to the prompt template
    parser.add_argument("--prompt-tmpl-path", type=str, 
                        default=os.path.abspath("config/prompts.toml"),
                        help="Path to the prompts TOML configuration file.")
    # Dataset ID for the response data
    parser.add_argument("--lm-response-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    # Split of the dataset to use
    parser.add_argument("--lm-response-ds-split", type=str, default="test",
                        help="Split of the lm response dataset to use for saving or retrieving.")
    # Evaluation batch size
    parser.add_argument("--eval-data-preprocess-bs", type=int, default=16)
    # Number of times to repeat evaluation
    parser.add_argument("--eval-repeat", type=int, default=10)
    # Number of workers to use for evaluation
    parser.add_argument("--eval-workers", type=int, default=4,
                        help="Number of workers to use for parallel evaluation.")
    # Whether to push evaluation to Hugging Face Hub
    parser.add_argument("--push-eval-to-hf-hub", action="store_true",
                        help="Whether to push generated evaluation to Hugging Face Dataset repository (Hub)")
    # ID of the Hugging Face dataset for evaluation
    parser.add_argument("--eval-ds-id", type=str, default=None,
                        help="Hugging Face Dataset repository ID")
    # Split of the evaluation dataset
    parser.add_argument("--eval-ds-split", type=str, default="eval",
                        help="Split of the lm eval dataset to use for saving.") 

    args = parser.parse_args()
    args = update_args(parser, args)

    # Load the configuration file
    config = load_config(args.from_config)
    
    # Set environment variables for Langfuse
    set_langfuse_env_vars(config)

    # Run the evaluation asynchronously
    asyncio.run(evaluate(args, config))
