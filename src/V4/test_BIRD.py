from agents import LLMSearchAgent, SemanticSearchAgent

from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import get_usage_metadata_callback

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import Image, display
from typing_extensions import Literal
from pydantic import BaseModel, Field
from datetime import datetime
from textwrap import dedent
from statistics import mean
from tqdm import tqdm
import numpy as np
import faiss
import json
import time

error_catching = True
vLLM_max_model_len = '8k'
agent_name = ''
model_name=''

results = {}
metadatadict = {}
start_time = time.time()
def run_BIRD_test(agent_name:str, llm_path:str):
    agent_name = agent_name.lower()

    match agent_name:
        case 'llmsearch':
            agent = LLMSearchAgent(llm_path = llm_path)


    max_workers = 10  # Adjust based on kv_cache
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(call_agent, i, entry): i 
            for i, entry in enumerate(mini_dev_sql) #number of items to test
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            i, result, metadata = future.result()
            results[str(i)] = result
            metadatadict[str(i)] = metadata

    results = dict(sorted(results.items(), key=lambda x: int(x[0])))
    metadata = dict(sorted(metadatadict.items(), key=lambda x: int(x[0])))

    elapsed = time.time() - start_time
    elapsed_min = elapsed/60

    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    #metadatadict

    #metadatadict = 'average': averages , 'individual': metadata

    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev-main\sql_result\{agent_name}\{model_name}\results_{dt_now}.json', 'w') as f:
        json.dump(results, f, indent=4)

    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev-main\sql_result\{agent_name}\{model_name}\metadata\metadata_{dt_now}.json', 'w') as f:
        json.dump(metadatadict, f, indent=4)

    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev-main\sql_result\{agent_name}\{model_name}\metadata\results_{dt_now}.log', 'w') as f:
        averages ={}
        # token_keys = ['tool_input_tokens', 'tool_output_tokens','gen_input_tokens', 'gen_output_tokens']
        # averages = {
        #     key: mean(entry['tokens'][key] for entry in metadatadict.values() if entry['tokens'][key] != -1) 
        #     for key in token_keys
        # }
        
        #maximum = {key: max(entry['tokens'][key] for entry in metadatadict.values()) for key in token_keys}
        averages['latency_s'] = mean(entry['latency_s'] for entry in metadatadict.values()if entry['latency_s']!= -1)
        averages['latency_m'] = mean(entry['latency_m'] for entry in metadatadict.values()if entry['latency_m']!= -1)
        
        # f.write(f"{len(metadatadict)} entries):\n")
        # f.write(f"Tool Input Tokens:  {averages['tool_input_tokens']:.2f}\n")
        # f.write(f"Tool Output Tokens: {averages['tool_output_tokens']:.2f}]\n")
        # f.write(f"Avg Tool Input Tokens:   {maximum['tool_input_tokens']:.2f}\n")
        # f.write(f"Avg Tool Output Tokens:  {maximum['tool_output_tokens']:.2f}\n")

        # f.write(f"Avg Gen Input Tokens:   {averages['gen_input_tokens']:.2f}\n")
        # f.write(f"Avg Gen Output Tokens:  {averages['gen_output_tokens']:.2f}\n")
        # f.write(f"Max Gen Input Tokens:   {maximum['gen_input_tokens']:.2f}\n")
        # f.write(f"Max Gen Output Tokens:  {maximum['gen_output_tokens']:.2f}\n")   

        f.write(f'Total Latency (seconds):  {round(elapsed,2)}s / Latency (minutes):  {round(elapsed_min,2)}min\n')
        f.write(f"Avg  Latency (seconds):  {averages['latency_s']:.2f}/ Avg  Latency (minutes):  {averages['latency_m']:.4f}\n")

        # f.write(f'Langgraph gen_llm max_tokens: {max_completion_tokens}\n')
        # f.write(f'vLLM model max-model-len: {vLLM_max_model_len}\n')