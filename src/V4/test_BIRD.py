from agents import BaselineAgent, LLMSearchAgent, SemanticSearchAgent
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, median, stdev
from datetime import datetime
from textwrap import dedent
from statistics import mean
from tqdm import tqdm
import json
import time
import os


agent_name = 'baseline' #lowercase
model_name='gpt-oss-20b'
llm_path='/home/012155624/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee'
max_completion_tokens = 2000 #int
vllm_max_model_len = '27k' # str

error_catching = True
max_workers = 10
data_subset=500

eval = True

results = {}
metadatadict = {}
start_time = time.time()
def run_BIRD_test(agent_name:str, llm_path:str, max_completion_tokens:int=2000, vllm_max_model_len:int=-1, data_subset:int=None, max_workers:int=10, 
                  error_catching:bool=False):
    with open(r'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\mysql\mini_dev_mysql.json') as f:
        mini_dev_sql = json.load(f)

    if data_subset:
        mini_dev_sql = mini_dev_sql[:data_subset]

    agent_name = agent_name.lower()

    match agent_name:
        case 'baseline':
            agent = BaselineAgent(llm_path,max_completion_tokens)
        case 'llmsearch':
            agent = LLMSearchAgent(llm_path,max_completion_tokens)
        case 'semanticsearch':
            agent = SemanticSearchAgent(llm_path,max_completion_tokens)

    # Execute Agents in parallel
    results = {}
    metadatadict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(agent.call_agent, i, entry, error_catching): i 
            for i, entry in enumerate(mini_dev_sql)
        }
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            i, result, metadata = future.result()
            results[str(i)] = result
            metadatadict[str(i)] = metadata


    # Aggregate and record results and meta_data
    results = dict(sorted(results.items(), key=lambda x: int(x[0])))
    metadata = dict(sorted(metadatadict.items(), key=lambda x: int(x[0])))

    elapsed = time.time() - start_time
    elapsed_min = elapsed/60

    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    directory = rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\sql_result\{agent_name}\{model_name}'
    os.makedirs(directory, exist_ok=True)

    predicated_path = rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\sql_result\{agent_name}\{model_name}\results_{dt_now}.json'

    with open(predicated_path, 'w') as f:
        json.dump(results, f, indent=4)

    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\sql_result\{agent_name}\{model_name}\metadata_{dt_now}.json', 'w') as f:
        json.dump(metadatadict, f, indent=4)

    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\sql_result\{agent_name}\{model_name}\metadatapretty_{dt_now}.log', 'w') as f:
        
        latence_s_values = [entry['latency_s'] for entry in metadatadict.values()if entry['latency_s']!= -1]
        latency_s_stats = {
                'mean': mean(latence_s_values),
                'median': median(latence_s_values),
                'std': stdev(latence_s_values) if len(latence_s_values) > 1 else 0,
                'max': max(latence_s_values),
                'min': min(latence_s_values)
            }
        f.write(f'Total Latency (seconds):  {round(elapsed,2)}s / Latency (minutes):  {round(elapsed_min,2)}min\n')
        f.write(f"Avg  Latency (seconds):  {latency_s_stats['mean']:.2f}\n")
        gen_token_keys = ['input_tokens', 'output_tokens']
        token_stats = {}
        for key in gen_token_keys:
            values = [entry[key][0] for entry in metadatadict.values() if entry[key] != -1]
            
            token_stats[key] = {
                'mean': mean(values),
                'median': median(values),
                'std': stdev(values) if len(values) > 1 else 0,
                'max': max(values),
                'min': min(values)
            }

        f.write(f"Avg Gen Input Tokens:   {token_stats['input_tokens']['mean']:.2f}\n")
        f.write(f"Avg Gen Output Tokens:  {token_stats['output_tokens']['mean']:.2f}\n")
        f.write(f"Max Gen Input Tokens:   {token_stats['input_tokens']['max']:.2f}\n")
        f.write(f"Max Gen Output Tokens:  {token_stats['output_tokens']['max']:.2f}\n")

        if agent_name in ['llmsearch']:
            tool_token_keys = ['tool_input_tokens', 'tool_output_tokens']
            tool_stats = {}
            for key in tool_token_keys:
                values = [entry[key][0] for entry in metadatadict.values() if entry[key] != -1]
                
                tool_stats[key] = {
                    'mean': mean(values),
                    'median': median(values),
                    'std': stdev(values) if len(values) > 1 else 0,
                    'max': max(values),
                    'min': min(values)
                }
            f.write(f"Tool Input Tokens:  {tool_stats['tool_input_tokens']:.2f}\n")
            f.write(f"Tool Output Tokens: {tool_stats['tool_output_tokens']:.2f}]\n")
            f.write(f"Avg Tool Input Tokens:   {tool_stats['tool_input_tokens']:.2f}\n")
            f.write(f"Avg Tool Output Tokens:  {tool_stats['tool_output_tokens']:.2f}\n")  



        f.write(f'Langgraph gen_llm max_tokens: {max_completion_tokens}\n')
        f.write(f'vLLM model max-model-len: {vllm_max_model_len}\n')
    
    return predicated_path





predicated_path = run_BIRD_test(agent_name=agent_name,llm_path=llm_path, max_completion_tokens=max_completion_tokens, vllm_max_model_len=vllm_max_model_len, 
              data_subset=data_subset, max_workers=max_workers, error_catching=error_catching)

if eval:
    # Go up to 'code' directory, then into mini_dev_main
    code_dir = Path(__file__).resolve().parent.parent.parent  # Up 2 levels to 'code'
    mini_dev_path = code_dir / 'mini_dev_main'

    sys.path.insert(0, str(mini_dev_path))

    from evaluation.run_eval import run_eval

    output_log_path = rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\sql_result\{agent_name}\{model_name}\MySQL.log'

    run_eval(predicated_path, output_log_path)