import json

def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents

contents = load_json(r'mini_dev-main\sql_result\results_baseline_Meta-Llama-3-8B-Instruct.json')
cleaned_results = {}
for key, statement in contents.items():
    x = statement.replace('\n','')
    cleaned_results[key] = x

with open('results_baseline_Meta-Llama-3-8B-Instruct_cleaned.json', 'w') as f:
    json.dump(cleaned_results, f, indent=2)