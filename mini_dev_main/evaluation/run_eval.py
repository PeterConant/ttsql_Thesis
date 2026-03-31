import subprocess
import sys

def run_eval(predicted_sql_path,output_log_path):

    #predicted_sql_path = rf"C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev_main\sql_result\Baseline\results_Qwen3-4B_defualt_max-model-len_20260313_174733.json"
    #output_log_path = "C:\\Users\\peter\\Documents\\SJSU\\Thesis\\code\\mini_dev_main\\sql_result\\Baseline\\MySQL.log" # Baseline\\MySQL.log


    # Define scripts with their arguments
    scripts = [
        [r'mini_dev_main\evaluation\evaluation_ex.py', '--predicted_sql_path', predicted_sql_path, '--output_log_path', output_log_path],
        [r'mini_dev_main\evaluation\evaluation_f1.py', '--predicted_sql_path', predicted_sql_path, '--output_log_path', output_log_path],  # No arguments
    ]

    for script_with_args in scripts:
        script_name = script_with_args[0]
        print(f"\n--- Running {script_name} ---")
        
        # Build the command: [python executable, script, arg1, arg2, ...]
        command = [sys.executable] + script_with_args
        
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Error in {script_name}:")
            print(result.stderr)
            # Optionally stop on error:
            # sys.exit(result.returncode)

    print("\nAll scripts completed!")