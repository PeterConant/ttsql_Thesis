from langchain.tools import tool, ToolRuntime
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from transformers import AutoTokenizer

from typing_extensions import Literal
from datetime import datetime
from textwrap import dedent
from pprint import pprint
from pathlib import Path
from tqdm import tqdm
import subprocess
import json
import time
import requests
import argparse
import ast

import os
from dotenv import load_dotenv

load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='hpc', help='local or hpc')

args = parser.parse_args()

llm = None
mini_dev_sql = None
database = SQLDatabase.from_uri("mysql+pymysql://readonly-agent:bird@localhost:3306/bird_mini_dev")

model_name = "Qwen3-4B"
model_path = "/home/012155624/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/"
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token='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')
max_completion_tokens=2048

if args.env == 'local':
    # Define LLM
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model="qwen3:1.7b", #  phi3, gemma3:12b, gpt-oss:20b, qwen3:1.7b,
        temperature=0,
        base_url="http://localhost:11434/"
    )
elif args.env == 'hpc':

    llm = ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model=model_path,
        max_completion_tokens=max_completion_tokens
    )

with open('mini_dev-main/mysql/mini_dev_mysql.json') as f:
    mini_dev_sql = json.load(f)



# Define tools
@tool
def get_schema() -> str:
    """Retrieves all available tables in database by running "SHOW TABLES;" statement.
    
    Args:
        No Args
    """
    global DB_ID
    query = f"SHOW TABLES;"

    tables = database.run(query)
    
    tables_list = ast.literal_eval(tables)

    database_info = []

    for table_tuple in tables_list:
        table_name = table_tuple[0]
        
        # Get schema
        schema_query = f"SHOW CREATE TABLE `{table_name}`;"
        schema = database.run(schema_query)
        
        # Get first 3 rows
        #sample_query = f"SELECT * FROM `{table_name}` LIMIT 3;"
        #sample_data = database.run(sample_query)
        
        # Format for LLM
        table_info = dedent(f"""
            Table: {table_name}
            Schema: {schema}
            ---
            """) #Sample Data (first 3 rows): {sample_data}
        database_info.append(table_info)

    # Combine all table information
    formatted_output = "\n".join(database_info)

    return formatted_output

@tool
def execute_query(query: str) -> str:
    """Execute a SQL SELECT query and return the results.
    
    Only SELECT statements are permitted for data safety.
    Returns formatted results or error messages.
    
    Args:
        query: The SQL SELECT query to execute
    """

    global QUERY

    # Strip whitespace and check query type
    query_stripped = query.strip()
    query_upper = query_stripped.upper()
    
    # Block non-SELECT queries
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
    if any(query_upper.startswith(keyword) for keyword in dangerous_keywords):
        return f"Error: Only SELECT queries are allowed. Detected forbidden operation."
    
    if not query_upper.startswith('SELECT'):
        return "Error: Query must start with SELECT."

    QUERY = query

    try:
        result = database.run(query_stripped)
        if not result or result.strip() == '':
            return "Query executed successfully but returned no results."
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}\nQuery was: {query_stripped}"
    
#tools = [execute_query]
#tools_by_name = {tool.name: tool for tool in tools}
#llm_with_tools = llm.bind_tools(tools)





# Nodes
class State(MessagesState):
    db_id: str
    query: str


def user_node(state: State):
    """Represents the user"""

    return {"messages":state["messages"]}



def get_schema_node(state: State):
    """Performs the tool call"""

    observation = get_schema.invoke(state["db_id"])
    result = ToolMessage(content=observation, tool_call_id="0000001")
    return {"messages":result}


# def check_and_call_llm(messages, max_input_tokens=6000):
#     # Count tokens in input
#     prompt = "\n".join([msg.content for msg in messages])
#     token_count = len(tokenizer.encode(prompt))
    
#     if token_count > max_input_tokens:
#         raise ValueError(f"Input too long: {token_count} tokens (max: {max_input_tokens})")
    
#     # Safe to call
#     response = llm.invoke(messages)
#     return response


class Generated_SQL(BaseModel):
    '''Generate a SQL statement for the user.'''
    justification: str = Field(description="A short explanation for the reasoning of the generated SQL.")
    sql: str = Field(description="The SQL Statement")
    

structured_sql_gen_llm = llm.with_structured_output(Generated_SQL,
                    strict=True,
                    include_raw=True
                )

def llm_SQLGenerator_call(state: State):
    """LLM decides whether to call a tool or not"""

    # system_message_content = dedent("""You are a helpful SQL generation agent designed to turn a user's question into an {dialect} statements.
    #                                 Given a user input and available tables schema's, gernate an SQL statement that answers the user's question.
    #                                 Test the statement with the available tool to make sure the syntax, names, and response are all correct.
    #                                 If the tool results comeback with errors, the error and run the statement again. 
    #                                 DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    #                                 database.

    #                                 When you have a response from the tool that answers the user's question without any additional steps, generate your final response.
    #                                 Your final response is the SQL statement, not a tool call. 
    #                                 """.format(dialect="MySQL"))
    
    system_message_content = dedent("""You are a helpful SQL generation agent designed to generate {dialect} statements.
                                Given a user input provided, use the retrieved database schema and values gernate an SQL statement that answers the user's question.

                                Your final response is the SQL statement. Follow the Output Format Instruction 
                                ### Output Format Instruction ###
                                - Return ONLY the raw SQL query.
                                - DO NOT include any explanations, introductory text, or natural language commentary before or after the query.
                                - DO NOT put the SQL inside a markdown code block (e.g., do not use ```sql ... ```).
                                - DO NOT include new lines "\\n"
                                """.format(dialect="MySQL"))

    # system_message_content = dedent("""You are a helpful SQL generation agent designed to generate {dialect} statements.
    #                             Given a user input provided, use the retrieved database schema and values gernate an SQL statement that answers the user's question. 
                                
    #                             Generate ONLY a raw {dialect} query. No explanations. No markdown. No newlines.""".format(dialect="MySQL"))


    #check_and_call_llm([SystemMessage(content=system_message_content)]
                #+ state["messages"], max_input_tokens=6000)
        response = structured_sql_gen_llm.invoke([SystemMessage(content=system_message_content)] + state["messages"][0:1] + state["messages"][2:])
    return {"messages":[response['raw']], 
            "sql": response['parsed'].sql, 
            "justification": response['parsed'].justification}



def execution_tool_node(state: State):
    """Performs the SQL statement execution tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages":result}



def should_continue(state: State) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop base upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop
    #return END
    return END





# Agent
def build_agent():
    # Build workflow
    agent_builder = StateGraph(State)

    # Add nodes
    agent_builder.add_node("user", user_node)
    agent_builder.add_node("get_schema", get_schema_node)
    agent_builder.add_node("llm_SQLGenerator_call", llm_SQLGenerator_call)
    #agent_builder.add_node("environment", execution_tool_node)
    


    # Add edges to connect nodes
    agent_builder.add_edge(START, "user")
    agent_builder.add_edge("user", "get_schema")
    agent_builder.add_edge("get_schema", "llm_SQLGenerator_call")
    agent_builder.add_edge("llm_SQLGenerator_call", END)

    return agent_builder.compile()



def run_minidev_MySQL(db_path=None):
    
    agent = build_agent()

    results = {}
    start_time = time.time()

    for i, entry in tqdm(enumerate(mini_dev_sql[:3]), total=len(mini_dev_sql[:3])):
        global DB_ID
        global QUERY
        DB_ID = entry['db_id']
        QUERY = "None"

        messages = [HumanMessage(content=dedent("""{question} You may find this informaiton helpful: {evidence}""".format(
                            question=entry['question'],
                            evidence=entry['evidence'],)))]
        try:
            response = agent.invoke({"messages": messages, 'db_id': DB_ID, 'query':"None"}) 
            results[str(i)] = f'{response['sql']}\t----- bird -----\t{entry["db_id"]}'
        except Exception as e:
            results[str(i)] = f"ERROR: {str(e)}"

    elapsed = time.time() - start_time
    elapsed_min = elapsed/60

    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev-main\sql_result\Baseline\results_baseline_{model_name}_{dt_now}.json', 'w') as f:
        json.dump(results, f, indent=4)
    with open(rf'C:\Users\peter\Documents\SJSU\Thesis\code\mini_dev-main\sql_result\Baseline\results_baseline_{model_name}_{dt_now}.log', 'w') as f:
        f.write(f'Latency: {elapsed}s / {elapsed_min}m\n')
        f.write(f'Max_tokens: {max_completion_tokens}')
        

run_minidev_MySQL()
