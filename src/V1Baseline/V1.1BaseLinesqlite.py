from langchain.tools import tool, ToolRuntime
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM

from typing_extensions import Literal
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
parser.add_argument('--env', default='local', help='local or hpc')

args = parser.parse_args()

llm = None
mini_dev_sql = None
databases = None
DB_ID: str

# Define Database
def load_databases(base_path: str):
    """Load .sqlite databases"""
    databases = {}
    for db_file in Path(base_path).rglob("*.sqlite"):
        db_id = db_file.parent.name
        uri = f"sqlite:///{db_file.resolve()}"
        databases[db_id] = SQLDatabase.from_uri(uri)
    return databases

def load_mysql_database():
    db = SQLDatabase.from_uri("mysql+pymysql://readonly-user:password@localhost:3306/BIRD")

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
        model="/home/012155624/gpt-oss-20b"
    )

with open('mini_dev/mini_dev_data/mini_dev_sqlite.json') as f:
    mini_dev_sql = json.load(f)

databases = load_databases("C:/Users/peter/Documents/SJSU/Thesis/code/mini_dev/minidev_0703/minidev/MINIDEV/dev_databases/")




# Define tools
@tool
def get_schema() -> str:
    """Retrieves all available tables in database by running "SHOW TABLES;" statement.
    
    Args:
        No Args
    """
    global DB_ID
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"

    db = databases[DB_ID]
    tables = db.run(query)
    
    tables_list = ast.literal_eval(tables)

    database_info = []

    for table_tuple in tables_list:
        table_name = table_tuple[0]
        
        # Get schema
        schema_query = f"PRAGMA table_info(\"{table_name}\");"
        schema = db.run(schema_query)
        
        # Get first 3 rows
        sample_query = f"SELECT * FROM \"{table_name}\" LIMIT 3;"
        sample_data = db.run(sample_query)
        
        # Format for LLM
        table_info = dedent(f"""
            Table: {table_name}
            Schema: {schema}
            Sample Data (first 3 rows): {sample_data}
            ---
            """)
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
    global DB_ID
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
    
    db = databases[DB_ID]

    QUERY = query

    try:
        result = db.run(query_stripped)
        if not result or result.strip() == '':
            return "Query executed successfully but returned no results."
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}\nQuery was: {query_stripped}"
    
tools = [execute_query]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)





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



def llm_SQLGenerator_call(state: State):
    """LLM decides whether to call a tool or not"""

    system_message_content = dedent("""You are a helpful SQL generation agent designed to turn a user's question into an SQLite statements.
                                    Given a user input and available tables schema's, gernate an SQL statement that answers the user's question.
                                    Test the statement with the available tool to make sure the syntax, names, and response are all correct.
                                    If the tool results comeback with errors, the error and run the statement again. 
                                    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
                                    database.

                                    When you have a response from the tool that answers the user's question without any additional steps, generate your final response.
                                    Your final response is the SQL statement, not a tool call. 
                                    """.format(dialect="SQLIite"))
    
    system_message_content = dedent("""You are a helpful SQL generation agent designed to turn a user's question into an SQLite statements.
                                Given a user input and available tables schema's, gernate an SQL statement that answers the user's question.

                                When you have a response from the tool that answers the user's question without any additional steps, generate your final response.
                                Your final response is the SQL statement, not a tool call. 
                                ### Output Format Instruction ###
                                - Return ONLY the raw SQL query.
                                - DO NOT include any explanations, introductory text, or natural language commentary before or after the query.
                                - DO NOT put the SQL inside a markdown code block (e.g., do not use ```sql ... ```).
                                - DO NOT include new lines "\\n"
                                """.format(dialect="SQLIite"))

    return {
        "messages":[
            #llm_with_tools.invoke(
            llm.invoke(
                [
                    SystemMessage(
                        content=system_message_content
                    )
                ]
                + state["messages"]
            )
        ]
    }



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
    agent_builder.add_node("environment", execution_tool_node)
    


    # Add edges to connect nodes
    agent_builder.add_edge(START, "user")
    agent_builder.add_edge("user", "get_schema")
    agent_builder.add_edge("get_schema", "llm_SQLGenerator_call")
    agent_builder.add_conditional_edges(
        "llm_SQLGenerator_call",
        should_continue,
        {
            # Name returned by should_continue : Name of next node to visit
            "Action" : "environment",
            END: END
        }
    )
    agent_builder.add_edge("environment", "llm_SQLGenerator_call")

    return agent_builder.compile()



def run_minidev_sqlite(db_path=None):

    agent = build_agent()

    results = {}

    for i, entry in tqdm(enumerate(mini_dev_sql), total=len(mini_dev_sql)):
        global DB_ID
        global QUERY
        DB_ID = entry['db_id']
        QUERY = "None"

        messages = [HumanMessage(content=dedent("""{question} You may find this informaiton helpful: {evidence}""".format(
                            question=entry['question'],
                            evidence=entry['evidence'],)))]
        response = agent.invoke({"messages": messages, 'db_id': DB_ID, 'query':"None"})
        last_response = f'{response["messages"][-1].content}\t----- bird -----\t{entry["db_id"]}' 
        results[str(i)] = last_response
        
    

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        

run_minidev_sqlite()
