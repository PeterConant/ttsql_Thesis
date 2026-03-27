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

database = SQLDatabase.from_uri("mysql+pymysql://readonly-agent:bird@localhost:3306/bird_mini_dev", max_string_length = 3000)


# Get Tables
def get_tables():
    """Performs the tool call"""

    query = f"SHOW TABLES;"

    tables = database.run(query)
    tables = ast.literal_eval(tables)
    _tables = []
    for table in tables:
        _tables.append(table[0])
    tables = _tables
    return tables

@tool
def get_tables_tool():
    """Performs the tool call"""

    tables = get_tables()
    response = "SHOW TABLES from bird_mini_dev:\n" + str(tables)
    result = ToolMessage(content=response, tool_call_id="get_tables_node")
    return {"messages":result}


# Get Schemas
def get_table_schemas(tables:list[str]):
    return database.get_table_info_no_throw(tables)

# Tools
@tool
def get_table_schemas_tool(table_list: list[str]) -> str:
    """Get detailed information about a list of tables including columns, types, constraints, and the top 3 rows.
    
    Args:
        table_names: List of table name strings

    Returns:
        Formatted string containing schema and sample data for each table
    """

    return get_table_schemas(table_list)



################## EMBEDDINGS SIMILARITY #########################

file_name = 'tableName_createTable_valueExample'
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.load(f'embeddings/{file_name}.npy')
with open(f'embeddings/{file_name}.json', 'r') as f:
    table_metadata = json.load(f)

def get_tables_semantic_search(query, k, similarity_threshold=0.0):
    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k)

    for rank, idx in enumerate(I[0]):
        score = D[0][rank]
        if score < similarity_threshold:
            continue

    response = "A similarity search between the user question and possible databases found these tables the most relavent to the user question: \n" + tables
    result = ToolMessage(content=response, tool_call_id="get_tables_node")
    return result, similarity_measures