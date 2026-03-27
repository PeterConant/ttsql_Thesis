from tools import get_tables, get_tables_tool, get_table_schemas, get_table_schemas_tool, get_tables_semantic_search
from nodes import user_node, tool_node, gen_llm_call, agent_llm_call, should_continue

from langchain.tools import get_tables, get_tables_tool, get_table_schemas, get_table_schemas_tool, get_tables_semantic_search
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph,  MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
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



############################################################################
#           AGENTS
############################################################################

class Agent:
    def __init__(self, llm_path,max_completion_tokens):
        self.llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model=llm_path,
            temperature=0,
            max_completion_tokens=max_completion_tokens
        )
        self.llm_path = llm_path

class SemanticSearchAgent(Agent):

    def __init__(self,llm_path,max_completion_tokens):
        super().__init__(llm_path=llm_path,max_completion_tokens=max_completion_tokens)
        self.agent = self.build_agent()


    def build_agent():
        # Build workflow
        agent_builder = StateGraph(State)

        # Add nodes
        #agent_builder.add_node("user", user_node)
        agent_builder.add_node("generator_llm_call", gen_llm_call)
        agent_builder.add_node("environment", tool_node)

        # Add edges to connect nodes

        agent_builder.add_edge(START, "user")
        agent_builder.add_edge("user", "generator_llm_call")

        return agent_builder.compile() 

    def call_semeanticSearch_agent():
        return "sql"

class LLMSearchAgent(Agent):

    def __init__(self,llm_path,max_completion_tokens,secondar_llm_path,secondary_max_completion_tokens):
        super().__init__(llm_path=llm_path,max_completion_tokens=max_completion_tokens)
        self.agent = self.build_agent()

    def build_agent(self):
        # Build workflow
        agent_builder = StateGraph(State)

        # Add nodes
        agent_builder.add_node("user", user_node)
        agent_builder.add_node("agent_llm_call", agent_llm_call)
        agent_builder.add_node("environment", tool_node)

        # Add edges to connect nodes

        agent_builder.add_edge(START, "user")
        agent_builder.add_edge("user", "agent_llm_call")
        agent_builder.add_conditional_edges(
            "agent_llm_call",
            should_continue,
            {
                # Name returned by should_continue : Name of next node to visit
                "Action": "environment",
                END: END,
            },
        )
        agent_builder.add_edge("environment", "agent_llm_call")

        return agent_builder.compile()

    def call_agent(self, i, entry, error_catching: bool):
        call_start_time = time.time()

        table_list = get_tables_node()

        messages = [HumanMessage(content=dedent("""{question} Supporting evidence: {evidence}
                                                Here are the all tables available in the database:
                                                {table_list}""".format(
                        question=entry['question'],
                        evidence=entry['evidence'],
                        table_list=table_list)))]
        

        if(error_catching):
            try:
                response = self.agent.invoke({"messages": messages, "llm":self.llm, "llm_path":self.llm_path, "tables": [], "input_tokens": [], "output_tokens": []})
                result= f'{response['sql']}\t----- bird -----\t{entry["db_id"]}'

                call_elapsed = time.time() - call_start_time
                call_elapsed_min = call_elapsed/60

                meta_data = {'tables': response['tables'], 
                            'tokens': {'input_tokens': response['input_tokens'], 'output_tokens': response['output_tokens']},
                            'latency_s': call_elapsed,
                            'latency_m': call_elapsed_min,
                            'db_id': entry["db_id"]}
            except Exception as e:
                result = f"ERROR: {str(e)}"
                call_elapsed = time.time() - call_start_time
                call_elapsed_min = call_elapsed/60
                meta_data = {'tables': ['ERROR'], 
                    'tokens': {'tool_input_tokens': -1, 'tool_output_tokens':-1,
                        'gen_input_tokens': -1, 'gen_output_tokens': -1},
                    'latency_s': -1,
                    'latency_m': -1,
                    'db_id': entry["db_id"]}


        else: 
            response = self.agent.invoke({"messages": messages, "tables": [], "input_tokens": [], "output_tokens": []})
            result = f'{response['sql']}\t----- bird -----\t{entry["db_id"]}'

            call_elapsed = time.time() - call_start_time
            call_elapsed_min = call_elapsed/60

            meta_data = {'tables': response['tables'], 
                        'tokens': {'input_tokens': response['input_tokens'], 'output_tokens': response['output_tokens']},
                        'latency_s': call_elapsed,
                        'latency_m': call_elapsed_min,
                        'db_id': entry["db_id"]}
        
        return i, result, meta_data