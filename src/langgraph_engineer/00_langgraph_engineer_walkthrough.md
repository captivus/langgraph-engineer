Certainly! I'll walk you through the "Langgraph Engineer" project, explaining each file and its purpose within the context of LangChain and LangGraph ecosystems.

1. `__init__.py`
This file is empty, which is common in Python projects. It marks the directory as a Python package, allowing you to import modules from this directory.

2. `agent.py`
This file defines the main structure of the LangGraph application.


```1:53:src/langgraph_engineer/agent.py
from typing import Literal

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import AIMessage


from langgraph_engineer.check import check
from langgraph_engineer.critique import critique
from langgraph_engineer.draft import draft_answer
from langgraph_engineer.gather_requirements import gather_requirements
from langgraph_engineer.state import AgentState, OutputState, GraphConfig



def route_critique(state: AgentState) -> Literal["draft_answer", END]:
    if state['accepted']:
        return END
    else:
        return "draft_answer"

def route_check(state: AgentState) -> Literal["critique", "draft_answer"]:
    if isinstance(state['messages'][-1], AIMessage):
        return "critique"
    else:
        return "draft_answer"


def route_start(state: AgentState) -> Literal["draft_answer", "gather_requirements"]:
    if state.get('requirements'):
        return "draft_answer"
    else:
        return "gather_requirements"


def route_gather(state: AgentState) -> Literal["draft_answer", END]:
    if state.get('requirements'):
        return "draft_answer"
    else:
        return END


# Define a new graph
workflow = StateGraph(AgentState, input=MessagesState, output=OutputState, config_schema=GraphConfig)
workflow.add_node(draft_answer)
workflow.add_node(gather_requirements)
workflow.add_node(critique)
workflow.add_node(check)
workflow.set_conditional_entry_point(route_start)
workflow.add_conditional_edges("gather_requirements", route_gather)
workflow.add_edge("draft_answer", "check")
workflow.add_conditional_edges("check", route_check)
workflow.add_conditional_edges("critique", route_critique)
graph = workflow.compile()
```


Key points:
- It imports various components from other files in the project.
- Defines routing functions that determine the flow of the graph based on the current state.
- Creates a `StateGraph` called `workflow`, which represents the entire agent's workflow.
- Adds nodes to the graph, corresponding to different functions like `draft_answer`, `gather_requirements`, `critique`, and `check`.
- Sets up conditional edges to control the flow between these nodes.
- Compiles the graph into an executable form.

3. `check.py`
This file contains logic to check and extract Python code from responses.


```1:30:src/langgraph_engineer/check.py
import re
from langgraph_engineer.state import AgentState


def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*(```|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


error_parsing = """Make sure your response contains a code block in the following format:

```python
...
```

When trying to parse out that code block, got this error: {error}"""


def check(state: AgentState):
    last_answer = state['messages'][-1]
    try:
        code_blocks = extract_python_code(last_answer.content)
    except Exception as e:
        return {"messages": [{"role": "user", "content": error_parsing.format(error=str(e))}]}
    if len(code_blocks) == 0:
        return {"messages": [{"role": "user", "content": error_parsing.format(error="Did not find a code block!")}]}
    if len(code_blocks) > 1:
        return {"messages": [{"role": "user", "content": error_parsing.format(error="Found multiple code blocks!")}]}
    return {"code": f"```python\n{code_blocks[0][0]}\n```"}
```


Key points:
- Defines a function to extract Python code blocks from text using regex.
- Implements error handling for code parsing.
- Returns extracted code or error messages as part of the state.

4. `critique.py`
This file handles the critique phase of the generated LangGraph application.


```1:60:src/langgraph_engineer/critique.py
from langgraph_engineer.loader import load_github_file
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel

critique_prompt = """You are tasked with critiquing a junior developers first attempt at building a LangGraph application. \
Here is a long unit test file for LangGraph. This should contain a lot (but possibly not all) \
relevant information on how to use LangGraph.

<unit_test_file>
{file}
</unit_test_file>

Based on the conversation below, attempt to critique the developer. If it seems like the written solution is fine, then call the `Accept` tool.

Do NOT critique the internal logic of the nodes too much - just make sure the flow (the nodes and edges) are correct and make sense. \
It's totally fine to use dummy LLMs or dummy retrieval steps."""


class Accept(BaseModel):
    logic: str
    accept: bool


def _swap_messages(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append({"role": "user", "content": m.content})
        else:
            new_messages.append({"role": "assistant", "content": m.content})
    return new_messages
...
def critique(state: AgentState, config):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
                   {"role": "user", "content": critique_prompt.format(file=file_contents)},
                   {"role": "assistant", "content": state.get('requirements')},

               ] + _swap_messages(state['messages'])
    model = _get_model(config, "openai", "critique_model").with_structured_output(Accept)
    response = model.invoke(messages)
    accepted = response.accept
    if accepted:
        return {
            "messages": [
                {"role": "user", "content": response.logic},
                {"role": "assistant", "content": "okay, sending to user"}],
            "accepted": True
        }
    else:
        return {
            "messages": [
                {"role": "user", "content": response.logic},
            ],
            "accepted": False
        }
```


Key points:
- Defines a prompt for critiquing a junior developer's LangGraph application.
- Uses a structured output (`Accept`) to determine if the solution is acceptable.
- Implements a function to critique the current state, potentially accepting or rejecting the solution.

5. `draft.py`
This file is responsible for drafting answers based on LangGraph functionality and user questions.


```1:42:src/langgraph_engineer/draft.py
from langgraph_engineer.loader import load_github_file
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState

prompt = """You are tasked with answering questions about LangGraph functionality and bugs.
Here is a long unit test file for LangGraph. This should contain a lot (but possibly not all) \
relevant information on how to use LangGraph.

<unit_test_file>
{file}
</unit_test_file>

Based on the information above, attempt to answer the user's questions. If you generate a code block, only \
generate a single code block - eg lump all the code together (rather than splitting up). \
You should encode helpful comments as part of that code block to understand what is going on. \
ALWAYS just generate the simplest possible example - don't make assumptions that make it more complicated. \
For "messages", these are a special object that looks like: {{"role": .., "content": ....}}

If users ask for a messages key, use MessagesState which comes with a built in `messages` key. \
You can import MessagesState from `langgraph.graph` and it is a TypedDict, so you can subclass it and add new keys to use as the graph state.

Make sure any generated graphs have at least one edge that leads to the END node - you need to define a stopping criteria!

You generate code using markdown python syntax, eg:

```python
...
```

Remember, only generate one of those code blocks!"""
...
def draft_answer(state: AgentState, config):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
        {"role": "system", "content": prompt.format(file=file_contents)},
                   {"role": "user", "content": state.get('requirements')}
    ] + state['messages']
    model = _get_model(config, "openai", "draft_model")
    response = model.invoke(messages)
    return {"messages": [response]}
```


Key points:
- Defines a prompt for answering questions about LangGraph.
- Implements a function to draft answers using the provided context and user questions.

6. `gather_requirements.py`
This file handles the gathering of requirements for building a LangGraph application.


```1:36:src/langgraph_engineer/gather_requirements.py
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from typing import TypedDict
from langchain_core.messages import RemoveMessage

gather_prompt = """You are tasked with helping build LangGraph applications. \
LangGraph is a framework for developing LLM applications. \
It represents agents as graphs. These graphs can contain cycles and often contain branching logic.

Your first job is to gather all the user requirements about the topology of the graph. \
You should have a clear sense of all the nodes of the graph/agent, and all the edges. 

You are conversing with a user. Ask as many follow up questions as necessary - but only ask ONE question at a time. \
Only gather information about the topology of the graph, not about the components (prompts, LLMs, vector DBs). \
If you have a good idea of what they are trying to build, call the `Build` tool with a detailed description.

Do not ask unnecessary questions! Do not ask them to confirm your understanding or the structure! The user will be able to \
correct you even after you call the Build tool, so just do enough to get an MVP."""


class Build(TypedDict):
    requirements: str


def gather_requirements(state: AgentState, config):
    messages = [
       {"role": "system", "content": gather_prompt}
   ] + state['messages']
    model = _get_model(config, "openai", "gather_model").bind_tools([Build])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": [response]}
    else:
        requirements = response.tool_calls[0]['args']['requirements']
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        return {"requirements": requirements, "messages": delete_messages}
```


Key points:
- Defines a prompt for gathering user requirements about the topology of the graph.
- Implements a function to interact with the user and collect requirements.
- Uses a structured output (`Build`) to signal when enough information has been gathered.

7. `loader.py`
This file provides functionality to load GitHub files with caching.


```1:40:src/langgraph_engineer/loader.py
import functools
from functools import lru_cache
import time
import requests


CACHE_DURATION = 24 * 60 * 60


def time_based_cache(seconds):
    def wrapper_cache(func):
        func = lru_cache(maxsize=None)(func)
        func.lifetime = seconds
        func.expiration = time.time() + func.lifetime

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


@time_based_cache(CACHE_DURATION)
def load_github_file(url):
    # Convert GitHub URL to raw content URL
    raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Send a GET request to the raw URL
    response = requests.get(raw_url)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        return f"Failed to load file. Status code: {response.status_code}"
```


Key points:
- Implements a time-based cache decorator to optimize file loading.
- Provides a function to load GitHub files, converting URLs to raw content URLs.

8. `model.py`
This file handles the selection and initialization of language models.


```1:12:src/langgraph_engineer/model.py
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def _get_model(config, default, key):
    model = config['configurable'].get(key, default)
    if model == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-2024-08-06")
    elif model == "anthropic":
        return ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")
    else:
        raise ValueError
```


Key points:
- Provides a function to get either an OpenAI or Anthropic chat model based on configuration.

9. `state.py`
This file defines the state structures used in the LangGraph application.


```1:16:src/langgraph_engineer/state.py
from langgraph.graph import MessagesState
from typing import TypedDict, Literal
class AgentState(MessagesState):
    requirements: str
    code: str
    accepted: bool


class OutputState(TypedDict):
    code: str


class GraphConfig(TypedDict):
    gather_model: Literal['openai', 'anthropic']
    draft_model: Literal['openai', 'anthropic']
    critique_model: Literal['openai', 'anthropic']
```


Key points:
- Defines `AgentState`, `OutputState`, and `GraphConfig` as TypedDict classes.
- These structures are used to maintain and pass state throughout the graph execution.

Overall, this project implements a LangGraph application that can:
1. Gather requirements from a user about a desired LangGraph application.
2. Draft an implementation based on those requirements.
3. Critique the implementation.
4. Check the generated code for correctness.
5. Iterate on the solution until it's acceptable.

The project leverages LangChain's components (like ChatOpenAI and ChatAnthropic) and LangGraph's `StateGraph` to create a flexible, stateful workflow for building LangGraph applications. It demonstrates how to use LangGraph to create complex, multi-step AI workflows with branching logic and state management.