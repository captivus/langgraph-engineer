# Hello, LangGraph Engineer!

# Getting Started
* Install LangGraph Studio
* Fork the LangGraph Engineer repo
* Setup your API keys in `.env`
* Download a good `.gitignore` file (GitHub has a good one)
* Rename `pyproject.toml` to `pyproject.toml.original`
* Run `poetry init`
* Install the dependencies noted in `pyproject.toml.original`
    *   "langgraph",
    *   "langchain_anthropic",
    *   "langchain_core",
    *   "langchain_openai"
* Run `poetry install`
* Run `poetry shell`
* Open LangGraph Engineer in LangGraph Studio

# TODO
* Add LangGraph, LangChain docs to Cursor
* Have Cursor explain the LangGraph Engineer codebase
    * Generate a walkthrough, file by file
* Have Cursor explain the LangGraph Engineer-generated code
* Have Cursor Composer rewrite the LangGraph Engineer-generated code for readability (e.g. keyword arguments, etc ...)
* Build `langgraph.json` for WOTD project & load into LangGraph Studio
    * This isn't necessary? Just add a `graphs` entry to the `langgraph.json` file?