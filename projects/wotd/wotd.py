from langgraph.graph import StateGraph, MessagesState, END
import requests
import openai
from datetime import datetime
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the state with a messages key
class WordOfTheDayState(MessagesState):
    date: str
    word: str
    definition: str
    examples: list

# Node: Date Input
def date_input(state: WordOfTheDayState) -> WordOfTheDayState:
    # Assume the date is provided in the state
    return {"date": state["date"]}

# Node: Word Generation
def word_generation(state: WordOfTheDayState) -> WordOfTheDayState:
    # Generate a random word based on the date
    random.seed(state["date"])
    word = random.choice(["apple", "banana", "cherry", "date", "elderberry"])
    return {"word": word}

# Node: Dictionary API Call
def dictionary_api_call(state: WordOfTheDayState) -> WordOfTheDayState:
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{state['word']}")
    if response.status_code == 200:
        definition = response.json()[0]['meanings'][0]['definitions'][0]['definition']
        return {"definition": definition}
    else:
        # Word not found, trigger word generation again
        return {"word": None}

# Node: OpenAI API Call for Examples
def openai_api_call(state: WordOfTheDayState) -> WordOfTheDayState:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": f"Generate two example sentences using the word '{state['word']}'."}]
    )
    examples = [msg['content'] for msg in response['choices'][0]['message']['content'].split('\n') if msg]
    return {"examples": examples}

# Node: Response Construction
def response_construction(state: WordOfTheDayState) -> WordOfTheDayState:
    response = f"Word of the Day: {state['word']}\nDefinition: {state['definition']}\nExamples:\n- {state['examples'][0]}\n- {state['examples'][1]}"
    return {"messages": [{"role": "system", "content": response}]}

# Node: Output
def output(state: WordOfTheDayState) -> WordOfTheDayState:
    # Output the final response
    print(state["messages"][0]["content"])
    return {}

# Define the graph
graph = StateGraph(WordOfTheDayState)
graph.add_node("date_input", date_input)
graph.add_node("word_generation", word_generation)
graph.add_node("dictionary_api_call", dictionary_api_call)
graph.add_node("openai_api_call", openai_api_call)
graph.add_node("response_construction", response_construction)
graph.add_node("output", output)

# Define the edges
graph.add_edge("date_input", "word_generation")
graph.add_edge("word_generation", "dictionary_api_call")
graph.add_conditional_edges("dictionary_api_call", lambda state: "word_generation" if state.get("word") is None else "openai_api_call")
graph.add_edge("openai_api_call", "response_construction")
graph.add_edge("response_construction", "output")
graph.add_edge("output", END)

# Compile the graph
app = graph.compile()

# Example invocation
print(app.invoke({"date": datetime.now().strftime("%Y-%m-%d")}))