import os
import json
import re
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=OPENAI_API_KEY
)

template = """
You are a recipe generator. Given a food name and number of people,
return the ingredients in JSON format:
[
  {{"name": "Ingredient", "amount": "value unit"}}
]
Food: {food_name}
People: {people_count}
"""


prompt = PromptTemplate(
    input_variables=["food_name", "people_count"],
    template=template
)

# Blinkit link generator
def get_blinkit_link(item: str):
    return f"https://blinkit.com/s/?q={item.replace(' ', '+')}"

# Extract JSON safely from GPT response
def extract_json(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else text

# Main recipe generator function
def generate_recipe(food_name: str, people_count: int):
    try:
        final_prompt = prompt.format(food_name=food_name, people_count=people_count)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Prompt formatting failed: {e}")

    response = llm.invoke(final_prompt)
    recipe_text = extract_json(response.content)

    try:
        ingredients = json.loads(recipe_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse recipe JSON")

    for ing in ingredients:
        ing["link"] = get_blinkit_link(ing["name"])

    return {"ingredients": ingredients}