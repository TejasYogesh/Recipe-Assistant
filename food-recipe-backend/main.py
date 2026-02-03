from fastapi import FastAPI
from pydantic import BaseModel
from recipe_agent import generate_recipe
from qa_agent import answer_question
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # 👈 your Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecipeRequest(BaseModel):
    food_name: str
    people_count: int

class QuestionRequest(BaseModel):
    question: str
    context: str

@app.post("/generate-recipe")
async def generate_recipe_endpoint(req: RecipeRequest):
    recipe = generate_recipe(req.food_name, req.people_count)
    return {"recipe": recipe}

@app.post("/ask-question")
async def ask_question_endpoint(req: QuestionRequest):
    answer = answer_question(req.question, req.context)
    return {"answer": answer}