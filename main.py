import os
import json
from contextlib import asynccontextmanager
from typing import Optional, List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import ARRAY
from pydantic import BaseModel

import google.generativeai as genai
from pgvector.sqlalchemy import Vector

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/contesthive")
ADMIN_PIN = os.getenv("ADMIN_PIN", "1234")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

genai.configure(api_key=GEMINI_API_KEY)

# ─── Database ─────────────────────────────────────────────────────────────────
# FORCE pg8000 driver to avoid psycopg2/C++ build errors on Windows
SAFE_DB_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://")
SAFE_DB_URL = SAFE_DB_URL.replace("postgres://", "postgresql+pg8000://") 
SAFE_DB_URL = SAFE_DB_URL.replace("postgresql+psycopg2://", "postgresql+pg8000://")

engine = create_engine(SAFE_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ─── Models ───────────────────────────────────────────────────────────────────
class Solution(Base):
    __tablename__ = "solutions"

    id = Column(Integer, primary_key=True, index=True)
    contest = Column(String(200), nullable=False)
    platform = Column(String(100), nullable=False)
    problem = Column(String(300), nullable=False)
    difficulty = Column(String(50), nullable=False)
    language = Column(String(50), nullable=False)
    topics = Column(Text, default="[]")          # JSON string
    code = Column(Text, nullable=False)
    explanation = Column(Text, default="")
    embedding = Column(Vector(768), nullable=True)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class SolutionCreate(BaseModel):
    contest: str
    platform: str
    problem: str
    difficulty: str
    language: str
    topics: List[str] = []
    code: str
    explanation: str = ""
    admin_pin: str

class SolutionUpdate(BaseModel):
    contest: Optional[str] = None
    platform: Optional[str] = None
    problem: Optional[str] = None
    difficulty: Optional[str] = None
    language: Optional[str] = None
    topics: Optional[List[str]] = None
    code: Optional[str] = None
    explanation: Optional[str] = None
    admin_pin: str

class SolutionOut(BaseModel):
    id: int
    contest: str
    platform: str
    problem: str
    difficulty: str
    language: str
    topics: List[str]
    code: str
    explanation: str

    class Config:
        from_attributes = True


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_pin(pin: str):
    if pin != ADMIN_PIN:
        raise HTTPException(status_code=401, detail="Invalid admin PIN")

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Gemini embedding model."""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 768

def solution_to_out(sol: Solution) -> dict:
    try:
        topics = json.loads(sol.topics) if sol.topics else []
    except Exception:
        topics = []
    return {
        "id": sol.id,
        "contest": sol.contest,
        "platform": sol.platform,
        "problem": sol.problem,
        "difficulty": sol.difficulty,
        "language": sol.language,
        "topics": topics,
        "code": sol.code,
        "explanation": sol.explanation,
    }


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create pgvector extension and tables on startup
    with engine.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        conn.commit()
    Base.metadata.create_all(bind=engine)
    yield


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="ContestHive API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "ok", "message": "ContestHive API is running 🚀"}


@app.get("/api/solutions")
def get_solutions(
    platform: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    language: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(Solution)

    if platform:
        query = query.filter(Solution.platform.ilike(f"%{platform}%"))
    if difficulty:
        query = query.filter(Solution.difficulty.ilike(f"%{difficulty}%"))
    if language:
        query = query.filter(Solution.language.ilike(f"%{language}%"))
    if search:
        query = query.filter(
            Solution.problem.ilike(f"%{search}%")
            | Solution.contest.ilike(f"%{search}%")
            | Solution.explanation.ilike(f"%{search}%")
        )

    solutions = query.all()
    return [solution_to_out(s) for s in solutions]


@app.get("/api/solutions/ai-search")
def ai_search(q: str = Query(..., description="Search query"), db: Session = Depends(get_db)):
    """Semantic search using cosine similarity with pgvector."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = get_embedding(q)

    # Cosine similarity search via pgvector operator <=>
    results = (
        db.query(Solution)
        .filter(Solution.embedding.isnot(None))
        .order_by(Solution.embedding.op("<=>")(query_embedding))
        .limit(10)
        .all()
    )

    return [solution_to_out(s) for s in results]


@app.get("/api/solutions/{solution_id}")
def get_solution(solution_id: int, db: Session = Depends(get_db)):
    sol = db.query(Solution).filter(Solution.id == solution_id).first()
    if not sol:
        raise HTTPException(status_code=404, detail="Solution not found")
    return solution_to_out(sol)


@app.post("/api/solutions", status_code=201)
def create_solution(data: SolutionCreate, db: Session = Depends(get_db)):
    verify_pin(data.admin_pin)

    embed_text = f"{data.problem} {data.explanation} {' '.join(data.topics)}"
    embedding = get_embedding(embed_text)

    sol = Solution(
        contest=data.contest,
        platform=data.platform,
        problem=data.problem,
        difficulty=data.difficulty,
        language=data.language,
        topics=json.dumps(data.topics),
        code=data.code,
        explanation=data.explanation,
        embedding=embedding,
    )
    db.add(sol)
    db.commit()
    db.refresh(sol)
    return solution_to_out(sol)


@app.put("/api/solutions/{solution_id}")
def update_solution(solution_id: int, data: SolutionUpdate, db: Session = Depends(get_db)):
    verify_pin(data.admin_pin)

    sol = db.query(Solution).filter(Solution.id == solution_id).first()
    if not sol:
        raise HTTPException(status_code=404, detail="Solution not found")

    if data.contest is not None:
        sol.contest = data.contest
    if data.platform is not None:
        sol.platform = data.platform
    if data.problem is not None:
        sol.problem = data.problem
    if data.difficulty is not None:
        sol.difficulty = data.difficulty
    if data.language is not None:
        sol.language = data.language
    if data.topics is not None:
        sol.topics = json.dumps(data.topics)
    if data.code is not None:
        sol.code = data.code
    if data.explanation is not None:
        sol.explanation = data.explanation

    # Regenerate embedding
    try:
        topics_list = json.loads(sol.topics) if sol.topics else []
    except Exception:
        topics_list = []
    embed_text = f"{sol.problem} {sol.explanation} {' '.join(topics_list)}"
    sol.embedding = get_embedding(embed_text)

    db.commit()
    db.refresh(sol)
    return solution_to_out(sol)


@app.delete("/api/solutions/{solution_id}")
def delete_solution(
    solution_id: int,
    admin_pin: str = Query(...),
    db: Session = Depends(get_db),
):
    verify_pin(admin_pin)

    sol = db.query(Solution).filter(Solution.id == solution_id).first()
    if not sol:
        raise HTTPException(status_code=404, detail="Solution not found")

    db.delete(sol)
    db.commit()
    return {"message": f"Solution {solution_id} deleted successfully"}
