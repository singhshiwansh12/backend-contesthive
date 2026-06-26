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

from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
from pgvector.sqlalchemy import Vector

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/contesthive")
ADMIN_PIN = os.getenv("ADMIN_PIN", "1234")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")



# ─── Database ─────────────────────────────────────────────────────────────────
# Fix for Neon: replace postgresql:// with postgresql+psycopg2://
SAFE_DB_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
engine = create_engine(
    SAFE_DB_URL,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=2,
)
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
    """Generate embedding using sentence-transformers (free, no API key)."""
    try:
        embedding = _model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
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
    """Semantic search using cosine similarity with pgvector with threshold."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = get_embedding(q)
    THRESHOLD = 0.7

    from sqlalchemy import text as sa_text
    rows = db.execute(
        sa_text("""
            SELECT id, (embedding <=> :emb) AS distance
            FROM solutions
            WHERE embedding IS NOT NULL
            AND (embedding <=> :emb) < :threshold
            ORDER BY distance ASC
            LIMIT 10
        """),
        {"emb": str(query_embedding), "threshold": THRESHOLD}
    ).fetchall()

    if not rows:
        return []

    ids = [row[0] for row in rows]
    solutions = db.query(Solution).filter(Solution.id.in_(ids)).all()
    sol_map = {s.id: s for s in solutions}
    return [solution_to_out(sol_map[i]) for i in ids if i in sol_map]


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


@app.get("/api/seed")
def seed_data(admin_pin: str = Query(...), db: Session = Depends(get_db)):
    verify_pin(admin_pin)

    existing = db.query(Solution).count()
    if existing > 0:
        return {"message": f"Already has {existing} solutions. Skipping."}

    SEEDS = [
        {"contest":"Weekly Contest 345","platform":"LeetCode","problem":"Two Sum","difficulty":"Easy","language":"C++","topics":["Array","Hash Table"],"code":"class Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {\n        unordered_map<int, int> mp;\n        for (int i = 0; i < nums.size(); i++) {\n            int complement = target - nums[i];\n            if (mp.find(complement) != mp.end()) return {mp[complement], i};\n            mp[nums[i]] = i;\n        }\n        return {};\n    }\n};","explanation":"Hash map approach for O(n) time complexity"},
        {"contest":"Biweekly Contest 110","platform":"LeetCode","problem":"Valid Parentheses","difficulty":"Easy","language":"C++","topics":["String","Stack"],"code":"class Solution {\npublic:\n    bool isValid(string s) {\n        stack<char> st;\n        for (char c : s) {\n            if (c=='('||c=='{'||c=='[') st.push(c);\n            else {\n                if (st.empty()) return false;\n                char top = st.top(); st.pop();\n                if (c==')' && top!='(') return false;\n                if (c=='}' && top!='{') return false;\n                if (c==']' && top!='[') return false;\n            }\n        }\n        return st.empty();\n    }\n};","explanation":"Stack to match brackets. O(n) time."},
        {"contest":"Weekly Contest 340","platform":"LeetCode","problem":"Best Time to Buy and Sell Stock","difficulty":"Easy","language":"Python","topics":["Array","Greedy"],"code":"class Solution:\n    def maxProfit(self, prices):\n        min_price = float('inf')\n        max_profit = 0\n        for price in prices:\n            min_price = min(min_price, price)\n            max_profit = max(max_profit, price - min_price)\n        return max_profit","explanation":"Track min price and max profit in single pass. O(n)."},
        {"contest":"Weekly Contest 346","platform":"LeetCode","problem":"Longest Substring Without Repeating Characters","difficulty":"Medium","language":"C++","topics":["String","Sliding Window"],"code":"class Solution {\npublic:\n    int lengthOfLongestSubstring(string s) {\n        unordered_set<char> st;\n        int left = 0, maxLen = 0;\n        for (int right = 0; right < s.length(); right++) {\n            while (st.find(s[right]) != st.end()) { st.erase(s[left]); left++; }\n            st.insert(s[right]);\n            maxLen = max(maxLen, right - left + 1);\n        }\n        return maxLen;\n    }\n};","explanation":"Sliding window with hash set for O(n) time."},
        {"contest":"Weekly Contest 350","platform":"LeetCode","problem":"3Sum","difficulty":"Medium","language":"C++","topics":["Array","Two Pointers","Sorting"],"code":"class Solution {\npublic:\n    vector<vector<int>> threeSum(vector<int>& nums) {\n        sort(nums.begin(), nums.end());\n        vector<vector<int>> res;\n        for (int i = 0; i < (int)nums.size()-2; i++) {\n            if (i>0 && nums[i]==nums[i-1]) continue;\n            int lo=i+1, hi=nums.size()-1;\n            while (lo<hi) {\n                int sum=nums[i]+nums[lo]+nums[hi];\n                if (sum==0) { res.push_back({nums[i],nums[lo],nums[hi]}); lo++; hi--; }\n                else if (sum<0) lo++; else hi--;\n            }\n        }\n        return res;\n    }\n};","explanation":"Sort + two pointers. O(n^2) time."},
        {"contest":"Weekly Contest 352","platform":"LeetCode","problem":"Coin Change","difficulty":"Medium","language":"Python","topics":["Dynamic Programming","Array"],"code":"class Solution:\n    def coinChange(self, coins, amount):\n        dp = [float('inf')] * (amount + 1)\n        dp[0] = 0\n        for coin in coins:\n            for x in range(coin, amount + 1):\n                dp[x] = min(dp[x], dp[x - coin] + 1)\n        return dp[amount] if dp[amount] != float('inf') else -1","explanation":"Bottom-up DP. O(amount * coins)."},
        {"contest":"Biweekly Contest 112","platform":"LeetCode","problem":"Number of Islands","difficulty":"Medium","language":"C++","topics":["DFS","Matrix"],"code":"class Solution {\npublic:\n    int numIslands(vector<vector<char>>& grid) {\n        int count = 0;\n        for (int i=0;i<(int)grid.size();i++)\n            for (int j=0;j<(int)grid[0].size();j++)\n                if (grid[i][j]=='1') { dfs(grid,i,j); count++; }\n        return count;\n    }\n    void dfs(vector<vector<char>>& g, int i, int j) {\n        if (i<0||i>=(int)g.size()||j<0||j>=(int)g[0].size()||g[i][j]!='1') return;\n        g[i][j]='0';\n        dfs(g,i+1,j); dfs(g,i-1,j); dfs(g,i,j+1); dfs(g,i,j-1);\n    }\n};","explanation":"DFS from each unvisited land cell. O(m*n)."},
        {"contest":"Weekly Contest 360","platform":"LeetCode","problem":"Median of Two Sorted Arrays","difficulty":"Hard","language":"C++","topics":["Binary Search","Divide and Conquer"],"code":"class Solution {\npublic:\n    double findMedianSortedArrays(vector<int>& A, vector<int>& B) {\n        if (A.size()>B.size()) swap(A,B);\n        int m=A.size(), n=B.size(), lo=0, hi=m;\n        while (lo<=hi) {\n            int i=(lo+hi)/2, j=(m+n+1)/2-i;\n            int maxLA=(i==0)?INT_MIN:A[i-1], minRA=(i==m)?INT_MAX:A[i];\n            int maxLB=(j==0)?INT_MIN:B[j-1], minRB=(j==n)?INT_MAX:B[j];\n            if (maxLA<=minRB && maxLB<=minRA) {\n                if ((m+n)%2==1) return max(maxLA,maxLB);\n                return (max(maxLA,maxLB)+min(minRA,minRB))/2.0;\n            } else if (maxLA>minRB) hi=i-1; else lo=i+1;\n        }\n        return 0;\n    }\n};","explanation":"Binary search on smaller array. O(log(min(m,n)))."},
        {"contest":"Weekly Contest 362","platform":"LeetCode","problem":"Trapping Rain Water","difficulty":"Hard","language":"C++","topics":["Array","Two Pointers"],"code":"class Solution {\npublic:\n    int trap(vector<int>& height) {\n        int lo=0, hi=height.size()-1, maxL=0, maxR=0, water=0;\n        while (lo<hi) {\n            if (height[lo]<height[hi]) {\n                if (height[lo]>=maxL) maxL=height[lo];\n                else water+=maxL-height[lo];\n                lo++;\n            } else {\n                if (height[hi]>=maxR) maxR=height[hi];\n                else water+=maxR-height[hi];\n                hi--;\n            }\n        }\n        return water;\n    }\n};","explanation":"Two pointers. O(n) time, O(1) space."},
        {"contest":"Codeforces Round 900 (Div. 2)","platform":"Codeforces","problem":"Sasha and the Beautiful Array","difficulty":"Easy","language":"C++","topics":["Greedy","Sorting"],"code":"#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n    int t; cin>>t;\n    while(t--) {\n        int n; cin>>n;\n        vector<int> a(n);\n        for(auto& x:a) cin>>x;\n        sort(a.begin(),a.end());\n        long long ans=0;\n        for(int i=1;i<n;i++) ans+=a[i]-a[i-1];\n        cout<<ans<<\"\\n\";\n    }\n}","explanation":"Sort array. Answer equals a[n-1]-a[0] via telescoping sum."},
    ]

    count = 0
    for s in SEEDS:
        embed_text = f"{s['problem']} {s['explanation']} {' '.join(s['topics'])}"
        embedding = get_embedding(embed_text)
        sol = Solution(
            contest=s["contest"], platform=s["platform"], problem=s["problem"],
            difficulty=s["difficulty"], language=s["language"],
            topics=json.dumps(s["topics"]), code=s["code"],
            explanation=s["explanation"], embedding=embedding,
        )
        db.add(sol)
        count += 1

    db.commit()
    return {"message": f"Seeded {count} solutions successfully!"}
