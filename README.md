# ContestHive Backend 🚀

FastAPI backend with PostgreSQL (pgvector) and Gemini AI for semantic search.

## Tech Stack
- **FastAPI** — REST API framework
- **PostgreSQL + pgvector** — Vector similarity search
- **Gemini AI** — Embeddings (`models/embedding-001`)
- **SQLAlchemy** — ORM

## Local Setup

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/backend-contesthive.git
cd backend-contesthive

# 2. Create virtualenv
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Setup env
cp .env.example .env
# Edit .env with your values

# 5. Run
uvicorn main:app --reload
```

## Seed Database

```bash
python seed.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/solutions` | List solutions (with filters) |
| GET | `/api/solutions/ai-search?q=...` | Semantic AI search |
| GET | `/api/solutions/{id}` | Get single solution |
| POST | `/api/solutions` | Create solution (PIN required) |
| PUT | `/api/solutions/{id}` | Update solution (PIN required) |
| DELETE | `/api/solutions/{id}?admin_pin=...` | Delete solution (PIN required) |

## Query Filters (GET /api/solutions)

```
?platform=LeetCode&difficulty=Easy&language=C++&search=two sum
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `ADMIN_PIN` | PIN for admin operations |
| `GEMINI_API_KEY` | Google Gemini API key |
| `FRONTEND_URL` | Frontend URL for CORS |

## Render Deployment

1. Create a PostgreSQL DB on Render → copy `DATABASE_URL`
2. New Web Service → connect this repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Add env vars
6. After deploy → Shell → `python seed.py`
