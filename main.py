from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import os
 
import password_analyzer as pa
 
# ── Resolve directory where app.py lives ─────────────────────────────────────
# This makes the app work no matter which directory you run uvicorn from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PassGuard API",
    description="Intelligent password strength analyzer with user-info awareness",
    version="2.0.0",
)
 
# Allow all origins during development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Load model and data at startup ───────────────────────────────────────────
ML_MODEL = None  # will be set in startup_event
 
@app.on_event("startup")
async def startup_event():
    global ML_MODEL
 
    data_dir   = os.path.join(BASE_DIR, 'data')
    rules_path = os.path.join(BASE_DIR, 'rule_set.pkl')
    markov_path= os.path.join(BASE_DIR, 'markov_model.json')
    model_path = os.path.join(BASE_DIR, 'model.pkl')
 
    print("Loading wordlists...")
    pa.load_wordlists(data_dir)
 
    print("Loading rule set...")
    pa.load_rule_set(rules_path)
 
    print("Loading Markov model...")
    pa.load_markov_model(markov_path)
 
    print("Loading ML model...")
    try:
        ML_MODEL = joblib.load(model_path)
        print("✅ model.pkl loaded.")
    except FileNotFoundError:
        ML_MODEL = None
        print("⚠️  model.pkl not found — ML risk will show 'unknown'. Run train_model.py first.")
 
# ── Serve index.html from templates/ folder ──────────────────────────────────
@app.get("/")
async def root():
    # Look in templates/ subfolder first, then fall back to BASE_DIR
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    if not os.path.exists(index_path):
        index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(
        status_code=404,
        detail=f"index.html not found. Looked in: {os.path.join(BASE_DIR, 'templates')} and {BASE_DIR}"
    )
 
# ═══════════════════════════════════════════════════════════════════════════
# Pydantic request / response models
# ═══════════════════════════════════════════════════════════════════════════
 
class UserInfo(BaseModel):
    """All fields optional — pass only what you have."""
    first_name: Optional[str] = None
    last_name:  Optional[str] = None
    username:   Optional[str] = None
    email:      Optional[str] = None
    birthdate:  Optional[str] = None   # "YYYY-MM-DD" or "DD/MM/YYYY"
    pet_name:   Optional[str] = None
    city:       Optional[str] = None
    company:    Optional[str] = None
    phone:      Optional[str] = None
 
 
class CheckRequest(BaseModel):
    password:   str                    = Field(..., min_length=1, example="Sudha1484@max")
    user_info:  Optional[UserInfo]     = Field(None, description="Personal info to check against password")
    hash_type:  Optional[str]          = Field("default", example="bcrypt")
    gpu_count:  Optional[int]          = Field(1, ge=1, le=1000)
 
 
class CheckResponse(BaseModel):
    score:              int
    entropy:            float
    crack_time:         str
    crack_time_advanced:str
    ml_risk:            str
    ml_confidence:      float
    issues:             list[str]
    suggestions:        list[str]
    user_info_matches:  list[str]
 
 
class SuggestRequest(BaseModel):
    length: Optional[int] = Field(16, ge=8,  le=64)
    count:  Optional[int] = Field(5,  ge=1,  le=20)
 
 
class SuggestFromUserRequest(BaseModel):
    password: str               = Field(..., min_length=1)
    count:    Optional[int]     = Field(5, ge=1, le=20)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════
 
@app.post("/check", response_model=CheckResponse, summary="Analyze password strength")
async def check(body: CheckRequest):
    """
    Full password analysis.
 
    - Checks entropy, patterns, dictionary hits, breach status
    - Detects personal info tokens (name, birthdate, pet, phone…)
    - Returns ML risk label + overall score
    - Penalises passwords that contain user's personal information
 
    **Example request:**
    ```json
    {
        "password": "Sudha1484@max",
        "user_info": {
            "first_name": "sudha",
            "last_name":  "raju",
            "birthdate":  "2005-06-01",
            "pet_name":   "max",
            "phone":      "07299961484"
        }
    }
    ```
    """
    # Convert Pydantic model → plain dict, strip None values
    user_info_dict = None
    if body.user_info:
        user_info_dict = {
            k: v for k, v in body.user_info.model_dump().items()
            if v is not None and str(v).strip()
        }
        if not user_info_dict:
            user_info_dict = None
 
    report = pa.analyze_password(
        body.password,
        ml_model=ML_MODEL,
        hash_type=body.hash_type or "default",
        gpu_count=body.gpu_count or 1,
        user_info=user_info_dict,
    )
 
    return report
 
 
@app.post("/suggest", response_model=list[str], summary="Generate strong passwords")
async def suggest(body: SuggestRequest):
    """
    Generate N cryptographically random strong passwords.
    All generated passwords pass the 4-class character requirement
    (lowercase, uppercase, digit, symbol) and meet the requested length.
    """
    passwords = []
    seen      = set()
    attempts  = 0
 
    while len(passwords) < body.count and attempts < body.count * 20:
        attempts += 1
        pwd = pa.generate_random_password(body.length)
        if pwd not in seen:
            seen.add(pwd)
            passwords.append(pwd)
 
    return passwords
 
 
@app.post("/suggest_from_user", response_model=list[str], summary="Improve a password")
async def suggest_from_user(body: SuggestFromUserRequest):
    """
    Return improved variants of the user's current password.
    Each variant is strengthened by adding missing character classes
    and increasing length if below minimum threshold.
    """
    suggestions = pa.generate_suggestions_from_user(body.password, count=body.count)
    return suggestions
 
 
# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check")
async def health():
    templates_index = os.path.join(BASE_DIR, "templates", "index.html")
    root_index      = os.path.join(BASE_DIR, "index.html")
    return {
        "status":            "ok",
        "base_dir":          BASE_DIR,
        "index_html":        os.path.exists(templates_index) or os.path.exists(root_index),
        "index_html_path":   templates_index if os.path.exists(templates_index) else root_index,
        "ml_model":          "loaded" if ML_MODEL is not None else "not loaded",
        "wordlists":         len(pa.ALL_WORDS),
        "rule_set":          len(pa.RULE_WORDS),
        "markov":            "loaded" if pa.MARKOV_MODEL is not None else "not loaded",
    }
 
 
# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)