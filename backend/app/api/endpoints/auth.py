from datetime import datetime
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File, Form
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from passlib.context import CryptContext
from app.models.db import get_db, COL_USERS
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
import os, uuid, jwt
import secrets

router = APIRouter(prefix="/auth", tags=["auth"])

# Use Argon2 instead of bcrypt
pwd_ctx = CryptContext(schemes=["argon2"], deprecated="auto")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"
from app.config import N8N_WEBHOOK_URL
ADMIN_BOOTSTRAP_TOKEN = os.getenv("ADMIN_BOOTSTRAP_TOKEN", None)

# Ensure uploads dir exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ApproveRequest(BaseModel):
    user_id: str = Field(...)
    approve: bool = Field(default=True)

def _hash(pw: str) -> str:
  # Argon2 supports long passwords; no truncation needed
  return pwd_ctx.hash(pw or "")

def _verify(pw: str, hashed: str) -> bool:
  try:
    return pwd_ctx.verify(pw or "", hashed)
  except Exception:
    return False

async def _find_user(db: AsyncIOMotorDatabase, email: str):
    return await db[COL_USERS].find_one({"email": email.lower().strip()})

def _jwt_for(user) -> str:
    payload = {"sub": str(user["_id"]), "role": user.get("role", "doctor"), "exp": int(datetime.utcnow().timestamp()) + 60 * 60 * 24}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def decode_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        return None

async def require_doctor(authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    payload = decode_jwt(token)
    if not payload or payload.get("role") != "doctor":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return payload

async def require_admin(authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    payload = decode_jwt(token)
    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return payload

@router.post("/signup")
async def signup(
    db: AsyncIOMotorDatabase = Depends(get_db),
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: EmailStr = Form(...),
    phone: str = Form(...),
    password: str = Form(...),
    credential_file: UploadFile | None = File(None),
):
    # unique email
    if await _find_user(db, str(email)):
        raise HTTPException(status_code=409, detail="Email already registered")

    # save credential file (optional)
    stored_url = None
    if credential_file:
        ext = os.path.splitext(credential_file.filename or "")[1].lower()
        if ext not in [".png", ".jpg", ".jpeg", ".pdf", ".webp"]:
            ext = ".dat"
        fname = f"{uuid.uuid4().hex}{ext}"
        
        # Create certificate subdirectory
        cert_dir = os.path.join(UPLOAD_DIR, "certificate")
        os.makedirs(cert_dir, exist_ok=True)
        
        stored_path = os.path.join(cert_dir, fname)
        try:
            contents = await credential_file.read()
            with open(stored_path, "wb") as f:
                f.write(contents)
            stored_url = f"/uploads/certificate/{fname}"
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to save certificate file")


    doc = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "email": str(email).lower(),
        "phone": phone.strip(),
        "password_hash": _hash(password),
        "role": "doctor",
        "status": "pending",  
        "credential_file": {
            "original": credential_file.filename if credential_file else None,
            "stored_url": stored_url,
            "content_type": credential_file.content_type if credential_file else None,
        },
        "created_at": datetime.utcnow(),
    }
    res = await db[COL_USERS].insert_one(doc)
    return {"id": str(res.inserted_id), "message": "Signup submitted. You will receive an email after verification.", "credential_file_url": stored_url}

@router.post("/login")
async def login(
    db: AsyncIOMotorDatabase = Depends(get_db),
    email: EmailStr = Form(...),
    password: str = Form(...),
):
    user = await _find_user(db, str(email))
    if not user or not _verify(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.get("status") != "approved":
        raise HTTPException(status_code=403, detail="Account not approved yet")

    token = _jwt_for(user)
    return {"access_token": token, "token_type": "bearer"}

# --- Admin Endpoints ---
@router.get("/admin/list")
async def admin_list_users(
    db: AsyncIOMotorDatabase = Depends(get_db),
    _admin = Depends(require_admin),
):
    # Only list doctors, exclude admins
    cur = db[COL_USERS].find({"role": "doctor"}, {"password_hash": 0})
    users = []
    async for u in cur:
        u["_id"] = str(u["_id"])
        if u.get("created_at"):
            try:
                u["created_at"] = u["created_at"].isoformat()
            except Exception:
                pass
        users.append(u)
    return {"users": users}

@router.post("/admin/approve")
async def admin_approve_user(
    req: ApproveRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    _admin = Depends(require_admin),
):
    try:
        oid = ObjectId(req.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")
    user = await db[COL_USERS].find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_status = "approved" if req.approve else "rejected"
    await db[COL_USERS].update_one({"_id": oid}, {"$set": {"status": new_status, "approved_at": datetime.utcnow()}})
  
    if N8N_WEBHOOK_URL and new_status == "approved":
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(N8N_WEBHOOK_URL, json={"type": "approval","name":user.get("first_name"), "email": user.get("email"), "user_id": req.user_id})
        except Exception:
            pass
    return {"ok": True, "status": new_status}

class AdminDeleteRequest(BaseModel):
    user_id: str

@router.post("/admin/delete")
async def admin_delete_user(
    req: AdminDeleteRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    _admin = Depends(require_admin),
):
    try:
        oid = ObjectId(req.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")
    res = await db[COL_USERS].delete_one({"_id": oid})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"ok": True}


class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

COL_RESETS = "password_resets"

@router.post("/password/reset/request")
async def request_password_reset(
    body: PasswordResetRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    user = await _find_user(db, str(body.email))
    if not user:
        # do not reveal whether email exists
        return {"ok": True}
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=2)
    await db[COL_RESETS].insert_one({
        "user_id": user["_id"],
        "email": user["email"],
        "token": token,
        "expires_at": expires_at,
        "used": False,
        "created_at": datetime.utcnow(),
    })
    # Notify via n8n webhook if configured
    if N8N_WEBHOOK_URL:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Provide frontend URL hint for n8n to build links (flat payload)
                url = "http://localhost:5173"
                if not url or not url.startswith("http"):
                    url = "http://localhost:5173"
                await session.post(N8N_WEBHOOK_URL, json={
                    "type": "reset_request",
                    "email": user.get("email"),
                    "token": token,
                    "URL": url
                })
        except Exception:
            pass
            pass
    return {"ok": True}

# --- Authenticated Password Change ---
class PasswordChangeBody(BaseModel):
    old_password: str
    new_password: str

@router.post("/password/change")
async def change_password(
    body: PasswordChangeBody,
    db: AsyncIOMotorDatabase = Depends(get_db),
    user = Depends(require_doctor),
):
    oid = ObjectId(user["sub"])
    doc = await db[COL_USERS].find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    if not _verify(body.old_password, doc.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Old password incorrect")
    await db[COL_USERS].update_one({"_id": oid}, {"$set": {"password_hash": _hash(body.new_password)}})
    return {"ok": True}

@router.post("/password/reset/confirm")
async def confirm_password_reset(
    body: PasswordResetConfirm,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    rec = await db[COL_RESETS].find_one({"token": body.token, "used": False})
    if not rec:
        raise HTTPException(status_code=400, detail="Invalid token")
    if rec.get("expires_at") and rec["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Token expired")
    await db[COL_USERS].update_one({"_id": rec["user_id"]}, {"$set": {"password_hash": _hash(body.new_password)}})
    await db[COL_RESETS].update_one({"_id": rec["_id"]}, {"$set": {"used": True, "used_at": datetime.utcnow()}})
    return {"ok": True}

class AdminBootstrapBody(BaseModel):
    email: EmailStr
    password: str
    token: str

@router.post("/admin/bootstrap")
async def admin_bootstrap(
    body: AdminBootstrapBody,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    if not ADMIN_BOOTSTRAP_TOKEN or body.token != ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    existing_admin = await db[COL_USERS].find_one({"role": "admin"})
    if existing_admin:
        raise HTTPException(status_code=409, detail="Admin already exists")
    if await _find_user(db, str(body.email)):
        raise HTTPException(status_code=409, detail="Email already used")
    doc = {
        "first_name": "Admin",
        "last_name": "User",
        "email": str(body.email).lower(),
        "phone": "",
        "institution": "",
        "credential_note": "",
        "password_hash": _hash(body.password),
        "role": "admin",
        "status": "approved",
        "credential_file": None,
        "created_at": datetime.utcnow(),
    }
    res = await db[COL_USERS].insert_one(doc)
    return {"id": str(res.inserted_id)}