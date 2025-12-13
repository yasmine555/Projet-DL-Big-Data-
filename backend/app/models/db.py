import os
from typing import Any, Dict
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "alz_rag")

_client = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client

async def get_db() -> AsyncIOMotorDatabase:
    return get_client()[MONGO_DB]

# Collection names
COL_USERS = "users"
COL_PATIENTS = "patients"
COL_RESULTS = "results"
COL_CONVERSATIONS = "conversations"
COL_MRI_SCANS = "mri_scans"

class MongoDoc(BaseModel):
    id: str | None = None
    def mongo(self) -> Dict[str, Any]:
        d = self.dict(exclude_none=True)
        if "id" in d:
            d["_id"] = d.pop("id")
        return d

# --- Convenience helpers for patients collection ---
async def patients_find(filter: Dict[str, Any]) -> list:
    db = await get_db()
    cursor = db[COL_PATIENTS].find(filter).sort("created_at", -1)
    return [doc async for doc in cursor]

async def patients_find_one(filter: Dict[str, Any]) -> Dict[str, Any] | None:
    db = await get_db()
    return await db[COL_PATIENTS].find_one(filter)

async def patients_insert_one(doc: Dict[str, Any]):
    db = await get_db()
    # add created_at if missing
    doc.setdefault("created_at", None)
    return await db[COL_PATIENTS].insert_one(doc)

async def patients_update_one(filter: Dict[str, Any], update: Dict[str, Any]) -> bool:
    db = await get_db()
    res = await db[COL_PATIENTS].update_one(filter, update)
    return res.matched_count > 0

async def patients_delete_one(filter: Dict[str, Any]):
    db = await get_db()
    return await db[COL_PATIENTS].delete_one(filter)

def to_object_id(id_str: str) -> ObjectId:
    return ObjectId(id_str)
