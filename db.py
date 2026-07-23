from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "form2database")

client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
db = client[DB_NAME]
documents_collection = db["documents"]
