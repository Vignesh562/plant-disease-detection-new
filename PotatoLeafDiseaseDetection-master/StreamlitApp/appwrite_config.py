from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.databases import Databases

# ✅ Hardcoded values – REPLACE these with your actual project info
APPWRITE_ENDPOINT = "https://fra.cloud.appwrite.io/v1"
APPWRITE_PROJECT = "688a1b610038ca502d2f"
APPWRITE_DATABASE_ID = "688a1e470000b53815e8"      # Replace this
APPWRITE_COLLECTION_ID = "688a1ed30009b55657c9"  # Replace this

# Initialize Appwrite client
client = Client()
client.set_endpoint("https://fra.cloud.appwrite.io/v1")
client.set_project("688a1b610038ca502d2f")

# Initialize services
account = Account(client)
database = Databases(client)

# Export everything needed
__all__ = ["client", "account", "database", "APPWRITE_DATABASE_ID", "APPWRITE_COLLECTION_ID"]
