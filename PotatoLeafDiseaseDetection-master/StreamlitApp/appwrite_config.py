# appwrite_config.py
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.account import Account

# Initialize Appwrite client
client = Client()

client.set_endpoint("https://fra.cloud.appwrite.io/v1")  # ✅ Use your correct endpoint
client.set_project("688a1b610038ca502d2f")               # ✅ Use your correct project ID



# Setup services
account = Account(client)
database = Databases(client)

# Export for other files to use
__all__ = ["client", "account", "database"]
