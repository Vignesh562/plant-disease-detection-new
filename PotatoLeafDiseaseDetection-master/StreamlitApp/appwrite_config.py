# appwrite_config.py
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.account import Account

# Initialize Appwrite client
client = Client()

client.set_endpoint("https://cloud.appwrite.io/v1")  # Do not change
client.set_project("your_actual_project_id")  # Replace with your real project ID
client.set_key("your_api_key")  # Secure key for server access (will guide on creating)

# Setup services
account = Account(client)
database = Databases(client)

# Export for other files to use
__all__ = ["client", "account", "database"]
