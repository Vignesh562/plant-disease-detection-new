# appwrite_config.py
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.account import Account

# Initialize Appwrite client
client = Client()

client.set_endpoint("https://fra.cloud.appwrite.io/v1")  # Do not change
client.set_project("688a1b610038ca502d2f")  # Replace with your real project ID

# Setup services
account = Account(client)

# To create a user
account.create(email="email@example.com", password="securepassword")

# To login
account.create_email_session(email="email@example.com", password="securepassword")

#Database 
databases = Databases(client)

databases.create_document(
    database_id="your-db-id",
    collection_id="predictions",
    document_id="unique()",
    data={
        "user_email": email,
        "image_url": image_url,
        "predicted_disease": disease,
        "description": disease_desc,
        "timestamp": datetime.now().isoformat()
    }
)


# Export for other files to use
__all__ = ["client", "account", "database"]
