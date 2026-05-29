from app.config import settings

print("Database:", settings.database_url)
print("Model:", settings.nebius_model)
print("Dataset:", settings.dataset_name)
print("API key exists:", bool(settings.nebius_api_key))