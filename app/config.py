from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    nebius_api_key: str

    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nebius_model: str = "meta-llama/Llama-3.3-70B-Instruct-fast"
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()