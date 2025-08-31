from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    mode: str = "DEV"

    max_worker: int = 10

    clients_file_path   : str = "clients.yaml"
    bbg_chat_file_path  : str = "data/bbg_chat.csv"

    bbg_chat_coverage_day_range : int = 30
    rfq_coverage_day_range      : int = 30
    crm_coverage_day_range      : int = 30

    embedding_model             : str = "text-embedding-3-small"
    embedding_instruction       : str = "You are a helpful assistant."
    embedding_query_instruction : str = "You are a helpful assistant."
    user_id                     : str = "market360"

    # Azure OpenAI Configuration
    azure_openai_endpoint           : str = ""
    azure_openai_api_key            : str = ""
    azure_openai_deployment_name    : str = "gpt-4.1-mini"
    azure_openai_api_version        : str = "2024-12-01-preview"
    azure_openai_model_name         : str = "gpt-4.1-mini"

    # Azure OpenAI Embeddings Configuration
    azure_openai_embeddings_endpoint        : str = ""
    azure_openai_embeddings_api_key         : str = ""
    azure_openai_embeddings_api_version     : str = "2024-12-01-preview"
    azure_openai_embeddings_deployment_name : str = "text-embedding-3-small"
    azure_openai_embeddings_model_name      : str = "text-embedding-3-small"


    # Embeddings
    embedding_folder_path : str = "data/embeddings"
    publication_file_path : str = "data/publications.json"

    # Publication chunk
    publication_chunk_size      : int = 1000
    publication_chunk_overlap   : int = 100

    # Recsys
    output_file_path : str = "results/recsys_output_{date}.csv"

    # Precision filtering
    precision_min_score: int = 5
    precision_min_confidence: float = 0.5
    precision_top_k: int = 30

settings = Settings()
