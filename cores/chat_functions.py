import pandas as pd
from loguru import logger
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from prompts.chat import *
from settings import settings
from cores.llm_functions import call_llm


def chat_feature_extraction(client_profile: dict, chat_history_df: pd.DataFrame):

    chat_history = [chat for chat in chat_history_df["chat"]]

    call_llm_client_template = partial(call_llm, prompt_inputs={"chat_history": "\n\n".join(chat_history)}, template_type="jinja2")

    prompt_templates = [
        ("chat_summary", generate_chat_summary_prompt()),
        ("chat_interest", generate_chat_interest_prompt()),
        ("chat_products", generate_chat_products_prompt()),
        ("chat_currencies", generate_chat_currencies_prompt()),
    ]

    def call_llm_client(prompt_template):
        return call_llm_client_template(prompt_template=prompt_template)
    

    with ThreadPoolExecutor(max_workers=settings.max_worker) as executor:
        futures = {executor.submit(call_llm_client, prompt_template): column_name for column_name, prompt_template in prompt_templates}

        for future in futures:
            column_name = futures[future]
            try:
                result = future.result()
                client_profile[column_name] = result
            except Exception as e:
                logger.error(f"Error calling LLM client for {column_name}: {e}")
            client_profile[column_name] = future.result()

    # Initialize Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_embeddings_endpoint,
        azure_deployment=settings.azure_openai_embeddings_deployment_name,
        openai_api_version=settings.azure_openai_embeddings_api_version,
        openai_api_key=settings.azure_openai_embeddings_api_key,
    )
    chat_history_documents = [Document(page_content=chat_item.get("msg", ""), metadata=chat_item) for chat_item in chat_history]

    vectorstore = FAISS.from_documents(chat_history_documents, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "fetch_k": 200,
            "score_threshold": 0.5,
        }
    )

    query = " ".join([client_profile["chat_interest"], client_profile["chat_products"], client_profile["chat_currencies"]])
    relevant_chat_documents = retriever.invoke(query)
    relevant_chats = [doc.metadata['chat'] for doc in relevant_chat_documents]

    client_profile["chat_history"] = relevant_chats
    client_profile["original_chat_history"] = chat_history[["name", "company_name", "msg", "type"]].to_dict("records")

    return client_profile
    