import os
import json
import numpy as np
import pandas as pd
from typing import List
from loguru import logger
from datetime import datetime

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS  

from prompts.recsys import *
from settings import settings
from utils.utils import replace_empty_string
from cores.llm_functions import call_structured_llm
from custom_types import ClientProfile, Publication, RelevanceModel
from cores.publication_functions import publication_feature_extraction, publication_preprocessing


def recsys_llm(client_profile: ClientProfile, candidates: List[Publication]):

    output_dicts = []

    for candidate in candidates:
        output_dict = {
            "title": candidate.title
        }

        def _post_process_func(result, **kwargs):
            return {"score": 0, "evidences": []} if ("score" not in result and "evidences" not in result) or result == " " else result

        def _call_structured_llm_wrapper(**kwargs):
            return call_structured_llm(**kwargs)

        def _sequential_structured_llm_calls():
            prompts = [
                {
                    "prompt_template": get_currency_relevance_prompt(),
                    "output_schema": RelevanceModel,
                    "template_type": "jinja2",
                    "name": "Currency Relevance",
                    "prompt_inputs": output_dict
                },
                {
                    "prompt_template": get_chat_topic_relevance_prompt(),
                    "output_schema": RelevanceModel,
                    "template_type": "jinja2",
                    "name": "Chat Topic Relevance",
                    "prompt_inputs": output_dict
                },
                {
                    "prompt_template": get_chat_product_relevance_prompt(),
                    "output_schema": RelevanceModel,
                    "template_type": "jinja2",
                    "name": "Chat Product Relevance",
                    "prompt_inputs": output_dict
                }
            ]

            results = []

            for prompt in prompts:
                try:
                    result = _call_structured_llm_wrapper(**prompt, post_process_func=_post_process_func)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error calling LLM for {prompt['name']}: {e}")

            currency_result, chat_topic_result, chat_product_result = results

            return currency_result, chat_topic_result, chat_product_result
        
        currency_result, chat_topic_result, chat_product_result = _sequential_structured_llm_calls()

        output_dict["currency relevance"] = currency_result
        output_dict["chat topic relevance"] = chat_topic_result
        output_dict["chat product relevance"] = chat_product_result

        try:
            currency_score = int(currency_result["score"])
        except:
            currency_score = 0
        
        try:
            chat_topic_score = int(chat_topic_result["score"])
        except:
            chat_topic_score = 0
        
        try:
            chat_product_score = int(chat_product_result["score"])
        except:
            chat_product_score = 0
            
        output_dict["weighted average score"] = round(np.average(a=[currency_score, chat_topic_score, chat_product_score], weights=[1, 1, 1]))
        output_dict["evidences"] = currency_result["evidences"] + chat_topic_result["evidences"] + chat_product_result["evidences"]

        output_dicts.append(output_dict)

    try:
        output_df = pd.DataFrame(output_dicts)
        output_df["currency relevance"] = output_df["currency relevance"].apply(replace_empty_string)
        output_df["chat topic relevance"] = output_df["chat topic relevance"].apply(replace_empty_string)
        output_df["chat product relevance"] = output_df["chat product relevance"].apply(replace_empty_string)
        output_df.to_csv(settings.output_file_path.format(date=datetime.now().strftime("%Y-%m-%d")), index=False)
        logger.success(f"Output dataframe created and saved to {settings.output_file_path.format(date=datetime.now().strftime("%Y-%m-%d"))}")
    except Exception as e:
        logger.error(f"Error creating output dataframe: {e}")

    return output_df



def recsys_rag(query : str,
               embedding_folder_path : str,
               publication_file_path : str):
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_embeddings_endpoint,
        azure_deployment=settings.azure_openai_embeddings_deployment_name,
        openai_api_version=settings.azure_openai_embeddings_api_version,
        openai_api_key=settings.azure_openai_embeddings_api_key,
    )

    if os.path.exists(embedding_folder_path):
        faiss_retriever = FAISS.load_local(embedding_folder_path, embeddings=embeddings, allow_dangerous_deserialization=True).as_retriever()
    else:
        doc_df = pd.read_json(publication_file_path)
        doc_df = publication_feature_extraction(doc_df)
        documents = publication_preprocessing(doc_df)
        faiss_retriever = FAISS.from_documents(documents, embeddings=embeddings)

    relevant_documents = faiss_retriever.invoke(query)

    return relevant_documents
