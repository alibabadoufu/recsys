import os
import json
import numpy as np
import pandas as pd
from typing import List
from loguru import logger
from datetime import datetime

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS  

from prompts.recsys import (
    get_currency_relevance_prompt,
    get_chat_topic_relevance_prompt,
    get_chat_product_relevance_prompt,
    get_passage_precision_prompt,
)
from settings import settings
from utils.utils import replace_empty_string
from cores.llm_functions import call_structured_llm
from custom_types import ClientProfile, Publication, RelevanceModel, Candidate, Passage, PassagePrecisionModel
from cores.publication_functions import publication_feature_extraction, publication_preprocessing


def recsys_llm(client_profile: ClientProfile, candidates: List[Publication]):

    output_dicts = []

    for candidate in candidates:
        output_dict = {
            "title": candidate.title
        }

        def _ensure_result_shape(result_dict):
            try:
                score = result_dict.get("score", 0)
                evidences = result_dict.get("evidences", [])
            except Exception:
                score = 0
                evidences = []
            return {"score": score, "evidences": evidences}

        def _call_structured_llm_wrapper(**kwargs):
            return call_structured_llm(**kwargs)

        def _sequential_structured_llm_calls():
            prompt_inputs_common = {
                "client_chat_currencies": client_profile.chat_currencies,
                "client_chat_interest": client_profile.chat_interest,
                "client_chat_summary": client_profile.chat_summary,
                "client_chat_products": client_profile.chat_products,
                "candidate_title": candidate.title or "",
                "candidate_summary": candidate.summary or "",
                "candidate_currencies": candidate.llm_extract_currencies or "",
                "candidate_topics": candidate.llm_extract_topics or "",
                "candidate_keywords": candidate.llm_extract_keywords or "",
                "candidate_instruments": candidate.llm_extract_instruments or "",
            }

            prompts = [
                {
                    "prompt_template": get_currency_relevance_prompt(),
                    "output_schema": RelevanceModel,
                    "template_type": "jinja2",
                    "name": "Currency Relevance",
                    "prompt_inputs": prompt_inputs_common,
                },
                {
                    "prompt_template": get_chat_topic_relevance_prompt(),
                    "output_schema": RelevanceModel,
                    "template_type": "jinja2",
                    "name": "Chat Topic Relevance",
                    "prompt_inputs": prompt_inputs_common,
                },
                {
                    "prompt_template": get_chat_product_relevance_prompt(),
                    "output_schema": RelevanceModel,
                    "template_type": "jinja2",
                    "name": "Chat Product Relevance",
                    "prompt_inputs": prompt_inputs_common,
                },
            ]

            results = []

            for prompt in prompts:
                try:
                    result = _call_structured_llm_wrapper(**prompt)
                    results.append(_ensure_result_shape(result))
                except Exception as e:
                    logger.error(f"Error calling LLM for {prompt['name']}: {e}")
                    results.append({"score": 0, "evidences": []})

            currency_result, chat_topic_result, chat_product_result = results

            return currency_result, chat_topic_result, chat_product_result
        
        currency_result, chat_topic_result, chat_product_result = _sequential_structured_llm_calls()

        output_dict["currency relevance"] = currency_result
        output_dict["chat topic relevance"] = chat_topic_result
        output_dict["chat product relevance"] = chat_product_result

        try:
            currency_score = int(currency_result.get("score", 0))
        except Exception:
            currency_score = 0
        try:
            chat_topic_score = int(chat_topic_result.get("score", 0))
        except Exception:
            chat_topic_score = 0
        try:
            chat_product_score = int(chat_product_result.get("score", 0))
        except Exception:
            chat_product_score = 0
            
        output_dict["weighted average score"] = round(np.average(a=[currency_score, chat_topic_score, chat_product_score], weights=[1, 1, 1]))
        output_dict["evidences"] = (
            (currency_result.get("evidences", []) or [])
            + (chat_topic_result.get("evidences", []) or [])
            + (chat_product_result.get("evidences", []) or [])
        )

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


def build_candidates_with_passages(
    query: str,
    documents: List,
    max_passages_per_pub: int = 3,
) -> List[Candidate]:
    grouped: dict[str, List] = {}
    for idx, doc in enumerate(documents):
        pub_hash = doc.metadata.get("hash") or doc.metadata.get("publication_id") or f"nohash-{idx}"
        grouped.setdefault(pub_hash, []).append((idx, doc))

    candidates: List[Candidate] = []
    for _, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x[0])[:max_passages_per_pub]
        any_doc = items_sorted[0][1]
        publication = Publication(**any_doc.metadata)
        passages = [Passage(text=d.page_content, rank=i, score=None) for i, (_, d) in enumerate(items_sorted, start=1)]
        candidates.append(Candidate(publication=publication, passages=passages))

    return candidates


def score_passages_precision(
    client_profile: ClientProfile,
    candidate: Candidate,
    llm_caller=call_structured_llm,
) -> List[dict]:
    results: List[dict] = []
    prompt_template = get_passage_precision_prompt()
    for p in candidate.passages:
        prompt_inputs = {
            "client_chat_interest": client_profile.chat_interest,
            "client_chat_products": client_profile.chat_products,
            "client_chat_currencies": client_profile.chat_currencies,
            "passage_text": p.text,
        }
        try:
            result = llm_caller(
                prompt_template=prompt_template,
                prompt_inputs=prompt_inputs,
                template_type="jinja2",
                output_schema=PassagePrecisionModel,
            )
        except Exception as e:
            logger.error(f"Passage precision scoring failed: {e}")
            result = {
                "score": 0,
                "relation_match": False,
                "relation_confidence": 0.0,
                "relation": {"instrument": "", "underlier": "", "tenor": "", "strategy": ""},
                "evidences": [],
                "passage_snippet": "",
            }
        results.append(result)
    return results



def recsys_rag(
    query: str,
    embedding_folder_path: str,
    publication_file_path: str,
    top_k: int = 10,
):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_embeddings_endpoint,
        azure_deployment=settings.azure_openai_embeddings_deployment_name,
        openai_api_version=settings.azure_openai_embeddings_api_version,
        openai_api_key=settings.azure_openai_embeddings_api_key,
    )

    if os.path.exists(embedding_folder_path):
        vectorstore = FAISS.load_local(
            embedding_folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        doc_df = pd.read_json(publication_file_path)
        doc_df = publication_feature_extraction(doc_df)
        documents = publication_preprocessing(doc_df)
        vectorstore = FAISS.from_documents(documents, embeddings=embeddings)
        os.makedirs(embedding_folder_path, exist_ok=True)
        vectorstore.save_local(embedding_folder_path)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    relevant_documents = retriever.invoke(query)
    return relevant_documents
