import copy
import pandas as pd
from loguru import logger
from typing import Optional
from functools import partial
from collections import defaultdict
from custom_types import ClientInput
from concurrent.futures import as_completed, ThreadPoolExecutor

from settings import settings

from custom_types import *
from cores.llm_functions import call_structured_llm
from cores.chat_functions import chat_feature_extraction
from cores.recsys_functions import recsys_llm, recsys_rag


class Market360Recsys:
    def __init__(self, recommendation_date):
        self.recommendation_date = recommendation_date

    @staticmethod
    def _prepare_bbg_chat_data():
        return pd.read_csv(settings.bbg_chat_file_path)

    def _generate_candidates(self, client_profile: ClientProfile, recommendation_date: int) -> List[Publication]:
        recommendation_date = recommendation_date or self.recommendation_date

        hash_archive = set()

        def _post_process_query_generation_result(result, **kwargs):
            return [_queries["query"] for _queries in result["queries"]]

        default_input_parameters = {"chat_interest"      : None,
                                    "chat_summary"       : None,
                                    "chat_products"      : None,
                                    "chat_currencies"    : None,
                                    "number_of_question" : 2}
        
        chat_interest_input_template   = copy.deepcopy(default_input_parameters)
        chat_summary_input_template    = copy.deepcopy(default_input_parameters)
        chat_products_input_template   = copy.deepcopy(default_input_parameters)
        chat_currencies_input_template = copy.deepcopy(default_input_parameters)
        
        chat_interest_input_template["chat_interest"]     = client_profile.chat_interest
        chat_summary_input_template["chat_summary"]       = client_profile.chat_summary
        chat_products_input_template["chat_products"]     = client_profile.chat_products
        chat_currencies_input_template["chat_currencies"] = client_profile.chat_currencies
        
        call_llm_query_chat_interest_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_interest_input_template,
                                                        post_process_func=_post_process_query_generation_result)

        call_llm_query_chat_summary_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_summary_input_template,
                                                        post_process_func=_post_process_query_generation_result)
        
        call_llm_query_chat_products_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_products_input_template,
                                                        post_process_func=_post_process_query_generation_result)
        
        call_llm_query_chat_currencies_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_currencies_input_template,
                                                        post_process_func=_post_process_query_generation_result)
        
        with ThreadPoolExecutor(max_workers=settings.max_worker) as executor:
            futures = {executor.submit(call_llm_query_chat_interest_template),
                       executor.submit(call_llm_query_chat_summary_template),
                       executor.submit(call_llm_query_chat_products_template),
                       executor.submit(call_llm_query_chat_currencies_template)}
            
            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Error occurred while processing {task_name}: {e}")
                else:
                    results[task_name] = result

        queries_chat_interest = results.get("chat_interest", "")
        queries_chat_summary = results.get("chat_summary", "")
        queries_chat_products = results.get("chat_products", "")
        queries_chat_currencies = results.get("chat_currencies", "")

        queries = queries_chat_interest + queries_chat_summary + queries_chat_products + queries_chat_currencies

        # get most relevant publications during recall stage
        retrieved_docs = []

        for query in queries:
            retrieved_docs.append(recsys_rag(query, 
                                             embedding_folder_path=settings.embedding_folder_path,
                                             publication_file_path=settings.publication_file_path,
                                             recommendation_date=recommendation_date))

        candidates = []
        unique_candidates = []
        for doc in retrieved_docs:
            if doc.metadata["hash"] not in unique_candidates and doc.metadata['hash'] not in hash_archive:
                candidates.append(Publication(**doc.metadata))

        return candidates

    def recommend(self, 
                  client: ClientInput,
                  recommendation_date : Optional[int] = settings.bbg_chat_coverage_day_range,
                  bbg_chat_coverage_day_range : Optional[int] = settings.bbg_chat_coverage_day_range):
        
        recommendation_date = recommendation_date or self.recommendation_date

        raw_bbg_chat_df = self._prepare_bbg_chat_data()

        client_profile = defaultdict()
        client_profile["country"]      = ""
        client_profile["company_name"] = client.company
        client_profile["sector"]       = ""

        client_profile = chat_feature_extraction(client_profile=client_profile,
                                                 chat_history_df=raw_bbg_chat_df)

        client_profile = ClientProfile(**client_profile)

        candidates = self._generate_candidates(client_profile=client_profile,
                                              recommendation_date=recommendation_date)
        
        recommendations = recsys_llm(client_profile = client_profile,
                                     candidates = candidates)

        return recommendations


    def generate_reports(self, 
                         client: ClientInput, 
                         recommendations: list,
                         recommendation_date: Optional[str] = None):
        pass
    
    def send_email(self, client: ClientInput, recommendations):
        pass