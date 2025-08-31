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
from cores.recsys_functions import recsys_llm, recsys_rag, build_candidates_with_passages, score_passages_precision
from prompts.recsys import (
    get_queries_from_chat_interest_prompt,
    get_queries_from_chat_summary_prompt,
    get_queries_from_chat_products_prompt,
    get_queries_from_chat_currencies_prompt,
)


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
            try:
                return [_q.get("query", "") for _q in result.get("queries", []) if _q.get("query", "")]
            except Exception:
                return []

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
                                                        prompt_template=get_queries_from_chat_interest_prompt())

        call_llm_query_chat_summary_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_summary_input_template,
                                                        prompt_template=get_queries_from_chat_summary_prompt())
        
        call_llm_query_chat_products_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_products_input_template,
                                                        prompt_template=get_queries_from_chat_products_prompt())
        
        call_llm_query_chat_currencies_template = partial(call_structured_llm, 
                                                        output_schema=GeneratedQueries, 
                                                        template_type="jinja2",
                                                        prompt_inputs=chat_currencies_input_template,
                                                        prompt_template=get_queries_from_chat_currencies_prompt())
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(call_llm_query_chat_interest_template): "chat_interest",
                executor.submit(call_llm_query_chat_summary_template): "chat_summary",
                executor.submit(call_llm_query_chat_products_template): "chat_products",
                executor.submit(call_llm_query_chat_currencies_template): "chat_currencies",
            }

            results = {"chat_interest": [], "chat_summary": [], "chat_products": [], "chat_currencies": []}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    resp = future.result()
                    queries = _post_process_query_generation_result(resp)
                    results[key] = queries
                except Exception as e:
                    logger.error(f"Error occurred while processing {key}: {e}")

        queries = (
            results.get("chat_interest", [])
            + results.get("chat_summary", [])
            + results.get("chat_products", [])
            + results.get("chat_currencies", [])
        )

        # get most relevant publications during recall stage
        retrieved_docs = []
        for query in queries:
            docs = recsys_rag(
                query,
                embedding_folder_path=settings.embedding_folder_path,
                publication_file_path=settings.publication_file_path,
                top_k=20,
            )
            retrieved_docs.extend(docs)

        # Build passage-aware candidates and then reduce to publications while preserving best passages
        passage_candidates = build_candidates_with_passages(query=" ".join(queries), documents=retrieved_docs, max_passages_per_pub=3)

        # Optional: attach a simple precision signal per publication using passage precision scoring
        for cand in passage_candidates:
            precision_results = score_passages_precision(client_profile, cand)
            # select best passage by score then relation_confidence
            try:
                best = sorted(
                    precision_results,
                    key=lambda r: (int(r.get("score", 0)), float(r.get("relation_confidence", 0.0))),
                    reverse=True,
                )[0]
            except Exception:
                best = {"score": 0, "relation_confidence": 0.0}
            cand.publication.metadata = (cand.publication.metadata or {})
            cand.publication.metadata.update({
                "precision_best_passage": best,
            })

        candidates = []
        seen_hashes = set()
        for cand in passage_candidates:
            doc_hash = cand.publication.hash or cand.publication.publication_id
            if not doc_hash:
                continue
            if doc_hash in seen_hashes or doc_hash in hash_archive:
                continue
            seen_hashes.add(doc_hash)
            candidates.append(cand.publication)

        return candidates

    def _filter_and_rerank_by_precision(self,
                                        client_profile: ClientProfile,
                                        publications: List[Publication],
                                        min_score: Optional[int] = None,
                                        min_confidence: Optional[float] = None,
                                        top_n: Optional[int] = None) -> List[Publication]:

        min_score = settings.precision_min_score if min_score is None else min_score
        min_confidence = settings.precision_min_confidence if min_confidence is None else min_confidence
        top_n = settings.precision_top_k if top_n is None else top_n

        def _extract_precision_tuple(pub: Publication):
            meta = pub.metadata or {}
            best = meta.get("precision_best_passage", {}) or {}
            try:
                score = int(best.get("score", 0))
            except Exception:
                score = 0
            try:
                confidence = float(best.get("relation_confidence", 0.0))
            except Exception:
                confidence = 0.0
            relation_match = bool(best.get("relation_match", False))
            return relation_match, score, confidence

        # Filter by thresholds
        filtered = []
        for pub in publications:
            relation_match, score, confidence = _extract_precision_tuple(pub)
            if relation_match and score >= min_score and confidence >= min_confidence:
                filtered.append((pub, score, confidence))

        # Sort by precision score then confidence desc
        filtered.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Trim to top_n
        top_filtered = [p for p, _, _ in (filtered[:top_n] if top_n else filtered)]

        return top_filtered

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

        # Stage 1: Recall
        candidates = self._generate_candidates(client_profile=client_profile,
                                               recommendation_date=recommendation_date)

        # Stage 2: Precision filter & rerank (aboutness + relation confidence)
        precision_candidates = self._filter_and_rerank_by_precision(client_profile=client_profile,
                                                                    publications=candidates)

        # Stage 3: Final LLM relevance scoring on precision-filtered set
        recommendations = recsys_llm(client_profile=client_profile,
                                     candidates=precision_candidates)

        return recommendations


    def generate_reports(self, 
                         client: ClientInput, 
                         recommendations: list,
                         recommendation_date: Optional[str] = None):
        pass
    
    def send_email(self, client: ClientInput, recommendations):
        pass