from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ClientProfile(BaseModel):
    country: Optional[str] = ""
    company_name: Optional[str] = ""
    sector: Optional[str] = ""
    original_chat_history: Optional[List[str]] = []
    chat_interest: Optional[str] = ""
    chat_summary: Optional[str] = ""
    chat_products: Optional[str] = ""
    chat_currencies: Optional[str] = ""
    chat_history: Optional[List[str]] = []


class ClientInput(BaseModel):
    sales_name              : str
    sales_email             : str
    company                 : str
    names                   : Optional[List[str | None]] = []
    emails                  : Optional[List[str | None]] = []  
    client_ids              : Optional[List[str | None]] = []
    bbg_chat_company_names  : Optional[List[str | None]] = []
    bbg_chat_sales_names    : Optional[List[str | None]] = []
    rfq_company_names       : Optional[List[str | None]] = []
    crm_company_names       : Optional[List[str | None]] = []
    is_private              : Optional[bool]             = False
    chat_sources            : Optional[List[str | None]] = []
    added_to_pipeline       : Optional[bool]             = True   
    schedule_region         : Optional[str | None]       = None


class GeneratedQuery(BaseModel):
    query: str = Field(description="Query")


class GeneratedQueries(BaseModel):
    queries: List[GeneratedQuery] = Field(description="Generated queries. Maximum 5")


class Publication(BaseModel):
    publication_id          : Optional[str] = ""
    title                   : Optional[str] = ""
    summary                 : Optional[str] = ""
    clean_content           : Optional[str] = ""
    llm_extract_topics      : Optional[str] = ""
    llm_extract_keywords    : Optional[str] = ""
    llm_extract_currencies  : Optional[str] = ""
    llm_extract_instruments : Optional[str] = ""
    author                  : Optional[str] = ""
    language                : Optional[str] = ""
    asset_class             : Optional[str] = ""
    region                  : Optional[str] = ""
    published_date          : Optional[str] = ""
    currencies              : Optional[str] = ""
    metadata                : Optional[dict] = None
    hash                    : Optional[str] = ""


class RelevanceModel(BaseModel):
    score: Literal[0, 5, 10] = Field(description="Relevance score. Choose one score amongst 0, 5, 10")
    evidences: List[str] = Field(description="Evidences")


class Passage(BaseModel):
    text: str
    rank: int
    score: Optional[float] = None


class Candidate(BaseModel):
    publication: Publication
    passages: List[Passage] = []


class ExtractedRelation(BaseModel):
    instrument: Optional[str] = ""
    underlier: Optional[str] = ""
    tenor: Optional[str] = ""
    strategy: Optional[str] = ""


class PassagePrecisionModel(BaseModel):
    score: Literal[0, 5, 10] = Field(description="Passage-level aboutness precision score. Choose one score amongst 0, 5, 10")
    relation_match: bool = Field(description="Whether the key relation (instrument on underlier) matches client intent")
    relation_confidence: float = Field(description="Confidence in the extracted relation match between 0 and 1")
    relation: ExtractedRelation = Field(description="Extracted key relation from the passage")
    evidences: List[str] = Field(description="Short snippets justifying the score")
    passage_snippet: str = Field(description="Short quote from the passage that best supports the decision")