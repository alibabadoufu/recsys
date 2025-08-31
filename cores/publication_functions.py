import pandas as pd
from functools import partial

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from settings import settings
from prompts.publication import (generate_keywords_prompt,
                                 generate_currencies_prompt, 
                                 generate_topics_prompt)
from cores.llm_functions import call_llm


def publication_feature_extraction(pub_df: pd.DataFrame):
    
    llm_extract_topics_partial      = partial(call_llm, prompt_template=generate_topics_prompt())
    llm_extract_keywords_partial    = partial(call_llm, prompt_template=generate_keywords_prompt())
    llm_extract_currencies_partial  = partial(call_llm, prompt_template=generate_currencies_prompt())

    pub_df.loc[:, "llm_extract_topics"]     = pub_df.loc[:, "clean_content"].apply(llm_extract_topics_partial)
    pub_df.loc[:, "llm_extract_keywords"]   = pub_df.loc[:, "clean_content"].apply(llm_extract_keywords_partial)
    pub_df.loc[:, "llm_extract_currencies"] = pub_df.loc[:, "clean_content"].apply(llm_extract_currencies_partial)

    return pub_df


def publication_preprocessing(pub_df: pd.DataFrame):
    string_template = ("{page_content}\n\n"
                       "Publication title: {title}\n\n"
                       "Publication summary: {summary}\n\n"
                       "Publication language: {language}\n\n"
                       "Publication asset class: {asset_class}\n\n"
                       "Publication keywords: {llm_extract_keywords}\n\n"
                       "Publication currencies: {llm_extract_currencies}\n\n"
                       "Publication topics: {llm_extract_topics}\n\n")
    
    publications = pub_df.apply(lambda row: Document(page_content=row["clean_content"],
                                                     metadata=row),
                                            axis=1).tolist()
    

    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.publication_chunk_size, 
                                                   chunk_overlap=settings.publication_chunk_overlap,
                                                   separators=["\n\n", "\n", " ", ""])
    
    text_chunks = text_splitter.split_documents(publications)

    for text_chunk in text_chunks:
        text_chunk.page_content = string_template.format(page_content=text_chunk.page_content,
                                                         title=text_chunk.metadata["title"],
                                                         summary=text_chunk.metadata["summary"],
                                                         language=text_chunk.metadata["language"],
                                                         asset_class=text_chunk.metadata["asset_class"],
                                                         llm_extract_keywords=text_chunk.metadata["llm_extract_keywords"],
                                                         llm_extract_currencies=text_chunk.metadata["llm_extract_currencies"],
                                                         llm_extract_topics=text_chunk.metadata["llm_extract_topics"])

    return text_chunks