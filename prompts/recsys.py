def get_currency_relevance_prompt() -> str:
    return """
    You are assessing ABOUTNESS for currency relevance.

    Task: Given a client's extracted chat interests and a candidate publication's metadata, decide if the publication is about the currencies the client cares about, not just a passing mention.

    Rules:
    - Consider currencies and currency pairs found in the candidate's fields: title, summary, Publication currencies, Publication topics, Publication keywords, and the main body.
    - Penalize background/incidental mentions (e.g., currency appears only in summaries/footnotes without support in body).
    - Reward repeated, section-wide mentions and appearances in the title/heading.
    - Be strict: if the candidate is mainly about a different currency context, return score 0.

    Input:
    - Client chat currencies: {{ client_chat_currencies }}
    - Client chat interest summary: {{ client_chat_interest }}
    - Candidate title: {{ candidate_title }}
    - Candidate summary: {{ candidate_summary }}
    - Candidate currencies: {{ candidate_currencies }}
    - Candidate topics: {{ candidate_topics }}
    - Candidate keywords: {{ candidate_keywords }}

    Output: Return a JSON object with fields "score" and "evidences" that conforms to the provided schema. Choose score strictly from {0, 5, 10}.
    - score: 10 = primarily about client's currencies; 5 = partially about; 0 = incidental.
    - evidences: short bullet snippets that justify your score (max 3).
    """


def get_chat_topic_relevance_prompt() -> str:
    return """
    You are assessing ABOUTNESS for topic relevance.

    Task: Given client chat topics/summary and a candidate publication, decide if the publication’s main topic aligns with the client's topics, not just keyword overlap.

    Rules:
    - Focus on the main subject of the publication (title and repeated body themes).
    - Down-weight generic market wrap sections if there is no supporting body content.
    - Be strict when client mentions a specific instrument or subtopic (e.g., options on EURUSD vs general EUR macro note).

    Input:
    - Client chat summary: {{ client_chat_summary }}
    - Client chat interest: {{ client_chat_interest }}
    - Candidate title: {{ candidate_title }}
    - Candidate summary: {{ candidate_summary }}
    - Candidate topics: {{ candidate_topics }}
    - Candidate keywords: {{ candidate_keywords }}

    Output: Return a JSON object with fields "score" and "evidences" that conforms to the provided schema. Choose score strictly from {0, 5, 10}.
    - score: 10 = clearly about the client’s topics; 5 = partially; 0 = off-topic/incidental.
    - evidences: short bullet snippets (max 3).
    """


def get_chat_product_relevance_prompt() -> str:
    return """
    You are assessing ABOUTNESS for product/instrument relevance.

    Task: Determine whether the candidate publication is mainly about the financial products/instruments reflected in the client's chat (e.g., options, swaps, convertibles), rather than just mentioning them.

    Rules:
    - Prefer precise matches of instrument type and underlier.
    - If the client interest indicates "options" and the candidate is about "convertible bonds", return 0 even if currencies overlap.
    - Treat clear mentions in title/section headings and repeated discussion in body as strong signals.

    Input:
    - Client chat products: {{ client_chat_products }}
    - Client chat interest: {{ client_chat_interest }}
    - Candidate title: {{ candidate_title }}
    - Candidate summary: {{ candidate_summary }}
    - Candidate instruments: {{ candidate_instruments }}
    - Candidate topics: {{ candidate_topics }}
    - Candidate keywords: {{ candidate_keywords }}

    Output: Return a JSON object with fields "score" and "evidences" that conforms to the provided schema. Choose score strictly from {0, 5, 10}.
    - score: 10 = primarily about the instruments client cares about; 5 = partially; 0 = incidental.
    - evidences: short bullet snippets (max 3).
    """


def get_queries_from_chat_interest_prompt() -> str:
    return """
    You are a query generator that creates search queries for retrieving research passages.

    Goal: Generate highly precise queries that reflect the client's interests. Avoid broad queries that cause incidental matches.

    Inputs:
    - Client chat interest: {{ chat_interest }}
    - Client chat products: {{ chat_products }}
    - Client chat currencies: {{ chat_currencies }}
    - Number of queries: {{ number_of_question }}

    Guidance:
    - Include instrument type when present (e.g., options, swaps, CDS, convertibles).
    - Include specific underliers (e.g., EURUSD, USDJPY) when present.
    - Prefer concise phrases suitable for retrieval; avoid stopwords.
    - Provide diverse phrasings to cover common synonyms.

    Output: JSON with field "queries" as a list of objects with a single key "query". Max {{ number_of_question }} queries.
    """


def get_queries_from_chat_summary_prompt() -> str:
    return """
    Generate retrieval queries from the client's chat summary.

    Inputs:
    - Client chat summary: {{ chat_summary }}
    - Number of queries: {{ number_of_question }}

    Rules:
    - Focus on the most specific actionable topics.
    - Prefer including instrument types and tickers/underliers when present.
    - Avoid generic market wrap terms.

    Output: JSON with field "queries" (list of {"query": <text>}). Max {{ number_of_question }} queries.
    """


def get_queries_from_chat_products_prompt() -> str:
    return """
    Generate retrieval queries emphasizing product/instrument specificity.

    Inputs:
    - Client chat products: {{ chat_products }}
    - Client chat currencies: {{ chat_currencies }}
    - Number of queries: {{ number_of_question }}

    Rules:
    - Include instrument types (e.g., options, swaps, CDS, convertibles) and underliers.
    - Use common strategy terms when present (e.g., risk reversal, strangle).

    Output: JSON with field "queries" (list of {"query": <text>}). Max {{ number_of_question }} queries.
    """


def get_queries_from_chat_currencies_prompt() -> str:
    return """
    Generate retrieval queries centered on the client's currencies.

    Inputs:
    - Client chat currencies: {{ chat_currencies }}
    - Number of queries: {{ number_of_question }}

    Rules:
    - If currency pairs are present (e.g., EUR/USD), include them exactly and with no whitespace variants (EURUSD, EUR-USD) across different queries.
    - If single currencies are present, do not expand to pairs unless clearly implied by client context.

    Output: JSON with field "queries" (list of {"query": <text>}). Max {{ number_of_question }} queries.
    """


def get_passage_precision_prompt() -> str:
    return """
    You act as a passage-level ABOUTNESS precision scorer and relation extractor.

    Goal: Determine if the passage is primarily about the client's intent, focusing on instrument-underlier relations. Extract the key relation and score precision.

    Inputs:
    - Client interest: {{ client_chat_interest }}
    - Client products: {{ client_chat_products }}
    - Client currencies: {{ client_chat_currencies }}
    - Passage text: {{ passage_text }}

    Instructions:
    - Identify instrument (e.g., options, swaps, CDS, convertibles), underlier (e.g., EURUSD), optional tenor (e.g., 1M), and strategy (e.g., risk reversal).
    - The passage should be considered precise only if the instrument and underlier are clearly connected in the passage (not separate unrelated mentions).
    - Score strictly in {0,5,10}. 10 = the passage’s main topic strongly matches client intent; 5 = partial; 0 = incidental.

    Output: Return JSON that conforms to the provided schema with fields: score, relation_match, relation_confidence (0-1), relation {instrument, underlier, tenor, strategy}, evidences (<=3), passage_snippet.
    """
