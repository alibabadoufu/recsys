def generate_chat_summary_prompt() -> str:
    return """
    {
        "task": "Provide the main topic or topics in the chat_history.",
        "steps": [
            "step_1: Identify and list the primary topic or category or provide a short description of the main subject matter of the text.",
            "step_2: If there are subtopics or secondary themes mentioned in the text, list them as well. If the text discusses multiple topics, provide a list of these topics and describe their relevance.",
            "step_3: Consider the context and tone of the text to determine the most appropriate topics. Take into account keywords, phrases, or specific terms that relate to the topics.",
            "step_4: If any notable entities (people, places, brands, products, etc.) are mentioned in the text that play a role in the topics, mention them and their associations.",
            "step_5: If the text suggests any actions, decisions, or recommendations related to the identified topics, provide a brief summary of these insights."
        ],
        "clarity": "Ensure that your labeling is clear, concise, and reflects the most significant topics or categories found in the text.",
        "chat_history": "{{ chat_history }}",
        "output_format": "A comma-separated non-repeated list of the main topics and subtopics in the chat history. Do not include any other text."
    }
    """


def generate_chat_interest_prompt() -> str:
    """
    {
        "task": "Identify at most 10 market events or news topics that users will be interested in related to financial markets, economies, or economic policies based on the chat history provided.",
        "context_analysis": {
            "currencies": "Identify currencies, currency pairs, and exchange rates mentioned (e.g., USD/SGD).",
            "geographic_regions": "Identify geographic regions, countries, or cities referenced (e.g., US, Singapore).",
            "economic_indicators": "Identify economic indicators, trends, or events discussed (e.g., market outlook, inflation, GDP growth).",
            "financial_instruments": "Identify financial instruments, asset classes, or investment products mentioned (e.g., stocks, bonds, ETFs).",
            "policy_topics": "Identify policy-related topics, such as central bank decisions, regulatory changes, or government initiatives.",
            "user_interests": "Analyze the conversation context, entities mentioned, and topics discussed to infer the user's interest in market events or news titles. Assume the user is interested in learning more about the topics they have discussed or asked about in the chat history."
        },
        "chat_history": "{{ chat_history }}",
        "example_input": [
            "Client: Hey, what's the current exchange rate for USD/SGD?",
            "Sales: As of today, the exchange rate is 1 USD = 1.35 SGD.",
            "Client: That's interesting. What's the outlook for the US market this year?",
            "Sales: Many analysts expect a moderate growth in the US market, driven by...",
            "Client: I see. What about Singapore's monetary policy? Any changes expected?"
        ],
        "example_output": "US Market Outlook, Singapore Currency Policy, USD/SGD Exchange Rate",
        "max_output_topics": 10,
        "output_instruction": "A comma-separated non-repeated list of market events or news topics in the chat_history. Do not include any other text."
    }
    """


def generate_chat_products_prompt() -> str:
    return """
    {
        "task": "Label the main financial products in the chat history.",
        "clarity": "Ensure that your labeling is clear and concise.",
        "chat_history": "{{ chat_history }}",
        "example_input": [
            "Client: I'm thinking of investing in ETFs. Any recommendations?",
            "Sales: ETFs are a great choice. You might consider Vanguard's VOO or the SPDR S&P 500 ETF.",
            "Client: What about bonds? Are they a good option right now?",
            "Sales: Bonds can provide stability. Treasury bonds are always a safe bet.",
            "Client: I've also heard about mutual funds. What do you think?",
            "Sales: Mutual funds can diversify your portfolio. Fidelity's Contrafund is a popular choice."
        ],
        "example_output": "ETFs, Vanguard's VOO, SPDR S&P 500 ETF, Bonds, Treasury bonds, Mutual funds, Fidelity's Contrafund",
        "max_output_products": 10,
        "output_format": "A comma-separated non-repeated list of financial products mentioned in the chats. Do not include any other text."
    }
    """


def generate_chat_currencies_prompt() -> str:
    return """
    {
        "role": "You are an expert text analyst.",
        "task": "Provide a list of non-repeated currencies mentioned in the chat history.",
        "chat_history": "{{ chat_history }}",
        "instructions": [
            "Do not explain what you are doing.",
            "Do not self reference.",
            "Extract only currencies mentioned in the chat conversation.",
            "Showcase the results in a list of non-repeated currencies."
        ],
        "example_output": "USD, EUR, GBP, JPY",
        "max_output_currency": 10,
        "output_format": "A comma-separated list of currencies mentioned in the chat history. Do not include any other text."
    }
    """
