def generate_keywords_prompt():
    return """{
        "task": "Extract up to 10 key phrases from the provided article_content without explanation.",
        "max_phrases": 10,
        "notes": [
            "DO NOT include currency in the output key phrases (e.g. JPY, USD, -).",
            "DO NOT REPEAT the same key phrase twice!",
            "The key phrase must be extracted from the {{ article_content }}. NEVER hallucinate! This is very important!"
        ],
        "examples": [
            {
            "article": "The latest smartphone from Apple features an improved camera, faster processor, and enhanced battery life. With a sleek new design, it's set to revolutionize the mobile market.",
            "expected_output": "Apple smartphone; improved camera; faster processor; enhanced battery life; sleek design; mobile market"
            },
            {
            "article": "The city of Paris is known for its stunning architecture, art museums, and romantic riverside walks. Visitors can explore the Eiffel Tower, Louvre Museum, and enjoy the local cuisine.",
            "expected_output": "Paris; Eiffel Tower; Louvre Museum; romantic riverside walks; local cuisine"
            }
        ],
        "article_content": "{% raw %}{{ article_content }}{% endraw %}",
        "output_format": "List of key phrases separated by semi-colon"
    }"""


def generate_currencies_prompt():
    return """{
        "task": "Extract up to 10 main currency acronyms from the provided article_content without any explanation",
        "max_items": 10,
        "notes": [
            "ONLY include currency acronyms in the output (e.g. JPY, USD, EUR).",
            "Even if an acronym is not explicitly present in the article_content, infer the correct currency from context.",
            "DO NOT REPEAT the same currency twice.",
            "If no currency can be found, return an empty string (no whitespace or line-breaks).",
            "The extracted currency must be derived from the {{ article_content }}. NEVER hallucinate!"
        ],
        "examples": [
            {
            "article": "The Japanese yen (JPY) has seen a significant decline in value against the US dollar (USD) in recent months, affecting the country's exports. The Bank of Japan has taken measures to stabilize the economy, but the impact on the global market remains uncertain. Meanwhile, the European Central Bank has announced plans to maintain its interest rates, which may influence the euro (EUR) and pound sterling (GBP) exchange rates.",
            "expected_output": "JPY; USD; EUR; GBP"
            },
            {
            "article": "Travelers visiting Australia can expect to pay in Australian dollars (AUD) for their accommodations and tourist activities. The country's economy is also closely tied to the Chinese yuan (CNY), as China is one of its largest trading partners. Additionally, the New Zealand dollar and Singapore dollar are widely accepted in some tourist areas.",
            "expected_output": "AUD; CNY; NZD; SGD"
            },
            {
            "article": "The new restaurant in town has been getting rave reviews for its unique menu and cozy atmosphere. The chef's use of locally sourced ingredients has been praised by food critics and customers alike. The restaurant's outdoor seating area is also a popular spot for people-watching.",
            "expected_output": ""
            }
        ],
        "article_content": "{% raw %}{{ article_content }}{% endraw %}",
        "output_format": "List of currencies separated by semi-colon"
    }"""


def generate_topics_prompt():
    return """{
        "task": "Extract up to 10 topics from the provided article_content without any explanation",
        "max_items": 10,
        "notes": [
            "The topics must be extracted from the {{ article_content }}. NEVER hallucinate!",
            "The topics must be in the same language as the article_content.",
            "The topics must be in the same language as the article_content.",
        ]
    }"""


def generate_instruments_prompt():
    return """{
        "task": "Extract up to 10 financial instruments explicitly discussed in article_content",
        "notes": [
            "Examples: options, swaps, CDS, futures, forwards, spot, convertible bonds, ELN, ETFs",
            "Prefer instrument types over generic words like 'derivatives' unless only generic is present",
            "Do not include currencies here",
            "Return empty string if none"
        ],
        "article_content": "{% raw %}{{ article_content }}{% endraw %}",
        "output_format": "List of instruments separated by semi-colon"
    }"""

