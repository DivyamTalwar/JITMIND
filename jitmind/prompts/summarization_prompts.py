HIERARCHICAL_SUMMARY_PROMPT = """
You are a summarization model. Write a concise, factual summary that captures the key information
from the provided CLUSTER_CONTENT. Preserve names, numbers, and critical details. Avoid speculation.

CLUSTER_CONTENT:
{cluster_content}

OUTPUT:
Return ONLY the summary paragraph.
"""

CONSOLIDATION_PROMPT = """
You are consolidating multiple short-term memories into one high-quality memory.
Merge duplicates, remove redundancy, and preserve all distinct facts.

MEMORIES:
{cluster_content}

OUTPUT:
Return ONLY the consolidated memory paragraph.
"""
