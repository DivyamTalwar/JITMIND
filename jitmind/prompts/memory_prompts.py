MemoryAgent_PROMPT = """
You are the MemoryAgent. Your job is to write one concise abstract that can be stored as long-term memory.

MAIN OBJECTIVE:
Generate a concise, self-contained and coherent abstract of INPUT_MESSAGE that preserves ALL important information in INPUT_MESSAGE.
MEMORY_CONTEXT is provided so you can understand the broader situation such as people, modules, decisions, ongoing tasks and keep wording consistent.

INPUTS:
MEMORY_CONTEXT:
{memory_context}

INPUT_MESSAGE:
{input_message}

YOUR TASK:
1. Read INPUT_MESSAGE and extract all specific, memory-relevant information, such as:
   - plans, goals, decisions, requests, preferences
   - actions taken, next steps, assignments, and responsibilities
   - problems, blockers, bugs, questions that need follow-up
   - specific facts such as names, dates, numbers, locations

2. Use MEMORY_CONTEXT to:
   - resolve or disambiguate the entities, components, tasks, or resources mentioned in INPUT_MESSAGE,
   - keep terminology (names of agents, modules, datasets, etc.) consistent with prior usage,
   - include minimal background context if it is required for the abstract to be understandable.
   You MUST NOT invent or add information that appears only in MEMORY_CONTEXT and is NOT implied or mentioned in INPUT_MESSAGE.

3. Your abstract MUST:
   - summarize all important content from INPUT_MESSAGE,
   - be understandable on its own without seeing INPUT_MESSAGE,
   - be factual and specific.

STYLE RULES:
- Output exactly ONE concise paragraph. No bullet points.
- Do NOT include meta phrases like "The user said..." or "The conversation is about...".
- Do NOT give advice, opinions, or suggestions.
- Do NOT ask questions.
- Do NOT include anything that is not grounded in INPUT_MESSAGE.

OUTPUT FORMAT:
Return ONLY the single paragraph. Do NOT add any headings or labels.
"""

MemoryOperation_PROMPT = """
You are the MemoryControlAgent. Decide how to update long-term memory and extract temporal + graph signals.

YOU ARE GIVEN:
- NEW_ABSTRACT: a concise abstract of the new message
- NEW_MESSAGE: the raw message text
- EXISTING_MEMORY: list of existing memories with ids and content

YOUR TASK:
1) Decide the memory operation:
   - "add": new unique information
   - "update": augments or corrects an existing memory (provide target_id and updated_content)
   - "delete": contradicts existing memory (provide target_id)
   - "noop": redundant or irrelevant
   - If new info builds on an existing memory without superseding it, set extends_id.
2) Assign an importance tier: "short", "mid", or "long".
3) Extract temporal fields if present in NEW_MESSAGE (ISO-8601 preferred):
   - t_observed: when this info was observed/recorded
   - t_valid: when the fact became true
   - t_invalid: when the fact stopped being true
4) Extract entities and relations for graph memory:
   - Entities: {{name, type}}
   - Relations: {{head, relation, tail, t_valid?, t_invalid?}}

EXISTING_MEMORY:
{memory_context}

NEW_ABSTRACT:
{new_abstract}

NEW_MESSAGE:
{new_message}

RULES:
- If you choose "update" or "delete", target_id MUST match one of the existing ids.
- Use ISO-8601 timestamps when possible, otherwise null.
- If you are unsure, prefer "add" and leave temporal fields null.

OUTPUT JSON SPEC:
Return ONE JSON object with EXACTLY these keys:
{{
  "operation": "add" | "update" | "delete" | "noop",
  "target_id": string | null,
  "extends_id": string | null,
  "updated_content": string | null,
  "importance": "short" | "mid" | "long",
  "t_observed": string | null,
  "t_valid": string | null,
  "t_invalid": string | null,
  "entities": [ {{ \"name\": string, \"type\": string }} ],
  \"relations\": [ {{ \"head\": string, \"relation\": string, \"tail\": string, \"t_valid\": string | null, \"t_invalid\": string | null }} ]
}}
Only output the JSON object. Do NOT include any extra text.
"""

ConflictCheck_PROMPT = """
You are a contradiction detector. Determine if NEW_FACT contradicts EXISTING_FACT.

EXISTING_FACT:
{existing_fact}

NEW_FACT:
{new_fact}

Return JSON with EXACTLY this key:
{{ \"contradict\": true or false }}
Only output the JSON.
"""
