PROFILE_UPDATE_PROMPT = """
You are a profile extraction agent. Your job is to update a user's profile from MESSAGE.

MESSAGE:
{message}

Extract updates into three buckets:
- static_updates: stable preferences/facts (name, role, long-term preferences)
- dynamic_updates: temporary or recent context (current project, transient needs)
- traits_updates: personality or style signals (communication preferences)

Rules:
- Only include information explicitly stated or strongly implied in MESSAGE.
- Use concise key/value pairs.
- If nothing new is learned, return empty objects.

Return JSON ONLY:
{{
  "static_updates": {{\"key\": \"value\"}},
  "dynamic_updates": {{\"key\": \"value\"}},
  "traits_updates": {{\"key\": \"value\"}}
}}
"""
