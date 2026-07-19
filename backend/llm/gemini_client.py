"""
Shared Gemini Client — Used by all 3 pipelines + judge.
Uses google.genai SDK with thinking disabled for token efficiency.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)
logger = logging.getLogger(__name__)

_client = None
GEMINI_MODEL_NAME = (os.getenv("GEMINI_MODEL", "gemini-2.5-flash") or "").strip()

# Shared output cap for ALL three pipelines (llm_only, basic_rag, graph_rag).
# Keeping a single source of truth guarantees the comparison is fair — every
# pipeline generates under the identical max_output_tokens ceiling.
MAX_OUTPUT_TOKENS = int((os.getenv("MAX_OUTPUT_TOKENS", "1000") or "1000").strip())

# Shared answer-length policy applied IDENTICALLY to all three pipelines. A
# uniform conciseness instruction is fair (no pipeline is singled out) and keeps
# completion tokens small and equal across pipelines — so the headline token
# reduction reflects the real difference: how much CONTEXT each pipeline feeds
# the model, not how long its answers are.
CONCISE_ANSWER_INSTRUCTION = (
    "Answer concisely in 2-4 sentences. Be direct and factual; do not pad the response."
)

# ── Round 3: ONE shared core system instruction for all 3 pipelines ──────────
# Identical across LLM-Only / RAG / GraphRAG so the comparison is fair. The only
# pipeline-specific addition is the minimal, disclosed evidence/citation clause,
# added identically to the two retrieval pipelines (LLM-Only has nothing to cite).
ROUND3_SYSTEM_CORE = (
    "You are an expert financial-filings analyst. Answer precisely, especially "
    "with numbers. Never refuse. " + CONCISE_ANSWER_INSTRUCTION
)
ROUND3_EVIDENCE_CLAUSE = (
    " Use the provided evidence when it is relevant; if it is insufficient, use "
    "your own expertise. End every answer with a final line exactly of the form "
    "'Sources: [id1, id2]' listing the bracketed evidence ids you actually used "
    "(write 'Sources: []' if you used none)."
)


def _get_client():
    """Lazily initialize Genai client."""
    global _client
    if _client is None:
        from google import genai
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        _client = genai.Client(api_key=api_key)
        logger.info(f"Gemini client initialized: {GEMINI_MODEL_NAME}")
    return _client


def gemini_generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    use_json_schema: bool = False,
) -> dict:
    """Generate a response using Gemini API with thinking disabled."""
    from google.genai import types

    client = _get_client()

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "system_instruction": system_prompt,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }

    # Only use JSON schema for GraphRAG to force 3-bullet format
    if use_json_schema:
        response_schema = {
            "type": "object",
            "properties": {
                "bullets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Exactly 3 bullet points"
                }
            },
            "required": ["bullets"]
        }
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=user_prompt,
        config=config,
    )

    usage = response.usage_metadata
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

    # Parse JSON response if applicable
    import json
    import re
    answer = response.text
    
    if use_json_schema:
        # Try JSON parsing first
        try:
            data = json.loads(response.text)
            bullets = data.get("bullets", [])[:3]
            answer = '\n'.join([f"• {b}" for b in bullets])
            logger.info(f"[OK] JSON schema parsed: {len(bullets)} bullets")
        except:
            # Fallback: extract bullet points using regex
            try:
                # Look for quoted strings that are likely bullets
                matches = re.findall(r'"([^"]{20,})"', response.text)
                if matches:
                    bullets = matches[:3]
                    answer = '\n'.join([f"• {b}" for b in bullets])
                    logger.info(f"[OK] Extracted {len(bullets)} bullets from JSON")
                else:
                    answer = response.text
                    logger.warning(f"[WARN] Could not parse JSON schema, using raw: {response.text[:100]}")
            except:
                answer = response.text
                logger.error(f"[ERR] All parsing failed, raw response: {response.text[:200]}")

    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def count_tokens_gemini(text: str) -> int:
    """Count tokens using Gemini's native tokenizer."""
    client = _get_client()
    result = client.models.count_tokens(model=GEMINI_MODEL_NAME, contents=text)
    return result.total_tokens


_context_encoder = None


def count_context_tokens(text: str) -> int:
    """Count the tokens of a retrieved-context string.

    Uses one local tokenizer (tiktoken cl100k_base) applied IDENTICALLY to every
    pipeline's context, so "context-token reduction" is an apples-to-apples
    metric — and costs zero API calls. It is an approximation of Gemini's
    tokenizer, but because the same counter is used everywhere the *relative*
    reduction across pipelines is valid.
    """
    if not text:
        return 0
    global _context_encoder
    if _context_encoder is None:
        try:
            import tiktoken
            _context_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _context_encoder = False
    if _context_encoder:
        return len(_context_encoder.encode(text))
    # crude fallback if tiktoken is unavailable: ~4 chars per token
    return max(1, len(text) // 4)
