"""
title: Token Saver Reloaded
author: fractuscontext
author_url: https://github.com/fractuscontext
license: MIT License
version: 0.9.4
tested_on_open_webui_version: v0.9.4

requirements: httpx>=0.24.0, rank-bm25, sentence-transformers, torch

description: A multi-function filter that optimizes token usage by trimming history, pruning images, removing filler words, injecting historical timestamps, and using AI/BM25 to retrieve forgotten messages beyond the context window.
"""

import logging
import datetime
import re
import time
import copy
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict
from pydantic import BaseModel, Field

try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

try:
    import httpx
except ImportError:
    httpx = None

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        n_last_messages: int = Field(
            default=50,
            description="Total number of message bubbles to retain in standard context.",
        )
        supplementary_messages_to_be_sent: int = Field(
            default=3,
            description="Number of extra relevant past messages to retrieve beyond the n_last_messages cutoff. Set to 0 to disable lookup.",
        )
        use_bm25_instead_of_sentence_transformer: bool = Field(
            default=False,
            description="If True, uses lightweight BM25 exact-keyword search. If False, uses heavy PyTorch semantic meaning search.",
        )
        debug_mode: bool = Field(
            default=False,
            description="If True, emits detailed diagnostics (library status, forgotten user message contents, evaluation time) to the server log as WARNING.",
        )
        ignore_images_after_n_messages: int = Field(
            default=10,
            description="Retain images only for the last N messages. Older images are stripped.",
        )
        api_base_url: str = Field(
            default="http://127.0.0.1:8080",
            description="Base URL for the Open WebUI API to fetch message timestamps.",
        )
        inject_system_instructions: bool = Field(
            default=True,
            description="Append instructions to the System Prompt explaining how to use time data.",
        )
        enable_token_optimizer: bool = Field(
            default=False,
            description="Remove common filler words from user messages to reduce token usage.",
        )
        min_words_to_optimize: int = Field(
            default=60,
            description="Skip optimization for messages shorter than this word count.",
        )
        max_words_to_optimize: int = Field(
            default=2000,
            description="Skip optimization for extremely long messages.",
        )
        optimize_current_only: bool = Field(
            default=True,
            description="Only optimize the current (last) message.",
        )

    # ---------------------------------------------------------
    # STATIC CONSTANTS & REGEX COMPILATION
    # ---------------------------------------------------------
    _RAW_FILTER_LIST = (
        "just",
        "really",
        "very",
        "actually",
        "basically",
        "literally",
        "simply",
        "quite",
        "rather",
        "somewhat",
        "fairly",
        "kind of",
        "sort of",
        "type of",
        "thing",
        "stuff",
        "lots of",
        "bunch of",
        "um",
        "uh",
        "hmm",
        "lol",
        "lmao",
        "bruh",
        "ngl",
        "tbh",
        "imo",
        "imho",
        "lowkey",
        "highkey",
        "deadass",
        "omg",
        "smh",
        "tho",
        "kinda",
        "sorta",
        "af",
        "sus",
        "periodt",
        "yeet",
        "stan",
        "clout",
        "salty",
        "bussin",
        "finna",
        "yolo",
        "fomo",
        "synergy",
        "game-changer",
        "world-class",
        "best-in-class",
        "next-level",
        "low-hanging-fruit",
        "boil-the-ocean",
        "circle-back",
        "take-offline",
        "touch-base",
        "move-the-needle",
        "deep-dive",
        "blue-sky",
        "thought-leader",
        "holistic",
        "value-add",
    )

    _SORTED_FILTER = sorted(_RAW_FILTER_LIST, key=len, reverse=True)
    _BAD_WORDS_RE = re.compile(
        r"(?i)\b(?:"
        + "|".join(re.escape(w) for w in _SORTED_FILTER)
        + r")\b(?:[,]+)?[ \t]?"
    )
    _WHITESPACE_RE = re.compile(r"[ \t]+")
    _GRAMMAR_FIX_RE = re.compile(r"\s+([,.;?!])|([,.;?!])\s*\2+")

    _PROTECTED_PATTERN = re.compile(
        r"("
        r"```[\s\S]*?```|"
        r"`[^`\n]+`|"
        r'"[^"\n]*"|'
        r"!\[.*?\]\(.*?\)|"
        r"\[.*?\]\(.*?\)|"
        r"^>.*$|"
        r"^\s*[-*+]\s+|"
        r"^\s*\d+\.\s+"
        r")",
        re.MULTILINE,
    )

    _RE_THOUGHT = re.compile(r'<details\s+type="reasoning".*?</details>', re.DOTALL)
    _RE_SYS_CONTEXT = re.compile(r"\n\n---\n\*\*System Context:\*\*.*$", re.DOTALL)
    _RE_HISTORY_TAG = re.compile(r"<message_time>.*?</message_time>\s*", re.DOTALL)

    def __init__(self):
        self.valves = self.Valves()
        self._history_cache: Dict[str, Tuple[float, Dict]] = {}
        self._cache_ttl = 90
        self._encoder = None
        self._bm25_class = None
        log.info("🐰 White Rabbit initialized - optimized logic ready!")

    # --- SMART LOOKUP ENGINES ---

    def _load_sentence_transformer(self):
        if self._encoder is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                device = (
                    "mps"
                    if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                log.info(f"Loading SentenceTransformer on {device}...")
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            except ImportError:
                log.error("sentence-transformers or torch is not installed!")
        return self._encoder

    def _load_bm25(self):
        if self._bm25_class is None:
            try:
                from rank_bm25 import BM25Okapi

                self._bm25_class = BM25Okapi
            except ImportError:
                log.error("rank_bm25 is not installed!")
        return self._bm25_class

    def _get_recovery_timestamp(
        self, msg: dict, history_lookup: dict, user_timezone: str
    ) -> str:
        """
        Peek at the history_lookup for a recovered message's timestamp without
        consuming it (ts_list[-1] not pop()), so Step 7's injection loop can
        still pop() for messages inside the kept window.
        """
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                [i.get("text", "") for i in content if i.get("type") == "text"]
            )
        norm = self._normalize_text(content)
        ts_list = history_lookup.get(norm)
        if ts_list:
            ts_str = self._format_timestamp(ts_list[-1], user_timezone)
            return f" <message_time>{ts_str}</message_time>"
        return ""

    def _perform_smart_lookup(
        self,
        all_messages: list,
        cutoff_idx: int,
        current_query: str,
        history_lookup: dict,
        user_timezone: str,
        kept_user_content: Set[str],
    ) -> str:
        if cutoff_idx <= 0 or len(all_messages) <= cutoff_idx:
            return ""
        forgotten_messages = all_messages[:-cutoff_idx]

        if not forgotten_messages or self.valves.supplementary_messages_to_be_sent <= 0:
            return ""

        corpus_texts = []
        valid_messages = []
        for msg in forgotten_messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    [i.get("text", "") for i in content if i.get("type") == "text"]
                )
            if content.strip():
                corpus_texts.append(content)
                valid_messages.append(msg)

        if not corpus_texts:
            return ""

        retrieved_context = "\n\n<recovered_memory>\nRelevant past context from outside the standard window:\n"
        found_match = False

        if self.valves.use_bm25_instead_of_sentence_transformer:
            BM25Class = self._load_bm25()
            if not BM25Class:
                return ""

            tokenized_corpus = [t.lower().split() for t in corpus_texts]
            bm25 = BM25Class(tokenized_corpus)
            tokenized_query = current_query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_n_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: self.valves.supplementary_messages_to_be_sent]

            for idx in top_n_indices:
                if scores[idx] > 0.5:
                    # CHANGED: skip if this message is already in the kept window
                    if self._normalize_text(corpus_texts[idx]) in kept_user_content:
                        continue
                    ts_tag = self._get_recovery_timestamp(
                        valid_messages[idx], history_lookup, user_timezone
                    )
                    retrieved_context += f"- [USER]{ts_tag}: {corpus_texts[idx]}\n"
                    found_match = True
        else:
            encoder = self._load_sentence_transformer()
            if not encoder:
                return ""

            try:
                from sentence_transformers import util

                corpus_embeddings = encoder.encode(corpus_texts, convert_to_tensor=True)
                query_embedding = encoder.encode(current_query, convert_to_tensor=True)
                hits = util.semantic_search(
                    query_embedding,
                    corpus_embeddings,
                    top_k=self.valves.supplementary_messages_to_be_sent,
                )[0]

                for hit in hits:
                    if hit["score"] > 0.30:
                        idx = hit["corpus_id"]
                        # CHANGED: skip if this message is already in the kept window
                        if self._normalize_text(corpus_texts[idx]) in kept_user_content:
                            continue
                        ts_tag = self._get_recovery_timestamp(
                            valid_messages[idx], history_lookup, user_timezone
                        )
                        retrieved_context += f"- [USER]{ts_tag}: {corpus_texts[idx]}\n"
                        found_match = True
            except Exception as e:
                log.error(f"Semantic search failed: {e}")

        retrieved_context += "</recovered_memory>\n"
        return retrieved_context if found_match else ""

    # --- CORE PIPELINE LOGIC ---

    def _format_timestamp(self, ts: float, tz_name: str = "UTC") -> str:
        if not ts:
            return ""
        try:
            dt_utc = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
            target_tz = zoneinfo.ZoneInfo(tz_name)
            return dt_utc.astimezone(target_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        text = self._RE_THOUGHT.sub("", text)
        text = self._RE_SYS_CONTEXT.sub("", text)
        text = self._RE_HISTORY_TAG.sub("", text)
        return " ".join(text.split())

    async def _fetch_history_map(
        self, chat_id: str, token: str
    ) -> Dict[str, List[float]]:
        if not httpx or not token:
            return {}
        now = time.time()
        if (
            chat_id in self._history_cache
            and now - self._history_cache[chat_id][0] < self._cache_ttl
        ):
            return self._history_cache[chat_id][1]

        url = f"{self.valves.api_base_url.rstrip('/')}/api/v1/chats/{chat_id}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code != 200:
                    return {}
                data = response.json()
                chat_data = (
                    data[0].get("chat", {})
                    if isinstance(data, list) and len(data) > 0
                    else (data.get("chat", {}) if "chat" in data else data)
                )
                msgs_dict = chat_data.get("history", {}).get("messages", {})

                lookup = defaultdict(list)
                for msg in sorted(
                    msgs_dict.values(), key=lambda x: x.get("timestamp", 0)
                ):
                    content = msg.get("content", "")
                    ts = msg.get("timestamp")
                    if isinstance(content, list):
                        content = " ".join(
                            [
                                i.get("text", "")
                                for i in content
                                if i.get("type") == "text"
                            ]
                        )
                    if content and ts:
                        norm_content = self._normalize_text(content)
                        if norm_content:
                            lookup[norm_content].append(ts)

                self._history_cache[chat_id] = (now, lookup)
                return lookup
        except Exception as e:
            log.debug(f"Timestamp fetch failed: {e}")
            return {}

    def _filter_text_optimized(self, text: str) -> str:
        if not text:
            return ""
        chunks = self._PROTECTED_PATTERN.split(text)
        result_builder = []
        for i, chunk in enumerate(chunks):
            if i % 2 == 1 or not chunk.strip():
                result_builder.append(chunk)
                continue
            scrubbed = self._BAD_WORDS_RE.sub(" ", chunk)
            scrubbed = self._GRAMMAR_FIX_RE.sub(r"\1", scrubbed)
            result_builder.append(self._WHITESPACE_RE.sub(" ", scrubbed))
        return "".join(result_builder)

    def _strip_history_tags(self, content: str) -> str:
        """Strip any <message_time> tags the model leaked into its own response."""
        return self._RE_HISTORY_TAG.sub("", content).lstrip()

    async def outlet(self, body, __user__=None, __metadata__=None, __request__=None):
        messages = body.get("messages", [])
        # Only process the LAST assistant message (the one just generated)
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = self._strip_history_tags(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            item["text"] = self._strip_history_tags(item["text"])
                break  # <-- stop after the first (last) assistant message
        return body

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __metadata__: Optional[dict] = None,
        __request__: Optional[object] = None,
    ) -> dict:
        start_time = time.time()
        variables = __metadata__.get("variables", {}) if __metadata__ else {}
        user_timezone = variables.get("{{CURRENT_TIMEZONE}}", "UTC")

        messages = copy.deepcopy(body.get("messages", []))
        system_prompt = next((m for m in messages if m.get("role") == "system"), None)
        conversation_history = [m for m in messages if m.get("role") != "system"]

        # 1. TRUNCATE
        cutoff = self.valves.n_last_messages
        kept_messages = (
            conversation_history[-cutoff:] if cutoff > 0 else conversation_history
        )
        reachable_history_count = max(0, len(conversation_history) - cutoff)

        # 2. DEBUG MODE — emits to server log only, nothing injected into prompt
        if self.valves.debug_mode:
            torch_installed = False
            try:
                import torch
                import sentence_transformers

                torch_installed = True
            except ImportError:
                pass

            bm25_installed = False
            try:
                import rank_bm25

                bm25_installed = True
            except ImportError:
                pass

            current_time = variables.get(
                "{{CURRENT_DATETIME}}",
                self._format_timestamp(time.time(), user_timezone),
            )

            forgotten_lines = []
            if reachable_history_count > 0:
                forgotten_msgs = [
                    m
                    for m in conversation_history[:-cutoff]
                    if m.get("role") != "system"
                ]
                for m in forgotten_msgs:
                    if m.get("role") != "user":
                        continue
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            [
                                i.get("text", "")
                                for i in content
                                if i.get("type") == "text"
                            ]
                        )
                    forgotten_lines.append(
                        f"    [USER]: {content[:300]}{'...' if len(content) > 300 else ''}"
                    )

            log.warning(
                "\n🐰 [DEBUG MODE]\n"
                f"  PyTorch & SentenceTransformers : {torch_installed}\n"
                f"  Rank-BM25                      : {bm25_installed}\n"
                f"  Forgotten messages             : {reachable_history_count}\n"
                f"  Filter evaluation time         : {current_time}\n"
                + (
                    "  Forgotten user messages:\n" + "\n".join(forgotten_lines)
                    if forgotten_lines
                    else ""
                )
            )

        # 3. FETCH TIMESTAMPS EARLY — shared by smart lookup (Step 4) and
        #    history tag injection (Step 7)
        history_lookup = defaultdict(list)
        if __metadata__ and __request__:
            chat_id = __metadata__.get("chat_id")
            auth_header = ""
            if hasattr(__request__, "headers"):
                auth_header = __request__.headers.get("authorization", "")
            token = (
                auth_header.split(" ")[1]
                if auth_header.lower().startswith("bearer ")
                else ""
            )
            if chat_id and token:
                history_lookup = await self._fetch_history_map(chat_id, token)

        # 4. SMART LOOKUP
        if (
            self.valves.supplementary_messages_to_be_sent > 0
            and kept_messages
            and len(conversation_history) > cutoff
        ):
            current_msg_content = kept_messages[-1].get("content", "")
            if isinstance(current_msg_content, list):
                current_msg_content = " ".join(
                    [
                        i.get("text", "")
                        for i in current_msg_content
                        if i.get("type") == "text"
                    ]
                )

            if isinstance(current_msg_content, str):
                # CHANGED: build deduplication set of normalized kept-window
                # user messages before calling smart lookup
                kept_user_content: Set[str] = set()
                for m in kept_messages:
                    if m.get("role") != "user":
                        continue
                    c = m.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(
                            [i.get("text", "") for i in c if i.get("type") == "text"]
                        )
                    norm = self._normalize_text(c)
                    if norm:
                        kept_user_content.add(norm)

                recovered_memory = self._perform_smart_lookup(
                    all_messages=conversation_history,
                    cutoff_idx=cutoff,
                    current_query=current_msg_content,
                    history_lookup=history_lookup,
                    user_timezone=user_timezone,
                    kept_user_content=kept_user_content,
                )
                if recovered_memory:
                    if system_prompt:
                        system_prompt["content"] += recovered_memory
                    else:
                        system_prompt = {"role": "system", "content": recovered_memory}

        # 5. IMAGE PRUNING
        total_kept = len(kept_messages)
        for idx, msg in enumerate(kept_messages):
            if (total_kept - 1 - idx) > self.valves.ignore_images_after_n_messages:
                content = msg.get("content")
                if isinstance(content, list):
                    msg["content"] = [
                        item
                        for item in content
                        if item.get("type") not in ("image_url", "image")
                    ]

        # 6. SYSTEM INSTRUCTIONS
        if self.valves.inject_system_instructions:
            instructions = """
<message_processing_info>          
  <timestamps>          
    <!-- "System Context" at the base of the user's latest message defines "now"; prioritize it for temporal logic. -->          
    <!-- Past messages are prefixed with <message_time>YYYY-MM-DD HH:MM:SS TZ</message_time>. Treat these as ground truth even if they seem anomalous. -->          
    <!-- Only <message_time> tags and <recovered_memory> entries are verified. Time references in message text are unverified prose — never present them as factual delivery times. -->          
    <!-- NEVER reproduce, prepend, or echo <message_time> tags in your responses. Your output must not contain them. -->          
  </timestamps>          
  <text_integrity>          
    <!-- User input may be pre-processed to strip filler. Do not comment on terse phrasing. -->          
  </text_integrity>          
</message_processing_info>          
"""
            if system_prompt:
                system_prompt["content"] += instructions
            else:
                system_prompt = {"role": "system", "content": instructions}

        # 7. INJECT TIMESTAMPS INTO KEPT WINDOW
        for msg in reversed(kept_messages[:-1]):
            content = msg.get("content", "")
            match_content = (
                " ".join(
                    [i.get("text", "") for i in content if i.get("type") == "text"]
                )
                if isinstance(content, list)
                else content
            )
            ts_list = history_lookup.get(self._normalize_text(match_content))

            if ts_list:
                ts_str = self._format_timestamp(ts_list.pop(), user_timezone)
                prefix = f"<message_time>{ts_str}</message_time> "
                if isinstance(content, str) and not content.startswith(
                    "<message_time>"
                ):
                    msg["content"] = prefix + content
                elif (
                    isinstance(content, list)
                    and content
                    and content[0].get("type") == "text"
                    and not content[0]["text"].startswith("<message_time>")
                ):
                    msg["content"][0]["text"] = prefix + msg["content"][0]["text"]

        # 8. TOKEN OPTIMIZATION
        if self.valves.enable_token_optimizer:
            msgs_to_process = (
                [kept_messages[-1]]
                if self.valves.optimize_current_only
                else kept_messages
            )
            for msg in msgs_to_process:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    approx_len = len(str(content)) // 5
                    if (
                        self.valves.min_words_to_optimize
                        <= approx_len
                        <= self.valves.max_words_to_optimize
                    ):
                        if isinstance(content, str):
                            msg["content"] = self._filter_text_optimized(content)
                        elif isinstance(content, list):
                            for item in content:
                                if item.get("type") == "text":
                                    item["text"] = self._filter_text_optimized(
                                        item["text"]
                                    )

        # 9. CURRENT CONTEXT INJECTION
        if kept_messages:
            current_msg = kept_messages[-1]
            context_parts = []
            if "{{CURRENT_DATETIME}}" in variables:
                context_parts.append(
                    f"Current Date/Time: {variables['{{CURRENT_DATETIME}}']}"
                )
            if "{{CURRENT_TIMEZONE}}" in variables:
                context_parts.append(f"Timezone: {variables['{{CURRENT_TIMEZONE}}']}")

            if context_parts:
                context_block = (
                    "\n\n---\n**System Context:**\n"
                    + "\n".join(context_parts)
                    + "\n---"
                )
                if isinstance(current_msg["content"], str):
                    current_msg["content"] += context_block
                elif isinstance(current_msg["content"], list):
                    current_msg["content"].append(
                        {"type": "text", "text": context_block}
                    )

        # 10. REASSEMBLE
        final_messages = [system_prompt] if system_prompt else []
        final_messages.extend(kept_messages)
        body["messages"] = final_messages

        if log.isEnabledFor(logging.INFO):
            log.info(f"🐰✨ Logic complete ({time.time() - start_time:.4f}s)")

        return body
