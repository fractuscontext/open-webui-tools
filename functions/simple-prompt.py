"""
title: Simple Prompt
author: fractuscontext
author_url: https://github.com/fractuscontext
license: MIT License
version: 0.8.4-2
tested_on_open_webui_version: v0.8.4

description: Appends configurable instructions to the System Prompt. The default prompt enforces high-density responses and suppresses repetitive "As an AI" disclaimers.
"""

from pydantic import BaseModel, Field
from typing import Optional
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DEFAULT_SYSTEM_PROMPT = """
---
<system_instructions>
  <objective>
    Provide expert, high-density, low-token responses.
  </objective>

  <formatting_rules>
    <rule>TARGET LENGTH: Aim for ~{target_word_count} words. Exceed this limit ONLY if excluding information would make the answer factually incorrect.</rule>
    <rule>HIERARCHY: Use **BOLD CAPS** for section headers. Do NOT use Markdown headers (#).</rule>
    <rule>LISTS: Use unordered lists (-) for items. Use ordered lists (1.) ONLY for sequential steps.</rule>
    <rule>STYLE: Direct, unbiased, active voice. No filler phrases ("Here is", "Certainly", "I hope this helps").</rule>
    <rule>IDENTITY: Do NOT state "As an AI" or "I am an AI" unless explicitly discussing news/events/announcements beyond your knowledge cutoff date.</rule>
  </formatting_rules>

  <execution>
    - Start answer immediately.
    - Remove cohesive tie-words (e.g., "It is important to note", "In conclusion").
    - If premise is false, correct immediately.
  </execution>
</system_instructions>
"""


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level. Higher numbers run last."
        )
        target_word_count: int = Field(
            default=240,
            description="Target word count. The AI will aim for this but may exceed it if necessary for correctness.",
        )
        system_prompt: str = Field(
            default=DEFAULT_SYSTEM_PROMPT,
            description="Instructions appended to the System Prompt. {target_word_count} is replaced with the valve value. Leave blank to disable.",
        )
        additional_prompt: str = Field(
            default="",
            description="Extra instructions appended after system_prompt. Leave blank to disable.",
        )

    def __init__(self):
        self.valves = self.Valves()
        log.info(
            f"🧊 Simple Prompt initialized - Target: {self.valves.target_word_count} words"
        )

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Inlet function handles the request before sending it to the LLM.
        Injects strict formatting instructions into the System Prompt.
        """

        # Each non-blank prompt is appended as its own paragraph
        instruction_prompt = "".join(
            "\n\n"
            + prompt.strip().replace(
                "{target_word_count}", str(self.valves.target_word_count)
            )
            for prompt in (self.valves.system_prompt, self.valves.additional_prompt)
            if prompt.strip()
        )
        if not instruction_prompt:
            return body

        # Ensure messages exist
        if "messages" in body:
            messages = body["messages"]

            # Search for an existing system message
            system_message = next(
                (msg for msg in messages if msg.get("role") == "system"), None
            )

            if system_message:
                # Append instructions to existing system prompt
                if isinstance(system_message.get("content"), str):
                    system_message["content"] += instruction_prompt
                else:
                    system_message["content"] = (
                        str(system_message.get("content", "")) + instruction_prompt
                    )
            else:
                # Create new system prompt if none exists
                new_system_message = {"role": "system", "content": instruction_prompt}
                messages.insert(0, new_system_message)

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        return body
