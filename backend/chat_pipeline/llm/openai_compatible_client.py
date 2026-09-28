"""
OpenAI-Compatible LLM Client for Tool Calling.

Provides async client for interacting with any OpenAI-compatible API.
Handles chat completions, tool calling orchestration, retry logic, and timeout handling.
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable, Tuple, AsyncGenerator
import httpx

from config.settings import settings
from utils.errors import (
    LMStudioError,
    LMStudioConnectionError,
    LMStudioTimeoutError
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class StreamEvent:
    """A single SSE event emitted by process_with_tools_stream()."""
    type: str  # "token" | "citation" | "done" | "error"
    content: Optional[str] = None          # for type="token"
    citations: Optional[List[Any]] = None  # for type="citation"
    session_id: Optional[str] = None       # for type="done"
    metadata: Optional[Dict[str, Any]] = None  # for type="done"
    message: Optional[str] = None          # for type="error"


def parse_thinking_tags(content: str) -> Tuple[str, str]:
    """
    Parse and separate thinking content from the main response.

    Returns:
        Tuple of (thinking_content, main_content)
    """
    if not content:
        return "", ""

    thinking = ""
    main_content = content

    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, content, re.DOTALL | re.IGNORECASE)

    if think_matches:
        thinking = "\n".join(match.strip() for match in think_matches)
        main_content = re.sub(think_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        main_content = re.sub(r'\n{3,}', '\n\n', main_content).strip()

    return thinking, main_content


def parse_text_tool_calls(content: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse tool calls from text content when LLM outputs them as text instead of
    using the native function calling format.

    Handles multiple formats:
    - Standard: <tool_call>{...}</tool_call>
    - Kimi K2.5: <|tool_calls_section_begin|>...<|tool_call_begin|>...<|tool_call_end|>...<|tool_calls_section_end|>
    - Inline JSON: {"name": "tool_name", "arguments": {...}}
    """
    if not content:
        return [], ""

    tool_calls = []
    cleaned_content = content

    # ── Format 1: Kimi K2.5 tool call format ─────────────────────────────────
    # <|tool_calls_section_begin|>
    # <|tool_call_begin|> functions.tool_name:N
    # <|tool_call_argument_begin|> {...}
    # <|tool_call_end|>
    # <|tool_calls_section_end|>
    kimi_section = re.search(
        r'<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>',
        content, re.DOTALL
    )
    if kimi_section:
        section_text = kimi_section.group(1)
        # Each individual tool call
        for call_match in re.finditer(
            r'<\|tool_call_begin\|>\s*functions\.(\w+):\d+\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>',
            section_text, re.DOTALL
        ):
            tool_name = call_match.group(1)
            args_str = call_match.group(2).strip()
            try:
                arguments = json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments)
                    }
                })
                logger.info(f"Parsed Kimi tool call: {tool_name}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Kimi tool call args for {tool_name}: {e}")

        if tool_calls:
            # Strip the entire tool calls section from visible content
            cleaned_content = content[:kimi_section.start()].strip()
            # Also strip any reasoning preamble before the tool call section
            cleaned_content = re.sub(
                r'(I only got one chunk.*?Let me search.*?\n|I should search.*?\n)',
                '', cleaned_content, flags=re.DOTALL
            ).strip()
            return tool_calls, cleaned_content

    # ── Format 2: <tool_call>...</tool_call> ──────────────────────────────────
    pattern1 = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = list(re.finditer(pattern1, content, re.DOTALL | re.IGNORECASE))

    for match in matches:
        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
            tool_calls.append(_create_tool_call(parsed))
            cleaned_content = cleaned_content.replace(match.group(0), '')
            logger.info(f"Parsed text-based tool call: {parsed.get('name')}")
        except json.JSONDecodeError:
            try:
                json_obj = _extract_json_object(json_str)
                if json_obj:
                    parsed = json.loads(json_obj)
                    tool_calls.append(_create_tool_call(parsed))
                    cleaned_content = cleaned_content.replace(match.group(0), '')
                    logger.info(f"Parsed text-based tool call (extracted): {parsed.get('name')}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to parse tool call JSON: {str(e)[:100]}")

    # ── Format 3: Inline JSON ─────────────────────────────────────────────────
    if not tool_calls:
        tool_names = ['rag_retrieval', 'fetch_file']
        for tool_name in tool_names:
            pattern = rf'\{{\s*"name"\s*:\s*"{tool_name}"[^}}]*"arguments"\s*:\s*\{{[^}}]*\}}\s*\}}'
            for match in re.finditer(pattern, content, re.DOTALL):
                try:
                    start_idx = match.start()
                    json_obj = _extract_json_object(content[start_idx:])
                    if json_obj:
                        parsed = json.loads(json_obj)
                        tool_calls.append(_create_tool_call(parsed))
                        cleaned_content = cleaned_content.replace(json_obj, '')
                        logger.info(f"Parsed inline tool call: {tool_name}")
                except Exception as e:
                    logger.warning(f"Failed to parse inline tool call: {e}")

    cleaned_content = cleaned_content.strip()
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    cleaned_content = re.sub(r'</?tool_call>', '', cleaned_content, flags=re.IGNORECASE)

    return tool_calls, cleaned_content


def _extract_json_object(text: str) -> Optional[str]:
    """Extract a complete JSON object from text by finding balanced braces."""
    start = text.find('{')
    if start == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i + 1]

    return None


def _create_tool_call(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Create an OpenAI-format tool call from parsed JSON."""
    tool_name = parsed.get("name", "")
    arguments = parsed.get("arguments", {})

    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
        }
    }


class OpenAICompatibleClient:
    """
    Async client for any OpenAI-compatible API with tool calling support.

    Supports chat completions, streaming, and multi-turn tool calling orchestration.
    Includes connection pooling, retry logic, and comprehensive error handling.

    The constructor requires all config to be passed explicitly — the caller
    decides which settings block (chat, vision, etc.) to use.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        """
        Initialize OpenAI-compatible client.

        Args:
            base_url: API base URL (e.g. http://host:1234/v1)
            api_key: API key; pass empty string for local providers
            model: Model name
            max_tokens: Maximum tokens for responses
            temperature: Default sampling temperature
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Normalise URL: accept either a base URL ("https://host/v1") or a
        # full completions URL ("https://host/v1/chat/completions").
        raw = base_url.rstrip("/")
        if raw.endswith("/chat/completions"):
            self._completions_url = raw
            self.base_url = raw[: -len("/chat/completions")]
        elif raw.endswith("/v1"):
            self.base_url = raw
            self._completions_url = raw + "/chat/completions"
        else:
            self.base_url = raw
            self._completions_url = raw + "/v1/chat/completions"

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            ),
            http2=True
        )

        logger.info(
            f"OpenAICompatibleClient initialized: base_url={self.base_url}, "
            f"completions_url={self._completions_url}, "
            f"model={self.model}, api_key={'set' if self.api_key else 'empty'}"
        )

    def _build_headers(self) -> Dict[str, str]:
        """
        Build request headers.

        Returns Authorization: Bearer header only when api_key is non-empty.
        """
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        await self.client.aclose()
        logger.info("OpenAICompatibleClient closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions in OpenAI format
            temperature: Sampling temperature (defaults to instance setting)
            max_tokens: Maximum tokens (defaults to instance setting)
            tool_choice: Tool choice strategy

        Returns:
            API response dict

        Raises:
            LMStudioConnectionError, LMStudioTimeoutError, LMStudioError
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        logger.debug(
            f"Sending chat completion: {len(messages)} messages, "
            f"{len(tools) if tools else 0} tools"
        )
        logger.info(f"[request] model={self.model} url={self._completions_url} messages={[{'role':m['role'],'len':len(str(m.get('content','')))} for m in messages]} tools={[t['function']['name'] for t in tools] if tools else []} tool_choice={payload.get('tool_choice','none')}")

        return await self._execute_with_retry(
            method="POST",
            endpoint=self._completions_url,
            payload=payload
        )

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        tool_choice: str = "auto"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a chat completion request.

        Yields chunks of the response as they arrive.

        Yields:
            Dict with 'type' and 'content':
            - {"type": "thinking", "content": "..."} for thinking content
            - {"type": "content", "content": "..."} for main content
            - {"type": "done", "full_content": "...", "thinking": "..."} when complete
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        full_content = ""
        thinking_content = ""
        in_thinking = False

        try:
            headers = self._build_headers()
            async with self.client.stream(
                "POST",
                self._completions_url,
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""

                        if reasoning:
                            # Reasoning tokens — accumulate but never stream
                            thinking_content += reasoning
                            full_content += reasoning
                            yield {"type": "thinking", "content": reasoning}

                        if content:
                            full_content += content

                            if "<think>" in content:
                                in_thinking = True

                            if in_thinking:
                                thinking_content += content
                                yield {"type": "thinking", "content": content}
                            else:
                                yield {"type": "content", "content": content}

                            if "</think>" in content:
                                in_thinking = False

                    except json.JSONDecodeError:
                        continue

            thinking, main_content = parse_thinking_tags(full_content)

            yield {
                "type": "done",
                "full_content": main_content,
                "thinking": thinking,
                "raw_content": full_content
            }

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise LMStudioError(f"Streaming failed: {e}") from e

    async def process_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_dispatcher: Any,
        session_id: str,
        max_rounds: int = None,
        status_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat request with multi-turn tool calling orchestration.

        Handles the full tool calling loop until the LLM returns a final response
        or max_rounds is reached.

        Returns:
            Dict with message, tool_calls_made, rounds, total_tokens, finish_reason,
            fetched_files, retrieved_chunks
        """
        max_rounds = max_rounds or settings.max_retrieval_rounds

        logger.info(
            f"Starting tool calling: session={session_id}, max_rounds={max_rounds}"
        )

        conversation_messages = messages.copy()
        tool_calls_made = 0
        rounds = 0
        total_tokens = 0
        fetched_files = []
        retrieved_chunks = []

        if status_callback:
            await status_callback("processing_started", {"session_id": session_id})

        while rounds < max_rounds:
            rounds += 1
            logger.debug(f"Tool calling round {rounds}/{max_rounds}")

            try:
                response = await self.chat_completion(
                    messages=conversation_messages,
                    tools=tools,
                    temperature=self.temperature
                )
            except Exception as e:
                logger.error(f"Chat completion failed in round {rounds}: {e}")
                raise

            usage = response.get("usage", {})
            total_tokens += usage.get("total_tokens", 0)

            choices = response.get("choices", [])
            if not choices:
                raise LMStudioError("No choices returned in API response")

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")

            tool_calls = message.get("tool_calls")
            content = message.get("content") or ""

            if not tool_calls and content:
                parsed_tool_calls, cleaned_content = parse_text_tool_calls(content)
                if parsed_tool_calls:
                    tool_calls = parsed_tool_calls
                    message["content"] = cleaned_content
                    message["tool_calls"] = tool_calls
                    logger.info(f"Parsed {len(tool_calls)} text-based tool call(s)")

            if not tool_calls:
                final_message = message.get("content") or ""
                _, final_message = parse_text_tool_calls(final_message)
                if not final_message.strip():
                    final_message = "I've processed your request and found the relevant information."

                logger.info(
                    f"Tool calling complete: rounds={rounds}, "
                    f"tool_calls={tool_calls_made}, tokens={total_tokens}"
                )

                if status_callback:
                    await status_callback("processing_complete", {
                        "rounds": rounds,
                        "tool_calls": tool_calls_made
                    })

                return {
                    "message": final_message,
                    "tool_calls_made": tool_calls_made,
                    "rounds": rounds,
                    "total_tokens": total_tokens,
                    "finish_reason": finish_reason,
                    "fetched_files": fetched_files,
                    "retrieved_chunks": retrieved_chunks
                }

            logger.info(f"Executing {len(tool_calls)} tool call(s) in round {rounds}")
            conversation_messages.append(message)

            for tool_call in tool_calls:
                tool_calls_made += 1
                tool_name = tool_call.get("function", {}).get("name")

                logger.debug(f"Executing tool: {tool_name}")

                if status_callback:
                    if tool_name == "rag_retrieval":
                        args_str = tool_call.get("function", {}).get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            mode = args.get("mode", "")
                            if "vector" in mode.lower():
                                await status_callback("searching_vectors", {"tool": tool_name})
                            elif "graph" in mode.lower() or "expansion" in mode.lower():
                                await status_callback("matching_graph", {"tool": tool_name})
                            elif "metadata" in mode.lower():
                                await status_callback("checking_metadata", {"tool": tool_name})
                        except Exception:
                            pass

                try:
                    result = tool_dispatcher.dispatch(tool_call, session_id)

                    if tool_name == "fetch_file" and result.get("success"):
                        file_result = result.get("result", {})
                        if file_result:
                            fetched_files.append(file_result)

                    if tool_name == "rag_retrieval" and result.get("success"):
                        rag_result = result.get("result", {})
                        rag_chunks = rag_result.get("chunks", [])
                        if rag_chunks:
                            retrieved_chunks.extend(rag_chunks)
                            logger.info(f"Captured {len(rag_chunks)} chunks from rag_retrieval")

                    tool_result_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_name,
                        "content": json.dumps(result.get("result", {}))
                    }
                    conversation_messages.append(tool_result_message)

                except Exception as e:
                    logger.error(f"Tool execution failed: {tool_name} - {e}")
                    error_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_name,
                        "content": json.dumps({"success": False, "error": str(e)})
                    }
                    conversation_messages.append(error_message)

        # Max rounds reached
        logger.warning(f"Max rounds ({max_rounds}) reached, requesting final response")

        conversation_messages.append({
            "role": "system",
            "content": (
                f"You have reached the maximum number of retrieval rounds ({max_rounds}). "
                "Please provide your best answer based on the information gathered so far."
            )
        })

        try:
            final_response = await self.chat_completion(
                messages=conversation_messages,
                tools=None,
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"Failed to get final response: {e}")
            raise

        choices = final_response.get("choices", [])
        if choices:
            final_message = choices[0].get("message", {}).get("content") or ""
            finish_reason = choices[0].get("finish_reason")
            _, final_message = parse_text_tool_calls(final_message)
            if not final_message.strip():
                final_message = "Based on the information I gathered, I've processed your request."
        else:
            final_message = "Unable to generate response."
            finish_reason = "error"

        usage = final_response.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)

        logger.info(
            f"Tool calling complete (max rounds): rounds={rounds}, "
            f"tool_calls={tool_calls_made}, tokens={total_tokens}"
        )

        if status_callback:
            await status_callback("processing_complete", {
                "rounds": rounds,
                "tool_calls": tool_calls_made,
                "max_rounds_reached": True
            })

        return {
            "message": final_message,
            "tool_calls_made": tool_calls_made,
            "rounds": rounds,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "fetched_files": fetched_files,
            "retrieved_chunks": retrieved_chunks
        }

    async def process_with_tools_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_dispatcher: Any,
        session_id: str,
        original_query: str,
        max_rounds: int = None,
        prompt_builder: Optional[Callable[[List[Dict[str, Any]], str], str]] = None,
    ) -> AsyncGenerator["StreamEvent", None]:
        """
        Process a chat request with tool calling, streaming the final answer.

        Runs tool-calling rounds non-streaming (reusing process_with_tools loop logic),
        then streams the final answer via chat_completion_stream().

        Args:
            prompt_builder: Optional callback that accepts (retrieved_chunks, original_query)
                and returns an enriched system prompt string with local paths already
                converted to HTTP URLs. When provided and chunks were retrieved, it
                replaces the initial system message before the final answer is streamed.

        Yields StreamEvent instances of type: token, citation, done, error.
        """
        max_rounds = max_rounds or settings.max_retrieval_rounds

        try:
            # Run tool-calling rounds non-streaming
            conversation_messages = messages.copy()
            tool_calls_made = 0
            rounds = 0
            total_tokens = 0
            fetched_files = []
            retrieved_chunks = []

            while rounds < max_rounds:
                rounds += 1
                logger.debug(f"Stream tool calling round {rounds}/{max_rounds}")

                try:
                    response = await self.chat_completion(
                        messages=conversation_messages,
                        tools=tools,
                        temperature=self.temperature
                    )
                except Exception as e:
                    logger.error(f"Chat completion failed in round {rounds}: {e}")
                    yield StreamEvent(type="error", message=str(e))
                    return

                logger.info(f"[stream] Round {rounds} request: {len(conversation_messages)} messages, {len(tools)} tools, system_prompt_len={len(conversation_messages[0].get('content','')) if conversation_messages else 0}")
                logger.info(f"[stream] Round {rounds} message roles: {[m['role'] for m in conversation_messages]}")

                usage = response.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)

                choices = response.get("choices", [])
                if not choices:
                    yield StreamEvent(type="error", message="No choices returned in API response")
                    return

                choice = choices[0]
                message = choice.get("message", {})

                tool_calls = message.get("tool_calls")
                # Treat empty list same as no tool calls
                if isinstance(tool_calls, list) and len(tool_calls) == 0:
                    tool_calls = None
                content = message.get("content") or ""

                logger.debug(f"[stream] Round {rounds}: tool_calls={bool(tool_calls)}, content_len={len(content)}, finish_reason={choice.get('finish_reason')}")

                if not tool_calls and content:
                    parsed_tool_calls, cleaned_content = parse_text_tool_calls(content)
                    if parsed_tool_calls:
                        tool_calls = parsed_tool_calls
                        message["content"] = cleaned_content
                        message["tool_calls"] = tool_calls

                if not tool_calls:
                    # No more tool calls — stream the final answer
                    break

                # Execute tool calls
                conversation_messages.append(message)

                for tool_call in tool_calls:
                    tool_calls_made += 1
                    tool_name = tool_call.get("function", {}).get("name")

                    try:
                        result = tool_dispatcher.dispatch(tool_call, session_id)

                        if tool_name == "fetch_file" and result.get("success"):
                            file_result = result.get("result", {})
                            if file_result:
                                fetched_files.append(file_result)

                        if tool_name == "rag_retrieval" and result.get("success"):
                            rag_result = result.get("result", {})
                            rag_chunks = rag_result.get("chunks", [])
                            if rag_chunks:
                                retrieved_chunks.extend(rag_chunks)

                        tool_result_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": tool_name,
                            "content": json.dumps(result.get("result", {}))
                        }
                        conversation_messages.append(tool_result_message)

                    except Exception as e:
                        logger.error(f"Tool execution failed: {tool_name} - {e}")
                        conversation_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": tool_name,
                            "content": json.dumps({"success": False, "error": str(e)})
                        })

            # Emit citation event if we have retrieved chunks
            if retrieved_chunks:
                logger.info(f"[stream] Tool calls complete: {tool_calls_made} calls, {len(retrieved_chunks)} chunks retrieved")
                yield StreamEvent(type="citation", citations=retrieved_chunks)
            else:
                logger.warning(f"[stream] No chunks retrieved after {tool_calls_made} tool calls, {rounds} rounds")

            # ── Build final answer messages: system(chunks) + user only ──────
            # Discard the tool-calling conversation entirely.
            # Final answer gets: enriched system prompt + original user message.
            if prompt_builder and retrieved_chunks:
                try:
                    enriched_prompt = prompt_builder(retrieved_chunks, original_query)
                except Exception as e:
                    logger.warning(f"[stream] prompt_builder failed: {e}")
                    from config.prompts import get_rag_response_prompt
                    enriched_prompt = get_rag_response_prompt()
            else:
                from config.prompts import get_rag_response_prompt
                enriched_prompt = get_rag_response_prompt()

            final_messages = [
                {"role": "system", "content": enriched_prompt},
                {"role": "user", "content": original_query},
            ]
            logger.info(f"[stream] Final answer request: system_len={len(enriched_prompt)}, query_len={len(original_query)}, chunks={len(retrieved_chunks)}, total_messages={len(final_messages)}")
            logger.info(f"[stream] Final messages roles: {[m['role'] for m in final_messages]}")

            # Stream the final answer — pass content tokens through directly.
            # reasoning/thinking tokens are suppressed (handled in chat_completion_stream).
            full_content = ""
            content_token_count = 0

            async for chunk in self.chat_completion_stream(
                messages=final_messages,
                tools=None,
                temperature=self.temperature
            ):
                chunk_type = chunk.get("type")
                if chunk_type == "thinking":
                    # Reasoning tokens — accumulate but never stream
                    full_content += chunk.get("content", "")
                elif chunk_type == "content":
                    token_text = chunk.get("content", "")
                    full_content += token_text
                    if token_text:
                        content_token_count += 1
                        yield StreamEvent(type="token", content=token_text)
                elif chunk_type == "done":
                    logger.info(f"[stream] Final answer complete: {content_token_count} content tokens, {len(full_content)} total chars")
                    yield StreamEvent(
                        type="done",
                        session_id=session_id,
                        metadata={
                            "rounds": rounds,
                            "tool_calls_made": tool_calls_made,
                            "total_tokens": total_tokens,
                            "original_query": original_query
                        }
                    )
                    return

        except Exception as e:
            logger.error(f"process_with_tools_stream error: {e}", exc_info=True)
            yield StreamEvent(type="error", message=str(e))

    async def _execute_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute HTTP request with exponential backoff retry logic.

        Raises:
            LMStudioConnectionError, LMStudioTimeoutError, LMStudioError
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                headers = self._build_headers()

                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=payload,
                    headers=headers
                )

                elapsed_ms = (time.time() - start_time) * 1000
                response.raise_for_status()
                result = response.json()

                logger.debug(
                    f"API request successful: {method} {endpoint} ({elapsed_ms:.2f}ms)"
                )

                return result

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"API timeout (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise LMStudioTimeoutError(
                        f"API request timed out"
                    ) from e

            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(
                    f"Connection failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise LMStudioConnectionError(
                        f"Failed to connect to API at {self.base_url}"
                    ) from e

            except httpx.HTTPStatusError as e:
                last_exception = e
                status = e.response.status_code
                logger.error(
                    f"API error (attempt {attempt + 1}/{self.max_retries}): "
                    f"status={status}, body={e.response.text}"
                )
                if status == 429:
                    # Rate limited — back off and retry
                    wait = self.retry_delay * (3 ** attempt)  # longer backoff for 429
                    logger.warning(f"Rate limited (429), waiting {wait:.1f}s before retry...")
                    await asyncio.sleep(wait)
                    continue
                if 400 <= status < 500:
                    raise LMStudioError(
                        f"API error: {status} - {e.response.text}"
                    ) from e
                if attempt == self.max_retries - 1:
                    raise LMStudioError(
                        f"API error: {status} - {e.response.text}"
                    ) from e

            except Exception as e:
                last_exception = e
                logger.error(
                    f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}",
                    exc_info=True
                )
                if attempt == self.max_retries - 1:
                    raise LMStudioError(f"API request failed: {e}") from e

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.debug(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

        raise LMStudioError(
            f"API request failed after {self.max_retries} attempts"
        ) from last_exception
