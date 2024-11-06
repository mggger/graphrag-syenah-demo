import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional

import tiktoken

from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch
from graphrag.query.structured_search.local_search.system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from dataclasses import dataclass
from typing import Any
import pandas as pd

DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}

log = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """A Structured Search Result."""

    response: str | dict[str, Any] | list[dict[str, Any]]
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    context_text: str | list[str] | dict[str, str]
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    latency: float

class FirstCharCallback(BaseLLMCallback):
    def __init__(self):
        self.first_char_time: Optional[float] = None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.first_char_time is None:
            self.first_char_time = time.time()

class CustomSearch(BaseSearch):
    """Search orchestration for local search mode."""

    def __init__(
            self,
            llm: BaseLLM,
            context_builder: LocalContextBuilder,
            token_encoder: tiktoken.Encoding | None = None,
            system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
            response_type: str = "multiple paragraphs",
            callbacks: list[BaseLLMCallback] | None = None,
            llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,
            context_builder_params: dict | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks or []
        self.response_type = response_type

    async def asearch(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> SearchResult:
        start_time = time.time()
        search_prompt = ""

        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )

            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            first_char_callback = FirstCharCallback()
            response = await self.acall_llm(search_messages, [first_char_callback] + self.callbacks, self.llm_params)

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                latency=first_char_callback.first_char_time - start_time if first_char_callback.first_char_time else None
            )

        except Exception:
            log.exception("Exception in _asearch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                latency=None
            )

    async def astream_search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator:
        start_time = time.time()

        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **self.context_builder_params,
        )
        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        search_prompt = self.system_prompt.format(
            context_data=context_text, response_type=self.response_type
        )
        search_messages = [
            {"role": "system", "content": search_prompt},
            {"role": "user", "content": query},
        ]

        yield context_records
        first_char_callback = FirstCharCallback()
        async for response in self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=[first_char_callback] + self.callbacks,
                **self.llm_params,
        ):
            yield response

        yield {"latency": first_char_callback.first_char_time - start_time if first_char_callback.first_char_time else None}

    def search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> SearchResult:
        start_time = time.time()
        search_prompt = ""
        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        log.info("GENERATE ANSWER: %d. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            first_char_callback = FirstCharCallback()
            response = self.call_llm(
                search_messages,
                [first_char_callback] + self.callbacks,
                self.llm_params
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                latency=first_char_callback.first_char_time - start_time if first_char_callback.first_char_time else None
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                latency=None
            )

    async def acall_llm(self, search_messages, callbacks, params):
        return await self.llm.agenerate(
            messages=search_messages,
            streaming=True,
            callbacks=callbacks,
            **params
        )

    def call_llm(self, search_messages, callbacks, params):
        return self.llm.generate(
            messages=search_messages,
            streaming=True,
            callbacks=callbacks,
            **params
        )