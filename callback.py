import streamlit as st
from typing import Any, Union

from graphrag.query.llm.base import BaseLLMCallback
import asyncio
import json

class StreamlitLLMCallback(BaseLLMCallback):
    def __init__(self):
        super().__init__()
        self.container = st.empty()
        self.text = ""

    def on_llm_new_token(self, token: str):
        """Handle when a new token is generated."""
        super().on_llm_new_token(token)
        self.text += token
        self.container.markdown(self.text)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        """Called when LLM starts running."""
        st.write("AI is generating a response...")

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        st.write("AI response complete.")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when LLM errors."""
        st.error(f"An error occurred: {str(error)}")




class StreamingCallback(BaseLLMCallback):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def async_on_llm_new_token(self, token: str, *args, **kwargs):
        await self.queue.put(json.dumps({"content": token}))

    def on_llm_new_token(self, token: str, *args, **kwargs):
        loop = asyncio.get_event_loop()
        loop.create_task(self.async_on_llm_new_token(token, *args, **kwargs))