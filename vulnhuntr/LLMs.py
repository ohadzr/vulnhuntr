import json
import logging
import re
from json import JSONDecodeError
from typing import List, Union, Dict, Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError
import anthropic
import os, sys
import openai
import dotenv
from llama_cpp import Llama
from langchain_nvidia_ai_endpoints import ChatNVIDIA, Model, NVIDIAEmbeddings, register_model
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from vulnhuntr.utils import extract_between_tags

dotenv.load_dotenv()

log = logging.getLogger(__name__)




class LLMError(Exception):
    """Base class for all LLM-related exceptions."""
    pass

class RateLimitError(LLMError):
    pass

class APIConnectionError(LLMError):
    pass

class APIStatusError(LLMError):
    def __init__(self, status_code: int, response: Dict[str, Any]):
        self.status_code = status_code
        self.response = response
        super().__init__(f"Received non-200 status code: {status_code}")

# Base LLM class to handle common functionality
class LLM:
    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.prev_prompt: Union[str, None] = None
        self.prev_response: Union[str, None] = None
        self.prefill = None

    def _validate_response(self, response_text: str, response_model: BaseModel) -> BaseModel:
        try:
            if self.prefill:
                response_text = self.prefill + response_text
            return response_model.model_validate_json(response_text)
        except ValidationError as e:
            log.warning("Response validation failed", exc_info=e)
            raise LLMError("Validation failed") from e
1
    def _add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def _handle_error(self, e: Exception, attempt: int) -> None:
        log.error(f"An error occurred on attempt {attempt}: {str(e)}", exc_info=e)
        raise e

    def _log_response(self, response: Dict[str, Any]) -> None:
        usage_info = response.usage_metadata
        log.debug("Received chat response", extra={"usage": usage_info})

    def chat(self, user_prompt: str, response_model: BaseModel = None, max_tokens: int = 4096, retry: int = 0) -> Union[BaseModel, str]:
        self._add_to_history("user", user_prompt)
        messages = self.create_messages(user_prompt)
        response = self.send_message(messages, max_tokens, response_model)
        # self._log_response(response)

        response_text = self.get_response(response)
        if response_model:
            try:
                response_text = self._validate_response(response_text, response_model) if response_model else response_text
                log.debug("response validated:", response_text=response_text)
            except Exception as e:
                if retry < 3:
                    print(f'retrying again due to {str(e)}: {retry} retry')
                    return self.chat(user_prompt, response_model,max_tokens, retry + 1)
                else:
                    raise LLMError(f"An unexpected error occurred: {str(e)}") from e
        self._add_to_history("assistant", response_text)
        return response_text

class ChatGPT(LLM):
    def __init__(self, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.model = "meta/llama-3.1-405b-instruct"
        self.function_id = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/eb0b34c6-6d51-4e9f-9d36-c5c26ddc5443"

        log.info("Using model: ", self.model)

        #if self.model not in [model.id for model in ChatNVIDIA.get_available_models()]:
        register_model(Model(
                    id=self.model,
                    model_type="chat",
                    client="ChatNVIDIA",
                    endpoint=self.function_id))

        self.client = ChatNVIDIA(
            model=self.model,
            endpoint=self.function_id,
            api_key=API_KEY,
            temperature=0.3
        )


    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model, retry: int = 0) -> BaseMessage:
        try:
            prompt = "\n".join([f"{msg['role']}:\n{msg['content']}\n" for msg in messages])
            # print("prompt:", prompt)
            response = self.client.invoke(prompt, max_tokens=max_tokens)

            return response
        except Exception as e:
            if retry < 3:
                print(f'retrying again due to {str(e)}: {retry} retry')
                print(response.content)
                self.send_message(messages, max_tokens, response_model, retry + 1)
            else:
                raise LLMError(f"An unexpected error occurred: {str(e)}") from e

    def _clean_response(self, response: str) -> str:

        cleaned_text = extract_between_tags('response_format', response)
        cleaned_text = extract_between_tags('response',cleaned_text)
        # Step 1: Remove markdown code block wrappers
        cleaned_text = cleaned_text.replace('```json','```')
        cleaned_text = extract_between_tags('```', cleaned_text)
        # Step 2: Correctly handle newlines and escaped characters
        # cleaned_text = cleaned_text.replace('\n', '').replace('\\\'', '\'')
        # Step 3: Replace escaped double quotes with regular double quotes
        # cleaned_text = cleaned_text.replace('\\"', '"')
        return cleaned_text.replace('\n', '')

    def get_response(self, response: BaseMessage) -> str:

        # if hasattr(response, 'response_json'):
        #     return json.dumps(response.response_json)
        response = response.content
        cleaned_response = self._clean_response(response)
        # print(cleaned_response)
        return cleaned_response


class LlamaCpp(LLM):
    def __init__(self, system_prompt: str = "", model_path: str = "", n_ctx: int = 4096*4, n_threads: int =1,
                 temperature: float = 0.7, verbose: bool = False) -> None:
        super().__init__(system_prompt)
        if not model_path:
            model_path = os.getenv("LLAMA_MODEL_PATH")
        if not model_path:
            raise ValueError("Model path must be provided either in the constructor or as an environment variable LLAMA_MODEL_PATH")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, temperature=temperature, verbose=verbose)
        self.config = {
            "input_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
            "input_suffix": "<|eot_id|>\n\n",
            "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "assistant_suffix": "<|eot_id|>\n\n",
            #"pre_prompt": "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.",
            "pre_prompt_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
            "pre_prompt_suffix": "<|eot_id|>\n\n",
            "antiprompt": ["<|start_header_id|>", "<|eot_id|>"]
        }

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system",
             "content": f"{self.config['pre_prompt_prefix']}{self.system_prompt}{self.config['pre_prompt_suffix']}"},

            {"role": "user",
             "content": f"{self.config['input_prefix']}{user_prompt}{self.config['input_suffix']}"}

        ]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> str:
        try:
            prompt = "\n".join([f"{msg['content']}" for msg in messages])
            response = ""
            for token in self.llm(prompt, max_tokens=max_tokens, stream=True):
                text = token['choices'][0]['text']
                # Check if the new text contains any antiprompt
                # if any(ap in text for ap in self.config['antiprompt']):
                #     break
                response += text
            return response
        except Exception as e:
            raise LLMError(f"An error occurred while processing the request: {str(e)}") from e

    def get_response(self, response: str) -> str:
        # TODO: maybe clean response?
        return response

    def _log_response(self, response: str) -> None:
        log.debug("Received chat response", extra=response)


class Claude(LLM):
    def __init__(self, system_prompt: str = "") -> None:
        super().__init__(system_prompt)
        self.client = anthropic.Anthropic(max_retries=3, base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))

    def create_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        if "Provide a very concise summary of the README.md content" in user_prompt:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            self.prefill = "{    \"scratchpad\": \"1."
            messages = [{"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": self.prefill}]
        return messages

    def send_message(self, messages: List[Dict[str, str]], max_tokens: int, response_model: BaseModel) -> Dict[str, Any]:
        try:
            # response_model is not used here, only in ChatGPT
            return self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages
            )
        except anthropic.APIConnectionError as e:
            raise APIConnectionError("Server could not be reached") from e
        except anthropic.RateLimitError as e:
            raise RateLimitError("Request was rate-limited") from e
        except anthropic.APIStatusError as e:
            raise APIStatusError(e.status_code, e.response) from e

    def get_response(self, response: Dict[str, Any]) -> str:
        return response.content[0].text.replace('\n', '')