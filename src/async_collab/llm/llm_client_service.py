import os
import json
import logging
from enum import Enum

import openai

from async_collab.llm.llm_client import LLMClient


class LLMModelName(Enum):
    gpt4_nano = "gpt-4.1-nano-2025-04-14"
    dev_gpt_4_turbo = "dev-gpt-4-turbo"
    dev_gpt_35_turbo = "dev-gpt-35-turbo"
    dev_gpt_4_turbo_chat_completions = "dev-gpt-4-turbo-chat-completions"
    dev_gpt_4o_2024_05_13 = "dev-gpt-4o-2024-05-13"
    dev_phi3_medium_128k_instruct = "microsoft/Phi-3-medium-128k-instruct"


class MyLLMClient(LLMClient):
    default_model: str

    def __init__(self, default_model: str = "microsoft/Phi-3-medium-128k-instruct"):
        self.default_model = default_model
        
        # Set up logger for LLM responses
        self.llm_response_logger = logging.getLogger("llm_responses")
        self.llm_response_logger.setLevel(logging.INFO)
        
        # Only add handler if it doesn't already exist
        if not self.llm_response_logger.handlers:
            os.makedirs("logs", exist_ok=True)
            handler = logging.FileHandler("logs/llm_responses.jsonl", mode="a")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.llm_response_logger.addHandler(handler)

        for envvar in ["LITELLM_API_KEY", "LITELLM_BASE_URL"]:
            if not os.environ.get(envvar):
                print(f"WARNING: {envvar} environment variable is not set")
        print("INFO: Using LITELLM_BASE_URL: " + os.environ.get("LITELLM_BASE_URL", "http://localhost:7599/v1"))

        self.client = openai.OpenAI(
            api_key=os.environ.get("LITELLM_API_KEY", "no_API_KEY_provided"),
            base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:7599/v1"),
        )


    def send_chat_request(self, request, model) -> dict:
        messages = []
        if len(request["system_instruction"]) > 0:
            messages.append({"role": "system", "content": request["system_instruction"]})
        messages.append({"role": "user", "content": request["prompt"]})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.model_dump()


    def send_request(self, request, model) -> dict:
        prompt = request["prompt"]
        if len(request["system_instruction"]) > 0:
            # print("[MyLLMClient] send_request system_instruction: " + request["system_instruction"])
            prompt = f"{request['system_instruction']}\n{prompt}"

        # print("[MyLLMClient] send_request prompt:\n" + prompt)

        completion_params = {
            "model": model,
            "prompt": prompt,
            "max_tokens": request["max_tokens"],
            "temperature": request["temperature"],
            "top_p": request["top_p"],
            "n": request["n"],
            "stream": request["stream"],
            "logprobs": request["logprobs"],
            "stop": request["stop"],
        }
        response = self.client.completions.create(**completion_params).model_dump()
        response["request"] = completion_params
        self.llm_response_logger.info(json.dumps(response))
        return response


    def get_response_str(
        self,
        user_prompt: str,
        temperature: float = 0,
        max_tokens: int = 2800,
        top_p: float = 0.95,
        system_instruction: str = "",
        stop: str | None = None,
        model: str | None = None,
    ) -> str | None:
        if model is None:
            model = self.default_model
        # if len(system_instruction) > 0:
        #     user_prompt = f"{system_instruction}\n{user_prompt}"

        request_data = {
            "prompt": user_prompt,
            "system_instruction": system_instruction,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "stream": False,
            "logprobs": None,
            "stop": stop,
        }
        response = self.send_request(request_data, model=model)
        if response is None or len(response) == 0 or len(response["choices"]) == 0:
            # print("[MyLLMClient] get_response_str: response is None")
            return None
        # print("[MyLLMClient] get_response_str: returning response")
        # print("[MyLLMClient] get_response_str response: " + response["choices"][0]["message"]["content"])
        # return response["choices"][0]["message"]["content"]
        # print("[MyLLMClient] get_response_str response: " + response["choices"][0]["text"])
        return response["choices"][0]["text"]


llm_client: LLMClient | None = None


def get_llm_client(
    default_model: str = str(LLMModelName.dev_phi3_medium_128k_instruct.value)
) -> LLMClient:
    global llm_client
    if llm_client is None or llm_client.default_model != default_model:
        llm_client = MyLLMClient(default_model=default_model)
    return llm_client


if __name__ == "__main__":
    llm_client = get_llm_client()
    response = llm_client.get_response_str(
        user_prompt="Tell me something about New York.",
        system_instruction="You are a helpful assistant that always provides answers as poems",
    )
    print(response)
