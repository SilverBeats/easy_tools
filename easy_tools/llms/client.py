#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import httpx
from omegaconf import DictConfig
from openai import NOT_GIVEN, OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from ..errors import LLMClientError
from ..utils.constant import LOGGER
from ..utils.timer import Timer
from ..utils.tools import get_base64, is_url


def encrypted_api_key(api_key: str, keep_size: int = 6) -> str:
    if api_key is None:
        return "None"

    if len(api_key) <= keep_size:
        return api_key

    return api_key[:keep_size] + "*" * (len(api_key) - keep_size)


def tasks_num_manage(func):
    def wrapper(self, *args, **kwargs):
        try:
            self._running_tasks_num += 1
            result = func(self, *args, **kwargs)
        finally:
            self._running_tasks_num -= 1
        return result

    return wrapper


@dataclass
class LLMResponse:
    # processed response
    response: Optional[Any] = None
    # original response
    details: Optional[Any] = None
    timecost: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0


@dataclass
class APIConfig:
    model: str
    api_base: str
    api_key: str = "xx"
    proxy: Optional[Union[Dict, DictConfig]] = None
    mask_api_key_keep_size: int = 6

    def to_dict(self):
        return asdict(self)


class ClientBase(ABC):
    @abstractmethod
    def response(self, query: str, **kwargs) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def embedding(self, query, **kwargs) -> LLMResponse:
        raise NotImplementedError


class LLMClient(ClientBase):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "xxx",
        proxy: Optional[Union[Dict, DictConfig]] = None,
        mask_api_key_keep_size: int = 6,
    ):
        if isinstance(proxy, DictConfig):
            proxy = dict(proxy)

        self._model = model
        self._api_key = api_key
        self._encrypted_api_key = encrypted_api_key(
            api_key,
            keep_size=mask_api_key_keep_size,
        )
        self._api_base = api_base
        self._proxy = proxy
        self._running_tasks_num = 0

        if httpx.__version__ <= "0.26.0":
            http_client = httpx.Client(proxies=self._proxy)
        else:
            http_client = httpx.Client(proxy=self._proxy)

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
            http_client=http_client,
        )
        self.residual_credit = 1

    @tasks_num_manage
    def embedding(self, query: str, **kwargs) -> LLMResponse:
        """get embedding vectors for query
        Args:
            query: a sentence or chunk
        """
        try:
            with Timer() as t:
                rst = self._client.embeddings.create(
                    input=query,
                    model=self._model,
                    **kwargs,
                )
                answer = rst.data[0].embedding
            timecost = t.elapsed
            in_tokens = rst.usage.prompt_tokens
            out_tokens = rst.usage.total_tokens - in_tokens
            return LLMResponse(
                response=answer,
                details=rst,
                timecost=timecost,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
        except Exception as e:
            raise LLMClientError(message=str(e))

    @tasks_num_manage
    def response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None,
        stream: bool = False,
        timeout: float = 600,
        **kwargs,
    ) -> LLMResponse:
        try:
            messages = self.generate_prompt(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history,
                images=images,
            )
            with Timer() as t:
                rst = self._client.with_options(
                    timeout=timeout,
                ).chat.completions.create(
                    model=self._model,
                    messages=messages,
                    stream=stream,
                    stream_options={{"include_usage": True}} if stream else NOT_GIVEN,
                    **kwargs,
                )
            timecost = t.elapsed

            if isinstance(rst, ChatCompletion):
                answer = rst.choices[0].message.content
                in_tokens = rst.usage.prompt_tokens
                out_tokens = rst.usage.completion_tokens
            else:
                answer = []
                in_tokens = out_tokens = None
                for chunk in rst:
                    if chunk.usage is None:
                        answer.append(chunk.choices[0].delta.content)
                    else:
                        in_tokens = chunk.usage.prompt_tokens
                        out_tokens = chunk.usage.completion_tokens
                answer = "".join(answer)

            return LLMResponse(
                response=answer,
                details=rst,
                timecost=timecost,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
        except Exception as e:
            raise LLMClientError(message=str(e))

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        images: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Args:
            prompt: string for use role
            system_prompt:  string for system
            history: dialog history
            images: if you use the vl model
        """

        messages = []
        if system_prompt is not None and history is not None:
            LOGGER.warning(
                "system_prompt and history are all NOT None. system_prompt will be ignored!",
            )
            system_prompt = None

        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        if history is not None:
            messages += history

        if images is not None:
            image_infos = []
            for image in images:
                if is_url(image):
                    image_infos.append(
                        {"type": "image_url", "image_url": {"url": image}},
                    )
                else:
                    base64_image = get_base64(image)
                    image_infos.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    )
            content = [{"type": "text", "text": prompt}] + image_infos
            messages.append({"type": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def close(self):
        self._client.close()

    @property
    def running_tasks_num(self):
        return self._running_tasks_num

    @property
    def is_deprecated(self):
        return self.residual_credit <= 0

    @property
    def encrypted_api_key(self):
        return self._encrypted_api_key


class LLMClientGroup:

    def __init__(self, api_configs: List[APIConfig]):
        self.clients = []
        with ThreadPoolExecutor(len(api_configs)) as executor:
            futures = []
            for api_config in api_configs:
                futures.append(
                    executor.submit(LLMClient, **api_config.to_dict()),
                )
            for future in as_completed(futures):
                self.clients.append(future.result())

    @property
    def available_clients(self) -> List[LLMClient]:
        return [client for client in self.clients if not client.is_deprecated]

    def find_available_client(
        self, ignored_clients: Optional[List[LLMClient]] = None
    ) -> Optional[LLMClient]:
        if ignored_clients is None:
            ignored_clients = []

        candidate_clients = [
            client for client in self.available_clients if client not in ignored_clients
        ]

        if len(candidate_clients) > 0:
            return min(candidate_clients, key=lambda x: x.running_tasks_num)

        return None
