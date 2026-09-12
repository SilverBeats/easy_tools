"""LLM 客户端与客户端组。

封装 OpenAI 兼容的 HTTP 调用，提供：

- :class:`LLMClient` —— 单个 endpoint 的客户端（同步调用，支持流式响应与嵌入）。
- :class:`LLMClientGroup` —— 多 endpoint 容器，按 running 任务数挑选最闲的 client。
- :class:`APIConfig` / :class:`LLMResponse` —— 配置 / 响应 dataclass。
- :func:`encrypted_api_key` —— API key 脱敏。
- :func:`tasks_num_manage` —— 方法装饰器，维护 `self._running_tasks_num` 计数。

底层走 :class:`openai.OpenAI` SDK + :class:`httpx.Client`，自动适配新旧 httpx
proxy kwarg（< 0.28 用 `proxies=`；>= 0.28 用 `proxy=`）。

Example:
    >>> from lwj_tools.llms.client import LLMClient, APIConfig, LLMClientGroup
    >>> configs = [
    ...     APIConfig(model="gpt-4o-mini", api_base="https://api.openai.com/v1",
    ...               api_key="sk-..."),
    ...     APIConfig(model="claude-haiku-4-5", api_base="https://api.anthropic.com/v1",
    ...               api_key="sk-ant-..."),
    ... ]
    >>> group = LLMClientGroup(configs)
    >>> client = group.find_available_client()
    >>> rst = client.response(prompt="Hello, who are you?")
    >>> print(rst.response)
"""

from __future__ import annotations

import functools
import mimetypes
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx
from omegaconf import DictConfig
from openai import NOT_GIVEN, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from packaging.version import Version

from ..common.ids import get_base64, is_url
from ..common.logging import get_logger
from ..common.pojo import DictLike
from ..date.timer import Timer
from ..errors import LLMClientError
from .message import HumanMessage, ImageContent, Message, SystemMessage, TextContent

LOGGER = get_logger("lwj_tools")


def encrypted_api_key(api_key: str, keep_size: int = 6) -> str:
    """对 API key 做掩码：保留前 `keep_size` 个字符，其余替换为 `*`。

    Args:
        api_key: 原始 API key。
        keep_size: 保留前缀长度。

    Returns:
        掩码后的字符串；`api_key is None` 时返回字面量 `"None"`；
        长度 <= `keep_size` 时原样返回。
    """
    if api_key is None:
        return "None"

    if len(api_key) <= keep_size:
        return api_key

    return api_key[:keep_size] + "*" * (len(api_key) - keep_size)


_tasks_lock_init_lock = threading.Lock()


def tasks_num_manage(func):
    """方法装饰器：维护 `self._running_tasks_num` 计数器。

    每次进入被装饰方法时 `+1`，退出时 `-1`（`finally` 保证异常路径也计数）。
    计数器与保护锁在首次调用时惰性创建；`LLMClient` 也会在 `__init__`
    里主动创建，避免竞态。

    Args:
        func: 被装饰的方法，首参应为 `self`。

    Returns:
        包装后的方法。
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_tasks_lock"):
            with _tasks_lock_init_lock:
                if not hasattr(self, "_tasks_lock"):
                    self._tasks_lock = threading.Lock()
                    if not hasattr(self, "_running_tasks_num"):
                        self._running_tasks_num = 0

        with self._tasks_lock:
            self._running_tasks_num += 1
        try:
            return func(self, *args, **kwargs)
        finally:
            with self._tasks_lock:
                self._running_tasks_num -= 1

    return wrapper


def _wrap_error(exc: Exception) -> LLMClientError:
    """把底层异常包装成 :class:`LLMClientError`，尽量保留结构化信息。

    尽力从原始异常提取 `status_code` / `code` / `type` 等属性，附加到
    :class:`LLMClientError` 实例上；若目标类不支持属性赋值则静默跳过。
    """
    error = LLMClientError(message=str(exc))
    for attr in ("status_code", "code", "type"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                setattr(error, attr, value)
            except Exception:
                pass
    return error


@dataclass
class LLMResponse(DictLike):
    """LLM 调用的统一响应结构。

    Attributes:
        response: 解析后的响应内容。文本类调用返回 `str`；流式调用返回拼接后的
            `str`；嵌入调用返回 `list[float]`。
        details: 底层调用细节。非流式为 OpenAI SDK 的原始
            :class:`ChatCompletion`；流式为已消费完的
            :class:`list[ChatCompletionChunk]`（便于事后调试）。
        timecost: 调用耗时（秒）。流式调用包含完整的流消费时间。
        in_tokens: 输入 token 数；不可用时为 `0`。
        out_tokens: 输出 token 数；不可用时为 `0`。
    """

    response: Optional[Any] = None
    details: Optional[Any] = None
    timecost: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0


@dataclass
class APIConfig(DictLike):
    """单个 LLM endpoint 的配置。

    Attributes:
        model: 模型名称。
        api_base: API 基础地址。
        api_key: API 密钥。
        proxy: 代理配置；支持 :class:`dict` 或 :class:`omegaconf.DictConfig`。
        mask_api_key_keep_size: API 密钥脱敏保留前缀长度。
    """

    model: str
    api_base: str
    api_key: str = "xx"
    proxy: Optional[Union[Dict, DictConfig]] = None
    mask_api_key_keep_size: int = 6


class ClientBase(ABC):
    """LLM 客户端抽象基类。"""

    @abstractmethod
    def response(self, query: str, **kwargs) -> LLMResponse:
        """文本生成调用，子类必须实现。"""
        raise NotImplementedError

    @abstractmethod
    def embedding(self, query: str, **kwargs) -> LLMResponse:
        """嵌入向量调用，子类必须实现。"""
        raise NotImplementedError


def _build_http_client(proxy: Optional[Union[Dict, DictConfig]]) -> httpx.Client:
    """构造 :class:`httpx.Client`，自动适配 httpx < 0.28 与 >= 0.28 的 proxy kwarg。

    Args:
        proxy: :class:`dict` 或 :class:`omegaconf.DictConfig` 形式的代理配置；
            `None` 表示不使用代理。

    Returns:
        构造好的 :class:`httpx.Client`。
    """
    if isinstance(proxy, DictConfig):
        proxy = dict(proxy)
    # httpx < 0.28 用 proxies=；>= 0.28 改为 proxy=
    if Version(httpx.__version__) < Version("0.28.0"):
        return httpx.Client(proxies=proxy)
    return httpx.Client(proxy=proxy)


def _encode_image(image: str) -> str:
    """把本地图像路径编码为 data URL；已是 URL 时原样返回。

    MIME 类型按扩展名推断，未识别时退化为 `image/jpeg`。
    """
    if is_url(image):
        return image

    mime, _ = mimetypes.guess_type(image)
    if mime is None:
        mime = "image/jpeg"

    base64_image = get_base64(image)
    return f"data:{mime};base64,{base64_image}"


class LLMClient(ClientBase):
    """单个 LLM endpoint 的客户端（基于 OpenAI 兼容 SDK + httpx）。

    支持文本生成（含流式）、嵌入、密钥脱敏。多 client 场景请用
    :class:`LLMClientGroup` 而不是手动管理多个实例。

    Attributes:
        residual_credit: 预留信用额度字段（当前固定为 `1`，保留以便后续接入扣减逻辑）。
        is_deprecated: 是否被标记为不可用；当前由 :attr:`residual_credit <= 0` 决定。

    Example:
        >>> client = LLMClient(
        ...     model="gpt-4o-mini",
        ...     api_base="https://api.openai.com/v1",
        ...     api_key="sk-...",
        ... )
        >>> rst = client.response(prompt="Hello")
        >>> print(rst.response)
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "xxx",
        proxy: Optional[Union[Dict, DictConfig]] = None,
        mask_api_key_keep_size: int = 6,
    ):
        """初始化客户端。

        Args:
            model: 模型名称。
            api_base: API 基础地址。
            api_key: API 密钥。
            proxy: 代理配置；支持 :class:`dict` 或 :class:`omegaconf.DictConfig`。
            mask_api_key_keep_size: API 密钥脱敏保留前缀长度。
        """
        self._model = model
        self._api_key = api_key
        self._encrypted_api_key = encrypted_api_key(
            api_key,
            keep_size=mask_api_key_keep_size,
        )
        self._api_base = api_base
        self._proxy = proxy

        # 计数器与保护锁：初始化时即创建，避免惰性初始化竞态。
        self._running_tasks_num = 0
        self._tasks_lock = threading.Lock()

        # 显式持有 httpx client 引用：OpenAI SDK 不会接管外部传入的 http_client
        # 生命周期，需要我们自己关闭。
        self._http_client = _build_http_client(proxy)
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
            http_client=self._http_client,
        )
        self.residual_credit = 1

    @tasks_num_manage
    def embedding(self, query: str, **kwargs) -> LLMResponse:
        """获取文本的嵌入向量。

        Args:
            query: 待嵌入的文本。
            **kwargs: 透传给 OpenAI `embeddings.create`。

        Returns:
            :class:`LLMResponse`，`response` 字段为 `list[float]`。

        Raises:
            LLMClientError: 调用失败时包装原始异常。
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
            in_tokens = getattr(rst.usage, "prompt_tokens", 0) or 0
            total = getattr(rst.usage, "total_tokens", in_tokens) or in_tokens
            return LLMResponse(
                response=answer,
                details=rst,
                timecost=timecost,
                in_tokens=in_tokens,
                out_tokens=max(0, total - in_tokens),
            )
        except Exception as e:
            raise _wrap_error(e) from e

    @tasks_num_manage
    def response(
        self,
        prompt: str,
        *,
        system_prompt: Optional[SystemMessage] = None,
        history: Optional[List[Message]] = None,
        images: Optional[List[str]] = None,
        stream: bool = False,
        timeout: float = 600,
        **kwargs,
    ) -> LLMResponse:
        """LLM chat completion 调用。

        Args:
            prompt: 用户 prompt。
            system_prompt: 系统 prompt；与 `history` 同时给定时会被忽略。
            history: 此前对话历史。
            images: 图像 URL 或本地路径列表；本地路径会被读取并 base64 内联。
            stream: 是否流式返回。`True` 时仍会消费完整流并返回拼接好的
                `str`，`details` 为已消费的 chunk 列表。
            timeout: HTTP 超时（秒）。
            **kwargs: 透传给 OpenAI `chat.completions.create`。

        Returns:
            :class:`LLMResponse`。`timecost` 包含完整的流消费时间；
            `in_tokens` / `out_tokens` 不可用时为 `0`。

        Raises:
            LLMClientError: 调用失败时包装原始异常。
        """
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
                    stream_options={"include_usage": True} if stream else NOT_GIVEN,
                    **kwargs,
                )

                if isinstance(rst, ChatCompletion):
                    answer = rst.choices[0].message.content
                    usage = getattr(rst, "usage", None)
                    in_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    out_tokens = getattr(usage, "completion_tokens", 0) or 0
                    details: Any = rst
                else:
                    # 流式：完整消费迭代器，accumulate 内容、token 与原始 chunk。
                    chunks: List[Any] = []
                    answer_parts: List[str] = []
                    in_tokens = 0
                    out_tokens = 0
                    for chunk in rst:
                        chunks.append(chunk)
                        if chunk.usage is not None:
                            in_tokens = chunk.usage.prompt_tokens or 0
                            out_tokens = chunk.usage.completion_tokens or 0
                            continue
                        if not chunk.choices:
                            # usage-only 或 tool_call-only chunk 时 choices 为空
                            continue
                        delta = chunk.choices[0].delta.content
                        answer_parts.append("" if delta is None else delta)
                    answer = "".join(answer_parts)
                    details = chunks

            timecost = t.elapsed
            return LLMResponse(
                response=answer,
                details=details,
                timecost=timecost,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
        except Exception as e:
            raise _wrap_error(e) from e

    def generate_prompt(
        self,
        prompt: str,
        system_prompt: Optional[SystemMessage] = None,
        history: Optional[List[Message]] = None,
        images: Optional[List[str]] = None,
    ) -> List[Dict]:
        """把 prompt / history / images 拼成 OpenAI `messages` 列表。

        当 `system_prompt` 与 `history` 同时给定时，`system_prompt` 会被忽略
        并打印 warning —— 调用方应在传入 :meth:`LLMChainWithValid.__call__` 这类
        入口前自行决定把 system message 放在哪里。

        Args:
            prompt: 用户 prompt。
            system_prompt: 系统 prompt。
            history: 历史消息列表。
            images: 图像 URL 或本地路径列表。

        Returns:
            OpenAI wire-format `messages` 列表（dict 形式）。
        """
        messages: List[Message] = []
        if system_prompt is not None and history is not None:
            LOGGER.warning(
                "system_prompt and history are all NOT None. system_prompt will be ignored!",
            )
            system_prompt = None

        if system_prompt is not None:
            messages.append(system_prompt)

        if history is not None:
            messages += history

        content = prompt
        if images is not None:
            image_infos = [ImageContent(_encode_image(url)) for url in images]
            content = [TextContent(prompt)] + image_infos

        messages.append(HumanMessage(content))
        return [msg.to_dict(True) for msg in messages]

    def close(self) -> None:
        """关闭底层 OpenAI 与 httpx 客户端，释放连接池。

        OpenAI SDK 不会接管外部传入的 `http_client` 生命周期，因此这里显式
        关闭我们持有的 :class:`httpx.Client`。两个 `close()` 均为幂等操作。
        """
        try:
            self._client.close()
        finally:
            self._http_client.close()

    @property
    def running_tasks_num(self) -> int:
        """正在并发执行的任务数（由 :func:`tasks_num_manage` 维护）。"""
        return self._running_tasks_num

    @property
    def is_deprecated(self) -> bool:
        """是否被标记为不可用。"""
        return self.residual_credit <= 0

    @property
    def encrypted_api_key(self) -> str:
        """:func:`encrypted_api_key` 脱敏后的 API key。"""
        return self._encrypted_api_key


class LLMClientGroup:
    """多 endpoint LLM 客户端容器。

    按 `api_configs` 顺序构造 `self.clients`，保持索引一致；
    :meth:`find_available_client` 选择 running 任务数最少的非 deprecated client。

    Example:
        >>> from lwj_tools.llms.client import APIConfig, LLMClientGroup
        >>> group = LLMClientGroup([
        ...     APIConfig(model="gpt-4o-mini", api_base="https://api.openai.com/v1",
        ...               api_key="sk-..."),
        ...     APIConfig(model="claude-haiku-4-5", api_base="https://api.anthropic.com/v1",
        ...               api_key="sk-ant-..."),
        ... ])
        >>> client = group.find_available_client()
    """

    def __init__(self, api_configs: List[APIConfig]):
        """顺序构造 `self.clients`，索引与 `api_configs` 一一对应。

        Args:
            api_configs: :class:`APIConfig` 列表。
        """
        self.clients = [LLMClient(**api_config.to_dict()) for api_config in api_configs]

    @property
    def available_clients(self) -> List[LLMClient]:
        """过滤掉 :attr:`LLMClient.is_deprecated` 为 True 的 client。"""
        return [client for client in self.clients if not client.is_deprecated]

    def find_available_client(
        self,
        ignored_clients: Optional[List[LLMClient]] = None,
    ) -> Optional[LLMClient]:
        """从可用 client 中挑 `running_tasks_num` 最小的一个。

        Args:
            ignored_clients: 本轮调用中需要跳过的 client（一般用于失败重试时屏蔽坏 client）。

        Returns:
            选中的 :class:`LLMClient`；没有可用 client 时返回 `None`。
        """
        if ignored_clients is None:
            ignored_clients = []

        candidate_clients = [
            client for client in self.available_clients if client not in ignored_clients
        ]

        if len(candidate_clients) > 0:
            return min(candidate_clients, key=lambda x: x.running_tasks_num)

        return None
