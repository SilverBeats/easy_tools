"""LLM 调用编排层。

把 :class:`lwj_tools.llms.client.LLMClient` / :class:`lwj_tools.llms.prompt.PromptTemplate`
组合成"参数 → prompt → LLM 调用 → 解析 → 结果"的工作流：

- :class:`ChainResult` —— 单次调用的结果记录（请求参数、prompt、响应、解析结果、
  error、状态码、token 数、耗时、所用 client 等）。
- :class:`BaseLLMChain` —— 抽象基类，提供 client 选择、API 参数拼接、单次调用
  与共享重试原语。
- :class:`LLMChain` —— 单轮 chain：调一次 LLM 即返回。
- :class:`LLMChainWithValid` —— 多轮校验 chain：把 valid 失败作为新 prompt 反馈
  给模型，让模型自纠错后重试，最多 ``max_chat_turns`` 轮。

Example:
    >>> from lwj_tools.llms.chain import LLMChain, LLMChainWithValid
    >>> from lwj_tools.llms.client import LLMClientGroup, APIConfig
    >>> from lwj_tools.llms.prompt import PromptTemplate
    >>> group = LLMClientGroup([
    ...     APIConfig(model="gpt-4o-mini", api_base="https://api.openai.com/v1",
    ...               api_key="sk-..."),
    ... ])
    >>> class MyPrompt(PromptTemplate):
    ...     def generate_fn(self, topic: str) -> str:
    ...         return f"Tell me about {topic} in one sentence."
    ...     def parse_fn(self, response: str) -> str:
    ...         return response.strip()
    >>> chain = LLMChain(group, MyPrompt())
    >>> result = chain("LLM evaluation")
    >>> result.result
"""
import threading
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Union

from ..common.pojo import DictLike
from ..date.timer import Timer
from ..errors import (
    SUCCEED_CODE,
    LLMClientError,
    PromptTemplateGeneratingError,
    PromptTemplateValidError,
)
from .client import LLMClient, LLMClientGroup
from .message import AssistantMessage, HumanMessage, Message, SystemMessage
from .prompt import PromptTemplate

warnings.simplefilter(action="once", category=UserWarning)


@dataclass
class ChainResult(DictLike):
    """单次 chain 调用的结果记录。

    Attributes:
        api_params: 实际送出的 API 参数（含 defaults + 调用时覆盖值）。
        prompt_args: 传给 :meth:`PromptTemplate.generate_prompt` 的原始参数。
        prompt: 渲染后的 prompt 字符串。
        prompt_template: 使用的 :class:`PromptTemplate` 实例。
        response: LLM 原始响应（通常是 ``str``，流式时为 :class:`Generator`）。
        result: :meth:`PromptTemplate.parse` 的结果。
        reasoning_content: 推理类模型的思维链内容（如有）。
        error: 调用过程中的异常（成功时为 ``None``）。
        status_code: :data:`errors.SUCCEED_CODE` 或其他错误码。
        timecost: 总耗时（秒）。
        in_tokens: 输入 token 数。
        out_tokens: 输出 token 数。
        client_key: 命中 client 的 :attr:`LLMClient.encrypted_api_key`。
    """

    api_params: Optional[dict] = None
    prompt_args: Any = None
    prompt: Optional[str] = None
    prompt_template: Optional[PromptTemplate] = None
    response: Union[str, Generator, Any] = None
    result: Any = None
    reasoning_content: Optional[str] = None
    error: Optional[Exception] = None
    status_code: Optional[str] = None
    timecost: float = 0
    in_tokens: int = 0
    out_tokens: int = 0
    client_key: Optional[str] = None

    def to_dict(self, deep: bool = True) -> dict:
        """把结果序列化为 dict。

        委托给基类 :class:`DictLike.to_dict`（自动递归序列化嵌套的 :class:`DictLike`），
        然后做三处特殊处理：

        - ``prompt_template`` 序列化为其类名（避免 PromptTemplate 内部状态泄漏）。
        - ``response`` 若为 :class:`Generator`，物化为 ``list``。
        - ``error`` 保持原异常对象。

        Args:
            deep: 是否递归序列化嵌套字段。

        Returns:
            序列化后的字典。
        """
        data = super().to_dict(deep=deep)

        if self.prompt_template:
            data["prompt_template"] = self.prompt_template.__class__.__name__

        if isinstance(data.get("response"), Generator):
            data["response"] = list(data["response"])

        return data


class BaseLLMChain(ABC):
    """Chain 抽象基类 —— 提供 client 选择、参数拼接、单次调用与重试原语。

    子类只需实现 :meth:`__call__` 的编排逻辑；重试循环统一由
    :meth:`_retry_invoke` 提供。
    """

    def __init__(
        self,
        client_group: LLMClientGroup,
        prompt_template: PromptTemplate,
        **api_params,
    ):
        """初始化 Chain。

        Args:
            client_group: :class:`LLMClientGroup` 实例。
            prompt_template: :class:`PromptTemplate` 实例。
            **api_params: 所有调用请求的默认 API 参数（如 ``model``、
                ``max_tokens`` 等 OpenAI SDK 支持的字段）。
        """
        self._client_group = client_group
        self._prompt_template = prompt_template
        self._lock = threading.Lock()
        self._default_api_params = api_params

    def _choose_client(self, ignored_clients) -> Optional[LLMClient]:
        """从 :attr:`_client_group` 选一个可用 client。

        Args:
            ignored_clients: 屏蔽列表（本轮重试中已失败的 client）。

        Returns:
            选中的 :class:`LLMClient`；无候选时返回 ``None``。
        """
        with self._lock:
            client = self._client_group.find_available_client(
                ignored_clients=ignored_clients,
            )
        return client

    def obtain_api_params(
        self,
        top_p: Optional[float],
        temperature: Optional[float],
        seed: Optional[int],
        **api_params,
    ) -> dict:
        """合并默认 API 参数与调用时参数，处理 ``top_k`` 等特殊字段。

        OpenAI 不直接支持 ``top_k``，本函数把它转入 ``extra_body``。
        显式传 ``top_p`` / ``temperature`` / ``seed`` 会覆盖默认值。

        Args:
            top_p: 采样参数；``None`` 表示使用 :attr:`_default_api_params` 里的值。
            temperature: 采样参数；语义同上。
            seed: 随机种子；语义同上。
            **api_params: 其他要合并进去的 API 参数。

        Returns:
            最终送出的 API 参数 dict。
        """
        new_api_params = {**self._default_api_params}

        extra_body = {}
        if "extra_body" in api_params:
            extra_body = api_params.pop("extra_body")

        if "top_k" in api_params:
            # OpenAI 不直接支持 top_k；放进 extra_body 让兼容服务读取
            extra_body["top_k"] = api_params["top_k"]

        if temperature is not None:
            new_api_params["temperature"] = temperature
        if top_p is not None:
            new_api_params["top_p"] = top_p
        if seed is not None:
            new_api_params["seed"] = seed
        if len(extra_body) > 0:
            new_api_params["extra_body"] = extra_body
        new_api_params.update(api_params)

        return new_api_params

    def _invoke(
        self,
        prompt: str,
        client: LLMClient,
        history: Optional[List[Message]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[SystemMessage] = None,
        stream: bool = False,
        timeout: float = 600,
        **api_params,
    ) -> dict:
        """单次非重试的 LLM 调用 + 解析。

        成功时返回 ``dict``，含 ``status_code=SUCCEED_CODE``、``result``、``response``
        等字段。本方法不捕获 LLM 错误 —— 由 :meth:`_retry_invoke` 负责。

        Args:
            prompt: 已渲染好的 prompt。
            client: 选定的 :class:`LLMClient`。
            history: 历史消息。
            images: 图像 URL / 路径列表。
            system_prompt: 系统 prompt。
            stream: 是否流式。
            timeout: HTTP 超时（秒）。
            **api_params: 透传给 :meth:`LLMClient.response` 的 API 参数。

        Returns:
            调用结果 dict。
        """
        chain_result = {}
        chain_result["client_key"] = client.encrypted_api_key

        llm_response = client.response(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            images=images,
            stream=stream,
            timeout=timeout,
            **api_params,
        )
        chain_result["response"] = llm_response.response
        chain_result["in_tokens"] = llm_response.in_tokens
        chain_result["out_tokens"] = llm_response.out_tokens
        chain_result["timecost"] = llm_response.timecost

        try:
            reasoning_content = (
                llm_response.details.choices[0].message.reasoning_content
                if hasattr(
                    llm_response.details.choices[0].message,
                    "reasoning_content",
                )
                else None
            )
        except Exception:
            reasoning_content = None
        chain_result["reasoning_content"] = reasoning_content

        parse_result = self._prompt_template.parse(llm_response.response)
        chain_result["result"] = parse_result
        chain_result["status_code"] = SUCCEED_CODE
        chain_result["error"] = None
        return chain_result

    def _retry_invoke(
        self,
        prompt: str,
        *,
        max_retries: int = 1,
        history: Optional[List[Message]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[SystemMessage] = None,
        stream: bool = False,
        timeout: float = 600,
        **api_params,
    ) -> Tuple[dict, Optional[LLMClient]]:
        """围绕 :meth:`_invoke` 的共享重试原语。

        异常处理矩阵：

        - :class:`PromptTemplateParsingError` —— ``continue``（解析瞬时失败，重试）
        - :class:`LLMClientError`（client 非 None）—— 忽略该 client 后 ``continue``
        - :class:`LLMClientError`（client 为 None）—— 无可用 client，``return``
        - :class:`Exception` —— ``continue``

        :class:`PromptTemplateGeneratingError` 不在本循环内捕获 —— 它由调用方在
        调用 :meth:`PromptTemplate.generate_prompt` 时单独处理，因为该异常是确定性的，
        重试无意义。

        Args:
            prompt: 已渲染好的 prompt。
            max_retries: 最大重试次数。
            history: 历史消息。
            images: 图像列表。
            system_prompt: 系统 prompt。
            stream: 是否流式。
            timeout: HTTP 超时（秒）。
            **api_params: 透传给 :meth:`LLMClient.response` 的 API 参数。

        Returns:
            ``(result_dict, client)``：成功时 result_dict 含
            ``status_code=SUCCEED_CODE`` 和 ``result``；失败时 result_dict 含
            ``error`` 字段。client 可能为 ``None``（无可用 client 时）。
        """
        ignored_clients: List[LLMClient] = []
        client: Optional[LLMClient] = None
        result: dict = {}

        for _ in range(max_retries):
            try:
                client = self._choose_client(ignored_clients)
                if client is None:
                    raise LLMClientError("No available LLMClient")
                result = self._invoke(
                    prompt,
                    client=client,
                    system_prompt=system_prompt,
                    history=history,
                    images=images,
                    stream=stream,
                    timeout=timeout,
                    **api_params,
                )
                return result, client  # success — exit immediately
            except LLMClientError as e:
                result["error"] = e
                if client is not None:
                    ignored_clients.append(client)
                    continue
                return result, client  # no available client
            except Exception as e:
                result["error"] = e
                continue

        return result, client


class LLMChain(BaseLLMChain):
    """单轮 LLM chain —— 调用一次 LLM 即返回。

    支持简单重试（``max_retries``），失败时把最后一次异常写入
    :attr:`ChainResult.error`。
    """

    def __call__(
        self,
        *prompt_tmpl_args,
        history: Optional[List[Message]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[SystemMessage] = None,
        stream: bool = False,
        max_retries: int = 1,
        timeout: float = 600,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> ChainResult:
        """调用 LLM。

        Args:
            *prompt_tmpl_args: 传给 :meth:`PromptTemplate.generate_prompt` 的参数。
            history: 对话历史。
            images: 图像 URL / 路径列表。
            system_prompt: 系统 prompt。
            stream: 是否流式。
            max_retries: 最大重试次数。
            timeout: HTTP 超时（秒）。
            top_p: 采样参数。
            temperature: 采样参数。
            seed: 随机种子。
            **api_params: 其他 API 参数。

        Returns:
            :class:`ChainResult`。
        """
        api_params = self.obtain_api_params(
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            **api_params,
        )

        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
            api_params=api_params,
        )

        with Timer() as t:
            try:
                prompt = self._prompt_template.generate_prompt(*prompt_tmpl_args)
            except PromptTemplateGeneratingError as e:
                chain_result.error = e
            else:
                llm_result, _ = self._retry_invoke(
                    prompt,
                    max_retries=max_retries,
                    history=history,
                    images=images,
                    system_prompt=system_prompt,
                    stream=stream,
                    timeout=timeout,
                    **api_params,
                )
                chain_result.update(llm_result)

            chain_result.timecost = t.elapsed
        return chain_result


class LLMChainWithValid(LLMChain):
    """多轮校验 LLM chain —— LLM 返回非法结果时，把错误作为新 prompt 反馈给模型重试。

    Args:
        max_chat_turns: 校验重试轮数上限（非法结果触发重轮的次数）。
    """

    def __call__(
        self,
        *prompt_tmpl_args,
        history: Optional[List[Message]] = None,
        images: Optional[List[str]] = None,
        system_prompt: Optional[SystemMessage] = None,
        stream: bool = False,
        max_chat_turns: int = 1,
        max_retries: int = 1,
        timeout: float = 600,
        # generate params
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **api_params,
    ) -> ChainResult:
        """调用 LLM，结果不合法时把错误反馈给模型让它自纠。

        Args:
            *prompt_tmpl_args: 传给 :meth:`PromptTemplate.generate_prompt` 的参数。
            history: 对话历史。
            images: 图像 URL / 路径列表。
            system_prompt: 系统 prompt。
            stream: 是否流式。
            max_chat_turns: 校验重试轮数上限。
            max_retries: 单轮内的最大重试次数。
            timeout: HTTP 超时（秒）。
            top_p: 采样参数。
            temperature: 采样参数。
            seed: 随机种子。
            **api_params: 其他 API 参数。

        Returns:
            :class:`ChainResult`。
        """
        api_params = self.obtain_api_params(
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            **api_params,
        )

        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
            api_params=api_params,
        )

        history = history or []
        if system_prompt is not None:
            history.append(system_prompt)

        prompt = ""

        with Timer() as t:
            for turn_idx in range(max_chat_turns):
                try:
                    if turn_idx == 0:
                        prompt = self._prompt_template.generate_prompt(
                            *prompt_tmpl_args
                        )

                    # system_prompt 已 append 到 history，这里不再传
                    llm_result, client = self._retry_invoke(
                        prompt,
                        max_retries=max_retries,
                        history=history,
                        images=images,
                        stream=stream,
                        timeout=timeout,
                        **api_params,
                    )
                    chain_result.update(llm_result)
                    if client is None:
                        raise LLMClientError("No available LLMClient")

                    self._prompt_template.valid(
                        chain_result.result, *prompt_tmpl_args, **api_params
                    )
                    break
                except (PromptTemplateGeneratingError, LLMClientError) as e:
                    chain_result.error = e
                    break
                except PromptTemplateValidError as e:
                    history.append(HumanMessage(prompt))
                    history.append(AssistantMessage(chain_result.response))
                    prompt = str(e)
                except Exception as e:
                    chain_result.error = e

            chain_result.timecost = t.elapsed
        return chain_result
