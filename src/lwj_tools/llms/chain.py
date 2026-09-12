"""LLM 调用编排层。

把 :class:`lwj_tools.llms.client.LLMClient` / :class:`lwj_tools.llms.prompt.PromptTemplate`
组合成"参数 → prompt → LLM 调用 → 解析 → 结果"的工作流。

契约：
    除 ``obtain_api_params`` 外，:meth:`BaseLLMChain.__call__` 及其子类
    的 ``__call__`` 不会向外抛异常。调用是否成功一律通过返回的
    :class:`ChainResult` 的 :attr:`ChainResult.error` 与
    :attr:`ChainResult.status_code` 判断：

    - 成功：``status_code == SUCCEED_CODE`` 且 ``error is None``
    - 失败：``status_code == FAILED_CODE`` 且 ``error`` 为具体异常

Example:
    >>> from lwj_tools.llms.chain import LLMChain
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
    >>> if result.status_code == SUCCEED_CODE:
    ...     print(result.result)
"""

from __future__ import annotations

import threading
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Union

from ..common.pojo import DictLike
from ..date.timer import Timer
from ..errors import (
    LLMClientError,
    PromptTemplateParsingError,
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
        timecost: 端到端耗时（秒），使用墙钟时间。
        in_tokens: 输入 token 数（跨轮、跨重试累计）。
        out_tokens: 输出 token 数（跨轮、跨重试累计）。
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
    timecost: float = 0
    in_tokens: int = 0
    out_tokens: int = 0
    client_key: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """便捷判断：是否成功。"""
        return  self.error is None

    def to_dict(self, deep: bool = True, *, detect_cycles: bool = True) -> dict:
        """把结果序列化为 dict。

        在基类递归序列化之上做三处特化：

        - ``prompt_template`` 序列化为其类名（避免内部状态泄漏）。
        - ``response`` 若为 :class:`Generator`，物化为 ``list``。
        - ``error`` 保持原异常对象。
        """
        data = super().to_dict(deep=deep, detect_cycles=detect_cycles)

        if self.prompt_template is not None:
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
            **api_params: 所有调用请求的默认 API 参数。
        """
        self._client_group = client_group
        self._prompt_template = prompt_template
        self._lock = threading.Lock()
        self._default_api_params = api_params

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _choose_client(self, ignored_clients: List[LLMClient]) -> Optional[LLMClient]:
        """从 :attr:`_client_group` 选一个可用 client。"""
        with self._lock:
            return self._client_group.find_available_client(
                ignored_clients=ignored_clients,
            )

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
        """
        new_api_params = {**self._default_api_params}

        # 复制一份，避免原地污染调用方传入的 extra_body
        extra_body = dict(api_params.pop("extra_body", {}))
        if "top_k" in api_params:
            extra_body["top_k"] = api_params["top_k"]

        if temperature is not None:
            new_api_params["temperature"] = temperature
        if top_p is not None:
            new_api_params["top_p"] = top_p
        if seed is not None:
            new_api_params["seed"] = seed
        if extra_body:
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

        成功时返回含 ``status_code=SUCCEED_CODE``、``result``、``response``
        等字段的 dict。本方法不捕获错误 —— 由 :meth:`_retry_invoke` 负责。
        """
        llm_response = client.response(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            images=images,
            stream=stream,
            timeout=timeout,
            **api_params,
        )

        reasoning_content: Optional[str] = None
        try:
            message = llm_response.details.choices[0].message
            reasoning_content = getattr(message, "reasoning_content", None)
        except Exception:
            pass

        parse_result = self._prompt_template.parse(llm_response.response)
        return {
            "client_key": client.encrypted_api_key,
            "reasoning_content": reasoning_content,
            "response": llm_response.response,
            "in_tokens": llm_response.in_tokens,
            "out_tokens": llm_response.out_tokens,
            "timecost": llm_response.timecost,
            "result": parse_result
        }

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

        - :class:`PromptTemplateParsingError` —— 解析瞬时失败，重试。
        - :class:`LLMClientError`（client 非 None）—— 屏蔽该 client 后重试。
        - :class:`LLMClientError`（client 为 None）—— 无可用 client，退出。
        - 其他 :class:`Exception` —— 确定性失败，记录后退出。

        :class:`PromptTemplateGeneratingError` 不在本循环内捕获 —— 由调用方
        在 :meth:`PromptTemplate.generate_prompt` 处单独处理。

        Returns:
            `(result_dict, client)`. 成功时, `result_dict` 中不会有 `error` 字段
        """
        ignored_clients: List[LLMClient] = []
        client: Optional[LLMClient] = None
        result: dict = {}

        total_in = 0
        total_out = 0
        total_time = 0.0

        for _ in range(max(1, max_retries)):
            try:
                client = self._choose_client(ignored_clients)
                if client is None:
                    raise LLMClientError("No available LLMClient")

                attempt = self._invoke(
                    prompt,
                    client=client,
                    system_prompt=system_prompt,
                    history=history,
                    images=images,
                    stream=stream,
                    timeout=timeout,
                    **api_params,
                )
            except PromptTemplateParsingError as exc:
                # 解析瞬时失败：记录后换一次重试机会
                result["error"] = exc
                continue
            except LLMClientError as exc:
                result["error"] = exc
                if client is None:
                    break
                ignored_clients.append(client)
                continue
            except Exception as exc:
                # 确定性异常，重试无意义
                result["error"] = exc
                break

            total_in += attempt.get("in_tokens", 0)
            total_out += attempt.get("out_tokens", 0)
            total_time += attempt.get("timecost", 0.0)

            result = {
                **attempt,
                "in_tokens": total_in,
                "out_tokens": total_out,
                "timecost": total_time,
            }
            break

        return result, client


class LLMChain(BaseLLMChain):
    """单轮 LLM chain —— 调用一次 LLM 即返回。

    支持简单重试（``max_retries``）。契约见模块文档：``__call__`` 不抛异常，
    成败由 :attr:`ChainResult.error` 判断。
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
            max_retries: 最大尝试次数（``1`` 表示只调用一次）。
            timeout: HTTP 超时（秒）。
            top_p / temperature / seed: 采样参数。
            **api_params: 其他 API 参数。

        Returns:
            :class:`ChainResult`
        """
        
        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
        )

        with Timer() as t:
            try:
                api_params = self.obtain_api_params(
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                    **api_params,
                )
                chain_result.api_params = api_params

                prompt = self._prompt_template.generate_prompt(*prompt_tmpl_args)
                chain_result.prompt = prompt

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
            except Exception as exc:
                chain_result.error = exc

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
            history: 对话历史（不会被就地修改）。
            images: 图像 URL / 路径列表。
            system_prompt: 系统 prompt。
            stream: 是否流式。
            max_chat_turns: 校验重试轮数上限。
            max_retries: 单轮内的最大尝试次数。
            timeout: HTTP 超时（秒）。
            top_p / temperature / seed: 采样参数。
            **api_params: 其他 API 参数。

        Returns:
            :class:`ChainResult`。
        """
        chain_result = ChainResult(
            prompt_template=self._prompt_template,
            prompt_args=prompt_tmpl_args,
        )
        try:
            api_params = self.obtain_api_params(
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                **api_params,
            )
            chain_result.api_params = api_params
        except Exception as exc:
            chain_result.error = exc
            return chain_result
        

        # 拷贝一份，避免污染调用方的 history
        history = list(history) if history else []
        if system_prompt is not None:
            history.append(system_prompt)

        max_chat_turns = max(1, max_chat_turns)
        prompt = ""
        total_in = 0
        total_out = 0

        with Timer() as t:
            for turn_idx in range(max_chat_turns):
                try:
                    if turn_idx == 0:
                        prompt = self._prompt_template.generate_prompt(*prompt_tmpl_args)
                        chain_result.prompt = prompt

                    llm_result, _ = self._retry_invoke(
                        prompt,
                        max_retries=max_retries,
                        history=history,
                        images=images,
                        stream=stream,
                        timeout=timeout,
                        **api_params,
                    )
                    total_in += llm_result.get("in_tokens", 0)
                    total_out += llm_result.get("out_tokens", 0)
                    chain_result.update(llm_result)

                    if llm_result.get("error", None) is not None:
                        break

                    self._prompt_template.valid(
                        chain_result.result, *prompt_tmpl_args, **api_params
                    )
                    break
                except PromptTemplateValidError as exc:
                    if turn_idx == max_chat_turns - 1: # 最后一轮扔失败
                        chain_result.error = exc
                        break
                    history.append(HumanMessage(prompt))
                    history.append(AssistantMessage(chain_result.response))
                    prompt = f"**Valid Error Message as Follow:**\n{exc.message}"
                except Exception as exc:
                    # 包含 PromptTemplateGeneratingError 及其他确定性异常
                    chain_result.error = exc
                    break

            chain_result.timecost = t.elapsed
            chain_result.in_tokens = total_in
            chain_result.out_tokens = total_out

        return chain_result
