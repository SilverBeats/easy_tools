"""Prompt 模板与 LLM 响应解析。

:class:`PromptTemplate` 把"参数 → prompt 字符串"和"LLM 响应 → 结构化结果"
两步封装为可复用的模板对象，并支持对解析结果做合法性校验：

- :meth:`PromptTemplate.generate_prompt` —— 参数 → prompt 字符串
- :meth:`PromptTemplate.parse` —— LLM 响应 → 结构化结果
- :meth:`PromptTemplate.valid` —— 结构化结果 → 校验（仅 ``response_format=json_object``
  / ``response_format=json_schema`` 模式有意义）

三个钩子（:meth:`generate_fn` / :meth:`parse_fn` / :meth:`valid_fn`）必须由子类
覆写；基类默认抛 :class:`NotImplementedError`，杜绝"忘覆写导致把 ``args`` tuple
当作 prompt 送给 LLM"的静默错误。

Example:
    >>> from lwj_tools.llms.prompt import PromptTemplate
    >>> class SumPrompt(PromptTemplate):
    ...     def generate_fn(self, num1: int, num2: int) -> str:
    ...         return f"{num1} + {num2} 等于多少？"
    ...     def parse_fn(self, llm_response: str) -> int:
    ...         # 实际项目中用正则或 json.loads
    ...         return int("".join(ch for ch in llm_response if ch.isdigit()))
    >>> pt = SumPrompt()
    >>> pt.generate_prompt(1, 2)
    '1 + 2 等于多少？'
    >>> pt.parse("等于 3")
    3
"""
from typing import Any, Callable, Optional

from ..errors import (
    PromptTemplateGeneratingError,
    PromptTemplateParsingError,
    PromptTemplateValidError,
)


class PromptTemplate:
    """Prompt 模板基类。

    必须覆写三个钩子之一（或多个）：

    - :meth:`generate_fn` —— 把参数拼成 prompt 字符串
    - :meth:`parse_fn` —— 把 LLM 响应解析成结构化结果
    - :meth:`valid_fn` —— 校验 parse 结果

    也可以在 :meth:`__init__` 里直接把钩子函数作为参数传入，等价于子类覆写。

    Example:
        >>> class MyPromptTemplate(PromptTemplate):
        ...     prompt = "{NUM_1} + {NUM_2} 等于多少？"
        ...     def generate_fn(self, num1: int, num2: int) -> str:
        ...         return self.prompt.format(NUM_1=num1, NUM_2=num2)
        ...     def parse_fn(self, llm_response: str) -> Any:
        ...         return llm_response
    """

    def __init__(
        self,
        name: Optional[str] = None,
        generate_fn: Optional[Callable] = None,
        parse_fn: Optional[Callable] = None,
        valid_fn: Optional[Callable] = None,
    ):
        """初始化模板。

        Args:
            name: 模板名（用于日志 / 调试），省略时回退到类名。
            generate_fn: 覆写 :meth:`generate_fn`；省略则用基类实现。
            parse_fn: 覆写 :meth:`parse_fn`；省略则用基类实现。
            valid_fn: 覆写 :meth:`valid_fn`；省略则用基类实现。
        """
        self._name = name
        # `or` 短路：显式传 None 也走默认实现（raise NIE）
        self._generate_fn = generate_fn or self.generate_fn
        self._parse_fn = parse_fn or self.parse_fn
        self._valid_fn = valid_fn or self.valid_fn

    def generate_prompt(self, *args, **kwargs) -> str:
        """调用 :meth:`generate_fn` 生成 prompt 字符串。

        Args:
            *args: 透传给 :meth:`generate_fn` 的位置参数。
            **kwargs: 透传给 :meth:`generate_fn` 的关键字参数。

        Returns:
            渲染后的 prompt 字符串。

        Raises:
            PromptTemplateGeneratingError: :meth:`generate_fn` 抛错（包装为业务异常）。
            NotImplementedError: :meth:`generate_fn` 未覆写时直接抛出。
        """
        try:
            return self._generate_fn(*args, **kwargs)
        except (PromptTemplateGeneratingError, NotImplementedError):
            raise
        except Exception as e:
            raise PromptTemplateGeneratingError(str(e)) from e

    def parse(self, result: Any, *args, **kwargs) -> Any:
        """调用 :meth:`parse_fn` 把 LLM 响应解析为结构化结果。

        Args:
            result: LLM 原始响应文本。
            *args: 透传给 :meth:`parse_fn`。
            **kwargs: 透传给 :meth:`parse_fn`。

        Returns:
            解析后的结构化结果。

        Raises:
            PromptTemplateParsingError: :meth:`parse_fn` 抛错（包装为业务异常）。
            NotImplementedError: :meth:`parse_fn` 未覆写时直接抛出。
        """
        try:
            return self._parse_fn(result, *args, **kwargs)
        except (PromptTemplateParsingError, NotImplementedError):
            raise
        except Exception as e:
            raise PromptTemplateParsingError(str(e)) from e

    def valid(self, result: dict, *args, **kwargs) -> None:
        """调用 :meth:`valid_fn` 校验 :meth:`parse` 的结果。

        仅在 ``response_format=json_object`` / ``response_format=json_schema``
        模式下有意义。

        Args:
            result: :meth:`parse` 的返回值，必须是 ``dict``。
            *args: 透传给 :meth:`valid_fn`。
            **kwargs: 透传给 :meth:`valid_fn`。

        Raises:
            PromptTemplateValidError: ``result`` 不是 ``dict``，或
                :meth:`valid_fn` 抛错（包装为业务异常）。
            NotImplementedError: :meth:`valid_fn` 未覆写时直接抛出。
        """
        if not isinstance(result, dict):
            raise PromptTemplateValidError(
                f"valid() expects dict, got {type(result).__name__}"
            )
        try:
            self._valid_fn(result, *args, **kwargs)
        except (PromptTemplateValidError, NotImplementedError):
            raise
        except Exception as e:
            raise PromptTemplateValidError(str(e)) from e

    @property
    def name(self) -> str:
        """返回 :attr:`_name`，未设置时回退到类名。"""
        return self._name if self._name is not None else type(self).__name__

    def generate_fn(self, *args, **kwargs) -> str:
        """把参数拼成 prompt 字符串 —— 必须由子类覆写。"""
        raise NotImplementedError(
            f"{type(self).__name__}.generate_fn must be overridden"
        )

    def parse_fn(self, llm_response: Any, *args, **kwargs) -> Any:
        """把 LLM 响应解析成结构化结果 —— 必须由子类覆写。"""
        raise NotImplementedError(f"{type(self).__name__}.parse_fn must be overridden")

    def valid_fn(self, result: dict, *args, **kwargs) -> None:
        """校验 parse 结果 —— 必须由子类覆写。"""
        raise NotImplementedError(f"{type(self).__name__}.valid_fn must be overridden")
