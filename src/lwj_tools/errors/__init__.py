#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一错误码与业务异常基类。

所有异常继承 :class:`BaseError`，可通过 ``code`` 字段做日志聚合与运行时分支判断。

错误码段划分：

- ``BASE_ERROR_CODE = E100`` —— 基类默认码（一般不直接抛出）
- ``E201 - E204`` —— LLM 相关（客户端、prompt 模板的生成/解析/校验）
- ``E301 - E303`` —— 文件 I/O（类型、读、写）
- ``E401`` —— 并发执行

``SUCCEED_CODE = "success"`` 用于“非异常但有状态”的返回值表示成功（如
:class:`lwj_tools.llms.prompt.PromptTemplate.valid` 的返回值）。
"""
SUCCEED_CODE = "success"
"""成功标记字符串（不是异常码，用在带状态返回的 API 中表示成功）。"""

BASE_ERROR_CODE = "E100"
"""基类默认错误码。"""


class BaseError(Exception):
    """所有自定义异常的基类。

    Args:
        message: 错误信息。
        code: 错误码，格式 ``E<三段数字>``。
    """

    def __init__(
        self,
        message: str = "",
        code: str = BASE_ERROR_CODE,
    ):
        super().__init__(message)
        self.message = message
        self.code = code

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message}, code={self.code})"

    def __str__(self):
        return self.__repr__()


class LLMClientError(BaseError):
    """LLM 调用失败（网络、鉴权、模型返回异常等），错误码 ``E201``。"""

    def __init__(self, message: str = ""):
        code = "E201"
        super().__init__(message, code)


class PromptTemplateGeneratingError(BaseError):
    """:class:`lwj_tools.llms.prompt.PromptTemplate.generate_prompt` 失败，错误码 ``E202``。"""

    def __init__(self, message: str = ""):
        code = "E202"
        super().__init__(message, code)


class PromptTemplateParsingError(BaseError):
    """:class:`lwj_tools.llms.prompt.PromptTemplate.parse` 失败，错误码 ``E203``。"""

    def __init__(self, message: str = ""):
        code = "E203"
        super().__init__(message, code)


class PromptTemplateValidError(BaseError):
    """:class:`lwj_tools.llms.prompt.PromptTemplate.valid` 返回 ``False`` 时抛出，错误码 ``E204``。"""

    def __init__(self, message: str = ""):
        code = "E204"
        super().__init__(message, code)


class FileTypeError(BaseError):
    """文件类型与期望不符，错误码 ``E301``。"""

    def __init__(self, message: str = ""):
        code = "E301"
        super().__init__(message, code)


class FileReadError(BaseError):
    """文件读取失败，错误码 ``E302``。"""

    def __init__(self, message: str = ""):
        code = "E302"
        super().__init__(message, code)


class FileWriteError(BaseError):
    """文件写入失败，错误码 ``E303``。"""

    def __init__(self, message: str = ""):
        code = "E303"
        super().__init__(message, code)


class ConcurrentError(BaseError):
    """并发任务执行失败（任一 worker 抛异常），错误码 ``E401``。"""

    def __init__(self, message: str = ""):
        code = "E401"
        super().__init__(message, code)
