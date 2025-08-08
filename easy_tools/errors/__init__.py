#!/usr/bin/env python3
# -*- coding: utf-8 -*-
SUCCEED_CODE = "success"
BASE_ERROR_CODE = "E100"


class BaseError(Exception):
    def __init__(
        self,
        message: str = "",
        code: str = BASE_ERROR_CODE,
    ):
        self.message = message
        self.code = code

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message},code={self.code})"

    def __str__(self):
        return self.__repr__()


class LLMClientError(BaseError):
    def __init__(self, message: str = ""):
        code = "E201"
        super().__init__(message, code)


class PromptTemplateGeneratingError(BaseError):

    def __init__(self, message: str = ""):
        code = "E202"
        super().__init__(message, code)


class PromptTemplateParsingError(BaseError):

    def __init__(self, message: str = ""):
        code = "E203"
        super().__init__(message, code)


class FileTypeError(BaseError):

    def __init__(self, message: str = ""):
        code = "E301"
        super().__init__(message, code)


class FileReadError(BaseError):

    def __init__(self, message: str = ""):
        code = "E302"
        super().__init__(message, code)


class FileWriteError(BaseError):

    def __init__(self, message: str = ""):
        code = "E303"
        super().__init__(message, code)


class ConcurrentError(BaseError):

    def __init__(self, message: str = ""):
        code = "E401"
        super().__init__(message, code)
