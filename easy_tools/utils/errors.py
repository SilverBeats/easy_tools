from typing import Any


class BaseError(Exception):
    def __init__(self, message: str, data: Any = None):
        self.message = message
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message},data={self.data})"

    def __str__(self):
        return self.__repr__()


class FileTypeError(BaseError):
    pass


class FileReadError(BaseError):
    pass


class FileWriteError(BaseError):
    pass
