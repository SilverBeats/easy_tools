#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Any, Callable, Optional

from ..errors import PromptTemplateGeneratingError, PromptTemplateParsingError


class PromptTemplate:
    def __init__(
        self,
        name: Optional[str] = None,
        generate_fn: Optional[Callable] = None,
        parse_fn: Optional[Callable] = None,
    ):
        """Manage the prompt generating and result parsing"""
        self._name = name
        self._generate_fn = generate_fn or self.generate_fn
        self._parse_fn = parse_fn or self.parse_fn

    def generate_prompt(self, *args, **kwargs):
        if self._generate_fn is None:
            raise NotImplementedError("Please implement the generate_fn")

        try:
            return self._generate_fn(*args, **kwargs)
        except Exception as e:
            raise PromptTemplateGeneratingError(str(e))

    def parse(self, result: Any, *args, **kwargs):
        if self._parse_fn is None:
            raise NotImplementedError("Please implement the parse_fn")

        try:
            return self._parse_fn(result, *args, **kwargs)
        except Exception as e:
            raise PromptTemplateParsingError(str(e))

    @property
    def name(self):
        return self._name or __name__

    @staticmethod
    def generate_fn(data, *args, **kwargs):
        return data

    @staticmethod
    def parse_fn(data, *args, **kwargs):
        return data
