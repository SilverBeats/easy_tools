"""对话消息与多模态内容构造。

封装 OpenAI chat completions API 的消息结构：

- :class:`Message` —— 基类（``role`` + ``content``），继承 :class:`DictLike`
  可直接序列化为 wire-format dict。
- :class:`HumanMessage` / :class:`SystemMessage` / :class:`AssistantMessage` ——
  role 固定的三种消息类型；``content`` 既接受纯文本 ``str``，也接受多模态
  ``list``（与 OpenAI 对各 role 的实际允许范围一致）。
- :class:`ImageContent` / :class:`TextContent` —— 多模态消息中的图像 / 文本
  片段，组装成 ``list`` 后作为 ``HumanMessage.content`` 传入即可。

Example:
    >>> from lwj_tools.llms.message import (
    ...     HumanMessage, SystemMessage, AssistantMessage,
    ...     ImageContent, TextContent,
    ... )
    >>> msgs = [
    ...     SystemMessage("You are a helpful assistant."),
    ...     HumanMessage([
    ...         TextContent("What's in this image?"),
    ...         ImageContent("https://example.com/cat.jpg"),
    ...     ]),
    ... ]
    >>> [m.to_dict(True) for m in msgs]
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user',
      'content': [{'type': 'text', 'text': "What's in this image?"},
                  {'type': 'image_url', 'image_url': {'url': 'https://example.com/cat.jpg'}}]}]
"""
from dataclasses import dataclass
from typing import Union

from ..common.pojo import DictLike


@dataclass
class Message(DictLike):
    """OpenAI 风格对话消息。

    Attributes:
        role: 消息角色，取值 ``"system"`` / ``"user"`` / ``"assistant"``。
        content: 消息内容；纯文本为 ``str``，多模态为 ``list``（元素是
            :class:`TextContent` 或 :class:`ImageContent`）。
    """

    role: str
    content: Union[str, list]


class HumanMessage(Message):
    """``role="user"`` 的对话消息。"""

    def __init__(self, content: Union[str, list]) -> None:
        super().__init__(role="user", content=content)


class SystemMessage(Message):
    """``role="system"`` 的对话消息。"""

    def __init__(self, content: Union[str, list]) -> None:
        super().__init__(role="system", content=content)


class AssistantMessage(Message):
    """``role="assistant"`` 的对话消息。"""

    def __init__(self, content: Union[str, list]) -> None:
        super().__init__(role="assistant", content=content)


class ImageContent(DictLike):
    """多模态消息中的图像片段。

    序列化为 ``{"type": "image_url", "image_url": {"url": <image_url>}}`` 的
    OpenAI wire-format dict。注意 :attr:`image_url` 本身是 ``dict``（不是
    字符串），便于 :meth:`DictLike.to_dict` 直接产出合规格式。

    Args:
        image_url: 图像 URL 或 ``data:image/...;base64,...`` 形式的内联数据。
        type: 固定为 ``"image_url"``，调用方一般无需修改。
    """

    def __init__(self, image_url: str, type: str = "image_url") -> None:
        self.type = type
        self.image_url = {"url": image_url}


class TextContent(DictLike):
    """多模态消息中的文本片段。

    序列化为 ``{"type": "text", "text": <text>}`` 的 OpenAI wire-format dict。

    Args:
        text: 文本内容。
        type: 固定为 ``"text"``，调用方一般无需修改。
    """

    def __init__(self, text: str, type: str = "text") -> None:
        self.type = type
        self.text = text
