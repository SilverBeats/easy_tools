"""``lwj_tools`` 顶层包。

本包是个人工具集，按职责拆分为多个子包：

- :mod:`lwj_tools.common` —— 通用工具（文件、ID、字符串、随机数、日志等）
- :mod:`lwj_tools.concurrent` —— 并发执行器（同步多进程/多线程、异步 httpx 池）
- :mod:`lwj_tools.date` —— 时间相关小工具
- :mod:`lwj_tools.errors` —— 业务异常基类与常用错误码
- :mod:`lwj_tools.evaluators` —— NLG 评估指标（GLEU、BLEU、BARTScore 等）
- :mod:`lwj_tools.io` —— 文件读写辅助
- :mod:`lwj_tools.llms` —— LLM 调用编排（message / prompt / client / chain）
- :mod:`lwj_tools.nn` —— PyTorch 辅助
- :mod:`lwj_tools.plot` —— 简单可视化（基于 matplotlib）
- :mod:`lwj_tools.train` —— 训练器与训练循环

部分子包依赖额外可选包（如 torch、openai），安装方式见 ``pyproject.toml`` 的
``optional-dependencies``（如 ``pip install "lwj_tools[all]"``）。
"""
