lwj\_tools.concurrent package
=============================

并发执行器集合。

``async_runner`` 基于 :mod:`asyncio` + :mod:`httpx`，适合 I/O 密集型任务（HTTP 请求
批量发送），复用单个 :class:`httpx.AsyncClient` 以利用连接池；``sync`` 基于
:mod:`concurrent.futures`，通过 :class:`MultiProcessRunner` /
:class:`MultiThreadingRunner` 切换进程池或线程池。

Usage example
-------------

.. code-block:: python

   import httpx
   from lwj_tools.concurrent.async_runner import AsyncRunner

   class FetchRunner(AsyncRunner):
       async def worker_func(self, idx, sample, client):
           resp = await client.get(sample["url"], timeout=10)
           resp.raise_for_status()
           return resp.text

   runner = FetchRunner(max_concurrency=8, timeout=10)
   results = runner([{"url": "https://example.com"}, ...])

Submodules
----------

lwj\_tools.concurrent.async\_runner module
------------------------------------------

.. automodule:: lwj_tools.concurrent.async_runner
   :members:
   :show-inheritance:
   :undoc-members:

lwj\_tools.concurrent.sync module
---------------------------------

.. automodule:: lwj_tools.concurrent.sync
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: lwj_tools.concurrent
   :members:
   :show-inheritance:
   :undoc-members:
