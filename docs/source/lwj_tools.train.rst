lwj\_tools.train package
========================

通用训练循环：抽象 :class:`Trainer` + 自由函数化的 step/eval/checkpoint 工具。

``checkpoint`` 负责保存/读取 best checkpoint；``loops`` 提供 model_forward、
optim_lr_scheduler 等可复用的钩子；``optimizer`` 提供 :class:`TrainingArguments`
和 :func:`build_optimizer` / :func:`build_scheduler`；``trainer`` 提供
:class:`Trainer` 抽象基类，把 loops 的自由函数串成完整 train/eval 流程。

Submodules
----------

lwj\_tools.train.checkpoint module
----------------------------------

.. automodule:: lwj_tools.train.checkpoint
   :members:
   :show-inheritance:
   :undoc-members:

lwj\_tools.train.loops module
-----------------------------

.. automodule:: lwj_tools.train.loops
   :members:
   :show-inheritance:
   :undoc-members:

lwj\_tools.train.optimizer module
---------------------------------

.. automodule:: lwj_tools.train.optimizer
   :members:
   :show-inheritance:
   :undoc-members:

lwj\_tools.train.trainer module
-------------------------------

.. automodule:: lwj_tools.train.trainer
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: lwj_tools.train
   :members:
   :show-inheritance:
   :undoc-members:
