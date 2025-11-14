存放一些我平常使用过的代码，封装好的。会存在不足和 Bug，欢迎留言，我视心情修复或完善。

# Install

```shell
pip install .
```

# 使用说明

## 1. 文件读写方面

目前支持的文件格式为：`json`, `jsonl`, `txt`, `yaml`, `yml`, `pkl`, `xlsx`, `xls`, `csv`, `tsv`, `npy`, `npz`

平常使用诸如 json，pandas，numpy 等工具包读/写文件进行封装，程序会根据文件路径的后缀自动选择对应的工具包进行读/写。

### 1.1 FileReader

**使用示例**

```python
import pandas as pd
from typing import List
from easy_tools.utils.io import FileReader

# 读 json 文件
file_path = 'a.json'
data: dict = FileReader.read(file_path)

# 读 csv 文件
# 1️⃣ 普通读
file_path = 'a.csv'
data: pd.DataFrame = FileReader.read(file_path, return_dict=True)
# 2️⃣ 返回字典
data: List[dict] = FileReader.read(file_path, return_dict=True)
# 3️⃣ 迭代（比如 csv 文件很大时）
for item in FileReader.read(file_path, return_iter=True):
    # 打印格式示例: 1,2
    print(item)

# 4️⃣ 字典 + 迭代读
for item in FileReader.read(
        file_path,
        has_header=True,
        return_dict=True,
        return_iter=True
):
    # 打印格式示例: {'age': 1, 'weight': 2}
    print(item)
```

### 1.2 FileWriter

**使用示例**

```python
import pandas as pd
from easy_tools.utils.io import FileWriter

data = {
    'a': 1,
    'b': 2
}

# 1. 保存 dataframe
file_path = 'test.csv'
FileWriter.dump(pd.DataFrame(data), file_path)

# 2. 保存为 json
file_path = 'test.json'
FileWriter.dump(data, file_path)
```

*⚠️注意：* 程序没有检查你传过来的 data 的数据类型，所以在保存时务必保证 data 数据类型的正确。

```python
data = {
    'a': 1,
    'b': 2
}
file_path = 'test.csv'
# 报错！data 是 dict，没有 to_csv 方法
FileWriter.dump(data, file_path)
```

## 2 时间工具

功能是计算程序运行时间

**使用示例**

```python
import time
from easy_tools.utils.timer import Timer, timecost


@timecost
def func1():
    time.sleep(2)
    return 2 * 3


def func2():
    time.sleep(3)
    return None


def main():
    result = func1()
    print("func1 result: ", result.result)
    print("func1 timecost: ", result.timecost)

    for unit in ["s", "ms", "ns"]:
        with Timer(unit=unit) as timer:
            func2()
        print("func2 timecost: ", timer.elapsed, unit)


if __name__ == "__main__":
    main()
```

## 3. 多线程/进程工具

提高效率的。

参数都比较简单和直观，主要解释两个：`worker_func` 、`callback_func`和`finish_func`

以爬数据的场景为例，可能经历的过程有：发送请求，处理数据、保存数据。

你可以写一个方法，里面包含了这三部。考虑到有些同学可能希望方法职责清晰一些，所以`worker_func`负责请求数据，`callback_func`
负责处理数据，`finish_func`负责保存数据。

**使用示例**

```python
import time
from easy_tools.utils.timer import Timer
from easy_tools.utils.concurrent import MultiProcessRunner, MultiThreadingRunner


def send_post(idx: int):
    """比如你要爬数据"""
    time.sleep(2)
    return {"status_code": 200, "result": {"a": 1, "b": 2}}


def file_data_process(file_path: str = "xxx", output_file_path: str = "xxx"):
    """比如你要处理多个文件"""
    # data read from file_path
    data = []
    # do something
    time.sleep(2)
    # write the new data to output_file_path or not just


def main():
    file_paths = ["xxx"] * 100
    process_runner = MultiProcessRunner(-1)
    with Timer() as t:
        process_runner(
            samples=file_paths,
            worker_func=file_data_process
        )
    print("Multi Process Timecost: ", t.elapsed)

    # api_url/userId=?
    user_ids = list(range(100))
    thread_runner = MultiThreadingRunner(-1)
    with Timer() as t:
        thread_runner(
            samples=user_ids,
            worker_func=send_post
        )
    print("Multi Thread Timecost: ", t.elapsed)


if __name__ == "__main__":
    main()

```

## 4. 杂七杂八工具

在某些场景下，可能会用到一些小工具。自行看代码吧，位置在`easy_tools/utils/tools.py`

## 5. LLM API 请求工具

不同厂家的 LLM 都有自己的 package 包进行 API 请求，但好在大多都支持 Openai 风格请求，所以这里封装了一个通用的请求工具，屏蔽底层不同厂的差异。

厂家提供的接口服务大多存在限速，然而我有时候想要在短时间内进行大量请求，故而会申请多个 API_KEY 以谋求临时解决，这样我就设置更大
worker 数量。

**使用示例**

```python
from easy_tools.llms.prompt import PromptTemplate
from easy_tools.utils.concurrent import MultiThreadingRunner
import json
from json_repair import repair_json
from easy_tools.llms.chain import LLMChain
from easy_tools.llms.client import (
    LLMClient,
    LLMClientGroup,
    LLMResponse,
    APIConfig
)
from functools import partial


# 1. 创建提示模板类
class CustomPromptTemplate(PromptTemplate):
    PROMPT = "Answer the question directly in json format: {NUM1} + {NUM2} = ?"

    # 重写生成提示词的方法
    def generate_fn(self, sample: tuple):
        # 须知, 此处的sample 是一个元组，因为我在 worker_func 中写的是 model(sample)
        # 如果我在 worker_func 中写的是 model(*sample)
        # 那么此处我应该写的是 def generate_fn(self, num1, num2):
        num1, num2 = sample
        return self.PROMPT.format(NUM1=num1, NUM2=num2).strip()

    # 重写解析 LLM 返回结果的方法
    def parse_fn(self, llm_response):
        try:
            return json.loads(repair_json(llm_response, ensure_ascii=False))
        except:
            return ""


def worker_func(sample: tuple, model: LLMChain):
    result = model(sample)
    return result


def main():
    api_keys = ["key1", "key2", "key3", "key4"]
    model = "test_model"
    api_base = "https://ark.cn-beijing.volces.com/api/v3"

    worker = MultiThreadingRunner(200)
    chain = LLMChain(
        client_group=LLMClientGroup(
            [
                APIConfig(model, api_base, api_key)
                for api_key in api_keys
            ]
        ),
        prompt_template=CustomPromptTemplate()
    )
    # 5e4 个样本，会均匀的分配给每个 key
    # LLMChain 内部每次都会选择任务最少的 api-key 进行调用
    samples = [(i, i + 1) for i in range(int(5e4))]
    all_results = worker(
        samples=samples,
        worker_func=partial(worker_func, model=chain),
    )
    print(all_results)


if __name__ == "__main__":
    main()
```
