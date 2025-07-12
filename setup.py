from setuptools import setup, find_packages

setup(
    name="easy_tools",
    version='0.1.0',  # 包的版本号
    packages=find_packages(),  # 自动找到所有包
    install_requires=[
        "numpy>=2.0.2",
        "pandas>=2.3.0",
        "openpyxl>=3.1.5",
        "regex>=2024.11.6",
        "omegaconf>=2.3.0"
    ],
    extras_require={
        "torch": ["torch>=2.7.1"],  # 可选依赖
    },
    author="silverbeats",  # 作者名
    author_email="silverbeats@qq.com",  # 作者邮箱
    description="This repository stores the tool methods or tool classes that I commonly use in my daily coding process.",  # 包的简短描述
    long_description=open('README.md', 'r', encoding='utf-8').read(),  # 更长的描述，通常是README的内容
    long_description_content_type='text/markdown',  # 描述的格式
    classifiers=[  # 一些分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # 指定支持的Python版本
)
