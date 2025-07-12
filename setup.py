from setuptools import setup, find_packages

setup(
    name="easy_tools",
    version="0.1.0",  # 包的版本号
    packages=find_packages(),  # 自动找到所有包
    install_requires=["numpy", "pandas", "openpyxl", "regex", "omegaconf"],
    extras_require={
        "torch": ["torch"],  # 可选依赖
    },
    author="silverbeats",
    author_email="silverbeats@qq.com",
    description="This repository stores the tool methods or tool classes that I commonly use in my daily coding process.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[  # 一些分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
