"""Setup configuration for infinity_parser2 package."""

from setuptools import setup, find_packages

install_requires = [
    "transformers>=5.3.0",
    "tokenizers>=0.22.2",
    "qwen-vl-utils>=0.0.14",
    "Pillow>=9.0.0",
    "pypdf>=3.0.0",
    "pymupdf>=1.20.0",
    "openai>=1.0.0",
    "huggingface-hub>=0.24.0",
    "tqdm>=4.66.0",
    "loguru>=0.7.0",
]

setup(
    name="infinity_parser2",
    version="0.4.0",
    description="Document parsing Python package supporting PDF and image parsing using Infinity-Parser2-Pro model.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="INF Tech",
    author_email="contact@inftech.ai",
    url="https://github.com/infly-ai/INF-MLLM",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.12",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="document parsing",
    entry_points={
        "console_scripts": [
            "parser=infinity_parser2.cli:main",
        ],
    },
)
