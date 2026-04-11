"""Setup configuration for infinity_parser2 package."""

from setuptools import setup, find_packages

setup(
    name="infinity_parser2",
    version="0.1.0",
    description="Document parsing Python package supporting PDF and image parsing using Infinity-Parser2-Pro model.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="INF Tech",
    author_email="contact@inftech.ai",
    url="https://github.com/infly-ai/INF-MLLM",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.10.0",
        "transformers>=5.3.0",
        "vllm>=0.17.1",
        "qwen-vl-utils>=0.0.14",
        "Pillow>=9.0.0",
        "pypdf>=3.0.0",
        "pymupdf>=1.20.0",
        "openai>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="document parsing PDF image OCR Qwen VL",
)
