"""Setup configuration for infinity_parser2 package."""

from setuptools import setup, find_packages

install_requires = [
    "transformers==5.3.0",
    "tokenizers>=0.22.2",
    "qwen-vl-utils>=0.0.14",
    "Pillow>=9.0.0",
    "pypdf>=3.0.0",
    "pymupdf>=1.20.0",
    "openai>=1.0.0",
    "msgspec>=0.19.0",
    "pybase64>=1.4.2",
    "gguf>=0.17.1",
    "cbor2>=5.7.0",
    "py-cpuinfo>=9.0.0",
    "distro>=1.9.0",
    "openai_harmony>=0.0.4",
    "fastapi>=0.135.1",
    "starlette>=0.50.0",
    "annotated_doc>=0.0.4",
    "typing_inspection>=0.4.2",
    "llguidance>=1.3.0",
    "diskcache>=5.6.3",
    "xgrammar>=0.1.29",
    "partial_json_parser>=0.2.1.1.post6",
    "huggingface-hub>=0.24.0",
    "numpy==2.4.3",
    "scikit-learn==1.8.0",
    "scipy==1.17.1",
    "opencv-python-headless>=4.13.0.92"
]

setup(
    name="infinity_parser2",
    version="0.1.0",
    description="Document parsing Python package supporting PDF and image parsing using Infinity-Parser2-Pro model.",
    author="INF Tech",
    author_email="contact@inftech.ai",
    url="https://github.com/infly-ai/INF-MLLM",
    packages=find_packages(),
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
