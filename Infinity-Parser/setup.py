from setuptools import setup, find_packages

setup(
    name="infinity_parser_cli",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "vllm>=0.10.1.1",
        "transformers>=4.40.0",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "parser=inference.main:main",
        ],
    },
)
