from setuptools import setup, find_packages

setup(
    name="infinity_parser_cli",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "parser=inference.main:main",
        ],
    },
)
