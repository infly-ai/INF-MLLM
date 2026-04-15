"""Command-line interface for Infinity-Parser2."""

import argparse
import os
import sys
from typing import List, Optional

from . import InfinityParser2


def parse_bool(value: str) -> bool:
    """Convert string to boolean."""
    if value.lower() in ("true", "1", "yes"):
        return True
    elif value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="parser",
        description="Infinity-Parser2: Document parsing tool using Infinity-Parser2-Pro model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a PDF file (default: doc2json -> markdown output)
  parser demo_data/demo.pdf

  # Parse with doc2md task type
  parser demo_data/demo.pdf --task doc2md

  # Parse with custom prompt
  parser demo_data/demo.pdf --task custom --prompt "Please transform the document's contents into Markdown format."

  # Parse multiple files
  parser demo_data/demo.pdf demo_data/demo.png --output-dir ./results

  # Parse a directory
  parser demo_data --output-dir ./results

  # Output raw JSON
  parser demo_data/demo.pdf --output-format json

  # Use transformers backend
  parser demo_data/demo.pdf --backend transformers

  # Use vllm-server backend
  parser demo_data/demo.pdf --backend vllm-server --api-url http://localhost:8000/v1/chat/completions
        """,
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Input file(s) or directory path. Supports PDF, PNG, JPG, JPEG, WEBP.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory. If not provided, result is printed to stdout.",
    )
    parser.add_argument(
        "--task",
        default="doc2json",
        choices=["doc2json", "doc2md", "custom"],
        help="Parsing task type. Defaults to 'doc2json'.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt used only when --task custom.",
    )
    parser.add_argument(
        "--output-format",
        default="md",
        choices=["md", "json"],
        help="Output format. Defaults to 'md'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference. Defaults to 4.",
    )
    parser.add_argument(
        "--backend",
        default="vllm-engine",
        choices=["transformers", "vllm-engine", "vllm-server"],
        help="Inference backend. Defaults to 'vllm-engine'.",
    )
    parser.add_argument(
        "--model-name",
        default="infly/Infinity-Parser2-Pro",
        help="Model name on HuggingFace Hub or local path.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for vllm-engine backend.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/v1/chat/completions",
        help="API URL for vllm-server backend.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for vllm-server backend.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Model cache directory.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=2048,
        help="Minimum number of pixels for image input (transformers backend only).",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=16777216,
        help="Maximum number of pixels for image input (transformers backend only).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Infinity-Parser2 0.1.0",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_paths = args.input
    if len(input_paths) == 1 and os.path.isdir(input_paths[0]):
        input_data = input_paths[0]
    else:
        input_data = input_paths

    if args.verbose:
        print(f"[Infinity-Parser2] Backend: {args.backend}")
        print(f"[Infinity-Parser2] Model: {args.model_name}")
        print(f"[Infinity-Parser2] Task: {args.task}")
        print(f"[Infinity-Parser2] Input: {input_data}")

    try:
        parser_client = InfinityParser2(
            model_name=args.model_name,
            backend=args.backend,
            tensor_parallel_size=args.tensor_parallel_size,
            api_url=args.api_url,
            api_key=args.api_key,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            model_cache_dir=args.model_cache_dir,
        )

        result = parser_client.parse(
            input_data=input_data,
            task_type=args.task,
            custom_prompt=args.prompt,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            output_format=args.output_format,
        )

        if result is not None:
            if isinstance(result, dict):
                for path, content in result.items():
                    print(f"=== {path} ===")
                    print(content)
            elif isinstance(result, list):
                for item in result:
                    print(item)
            else:
                print(result)
        elif args.verbose:
            print("[Infinity-Parser2] Results saved to output directory.")

        return 0

    except Exception as e:
        print(f"[Infinity-Parser2] Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
