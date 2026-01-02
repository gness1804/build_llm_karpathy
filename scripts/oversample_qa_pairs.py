#!/usr/bin/env python3
"""
Oversample Q&A pairs from a master document based on oversample weights.

This script reads a markdown file containing Q&A pairs, duplicates each pair
based on its OVERSAMPLE_WEIGHT using the formula: base_multiplier ^ oversample_weight,
removes the OVERSAMPLE_WEIGHT line, randomizes the order, and writes to a new file.
"""

import argparse
import random
import re
import sys
from pathlib import Path


def parse_qa_pairs(content: str) -> list[tuple[str, int]]:
    """
    Parse Q&A pairs from the document content.
    
    Returns a list of tuples: (qa_pair_content, oversample_weight)
    where qa_pair_content includes QUESTION, ANSWER, and <END_OF_SET> but excludes OVERSAMPLE_WEIGHT.
    Preserves exact formatting of the original content.
    """
    pairs = []
    # Pattern to match entire Q&A pair structure:
    # QUESTION: ... ANSWER: ... OVERSAMPLE_WEIGHT: N ... <END_OF_SET>
    # We capture everything before OVERSAMPLE_WEIGHT and the weight number separately
    pattern = r'(QUESTION:.*?ANSWER:.*?)(?=\s*OVERSAMPLE_WEIGHT:)\s*OVERSAMPLE_WEIGHT:\s*(\d+)\s*<END_OF_SET>'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        qa_content = match.group(1)  # QUESTION and ANSWER with exact formatting preserved
        oversample_weight = int(match.group(2))
        
        # Add END_OF_SET token, preserving original spacing
        # Remove trailing whitespace from qa_content and add END_OF_SET on new line
        qa_content_clean = qa_content.rstrip()
        qa_pair = qa_content_clean + '\n<END_OF_SET>\n'
        pairs.append((qa_pair, oversample_weight))
    
    return pairs


def oversample_pairs(pairs: list[tuple[str, int]], base_multiplier: float, max_copies: int | None = None) -> list[str]:
    """
    Duplicate Q&A pairs based on oversample weights.
    
    Formula: base_multiplier ^ oversample_weight copies of each pair.
    If max_copies is specified, caps the number of copies per pair to prevent over-reliance.
    Supports both integers and floats (fractions) for base_multiplier.
    """
    oversampled = []
    
    for qa_pair, weight in pairs:
        num_copies = base_multiplier ** weight
        # Round to nearest integer since we can't have fractional copies
        num_copies = round(num_copies)
        if max_copies is not None:
            num_copies = min(num_copies, max_copies)
        oversampled.extend([qa_pair] * num_copies)
    
    return oversampled


def main():
    parser = argparse.ArgumentParser(
        description='Oversample Q&A pairs from a master document based on oversample weights.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input markdown file containing Q&A pairs'
    )
    parser.add_argument(
        'base_multiplier',
        type=float,
        help='Base multiplier for oversampling (e.g., 2 means weight 1 = 2 copies, weight 2 = 4 copies). Supports integers and fractions (e.g., 1.5, 2.5)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: input_file with _oversampled suffix)'
    )
    parser.add_argument(
        '-m', '--max-copies-per-example',
        type=int,
        default=None,
        metavar='MAX',
        help='Maximum number of copies per Q&A pair (caps oversampling to prevent training crutches)'
    )
    
    args = parser.parse_args()
    
    # Resolve input file path
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file path
    if args.output:
        output_path = Path(args.output)
    else:
        # Add _oversampled before the extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_oversampled{suffix}"
    
    # Read input file
    try:
        content = input_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse Q&A pairs
    pairs = parse_qa_pairs(content)
    if not pairs:
        print("Warning: No Q&A pairs found in the input file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(pairs)} Q&A pairs in input file.")
    
    # Oversample pairs
    oversampled = oversample_pairs(pairs, args.base_multiplier, args.max_copies_per_example)
    if args.max_copies_per_example:
        print(f"Created {len(oversampled)} total copies after oversampling (max {args.max_copies_per_example} per pair).")
    else:
        print(f"Created {len(oversampled)} total copies after oversampling.")
    
    # Randomize order
    random.shuffle(oversampled)
    print("Randomized order of Q&A pairs.")
    
    # Combine into final content (with blank lines between pairs)
    # Each pair already ends with a newline, so join with a single newline
    # to create blank lines between pairs
    final_content = '\n'.join(oversampled)
    
    # Write output file
    try:
        output_path.write_text(final_content, encoding='utf-8')
        print(f"Successfully wrote oversampled document to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

