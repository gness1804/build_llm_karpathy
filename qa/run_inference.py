#!/usr/bin/env python3
"""
Script to run inference on prompts from test_prompts.md

Usage:
    qa/run_inference.py --prompt 'parent overstepping' --checkpoint 'path/to/checkpoint.pt'
    qa/run_inference.py --prompt 'simple romantic miscommunication' --checkpoint 'path/to/checkpoint1.pt'
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_env_file(env_path):
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        dict: Dictionary of environment variables
    """
    env_vars = {}
    if not env_path.exists():
        print(f"⚠️  Warning: .env file not found at {env_path}")
        return env_vars
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # Strip trailing backslashes and whitespace (common in shell script exports)
                value = value.rstrip(' \\').strip()
                env_vars[key] = value
    
    return env_vars


def parse_prompts_file(prompts_path):
    """
    Parse test_prompts.md to extract prompts and stems, creating a mapping.
    
    Args:
        prompts_path: Path to test_prompts.md
        
    Returns:
        dict: Dictionary mapping shorthand names to dicts with 'prompt' and 'stem' keys
    """
    prompts = {}
    
    if not prompts_path.exists():
        print(f"❌ Error: test_prompts.md not found at {prompts_path}")
        sys.exit(1)
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match prompt sections with STEM
    # Matches: ### Easy 1 – Simple romantic miscommunication
    # Then captures PROMPT: and STEM: sections
    pattern = r'###\s+(?:Easy|Medium|Hard)\s+\d+\s*[–-]\s*(.+?)\n\nPROMPT:\n(.+?)\n\nSTEM:\n(.+?)(?=\n\nWhat to look for:)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        title = match.group(1).strip()
        prompt_text = match.group(2).strip()
        stem_text = match.group(3).strip()
        
        # Create shorthand name from title (lowercase, replace spaces/special chars)
        shorthand = title.lower()
        # Remove special characters and normalize spaces
        shorthand = re.sub(r'[^\w\s]', '', shorthand)
        shorthand = re.sub(r'\s+', ' ', shorthand).strip()
        
        prompts[shorthand] = {
            'prompt': prompt_text,
            'stem': stem_text
        }
    
    # Handle special case: "Hard 4 - longer prompt about friend group and growing distant."
    # This one doesn't follow the standard format - it has PROMPT and STEM but no "PROMPT:" label
    hard4_pattern = r'###\s+Hard\s+4\s*[–-]\s*(.+?)\n\n(.+?)\n\nSTEM:\n(.+?)(?=\n\nWhat to look for:)'
    hard4_match = re.search(hard4_pattern, content, re.DOTALL)
    if hard4_match:
        title = hard4_match.group(1).strip()
        prompt_text = hard4_match.group(2).strip()
        stem_text = hard4_match.group(3).strip()
        shorthand = title.lower()
        shorthand = re.sub(r'[^\w\s]', '', shorthand)
        shorthand = re.sub(r'\s+', ' ', shorthand).strip()
        prompts[shorthand] = {
            'prompt': prompt_text,
            'stem': stem_text
        }
    
    return prompts


def find_prompt_by_shorthand(shorthand, prompts):
    """
    Find a prompt by shorthand name (fuzzy matching).
    
    Args:
        shorthand: The shorthand name to search for
        prompts: Dictionary of prompts (with 'prompt' and 'stem' keys)
        
    Returns:
        tuple: (matched_key, prompt_dict) or (None, None) if not found
    """
    shorthand_lower = shorthand.lower().strip()
    
    # Exact match
    if shorthand_lower in prompts:
        return shorthand_lower, prompts[shorthand_lower]
    
    # Partial match - check if shorthand is contained in any key
    for key, prompt_dict in prompts.items():
        if shorthand_lower in key or key in shorthand_lower:
            return key, prompt_dict
    
    # Fuzzy match - check if any words match
    shorthand_words = set(shorthand_lower.split())
    best_match = None
    best_score = 0
    
    for key, prompt_dict in prompts.items():
        key_words = set(key.split())
        common_words = shorthand_words & key_words
        if common_words and len(common_words) > best_score:
            best_score = len(common_words)
            best_match = (key, prompt_dict)
    
    return best_match if best_match else (None, None)


def list_available_prompts(prompts):
    """Print all available prompts."""
    print("\nAvailable prompts:")
    print("=" * 60)
    for key in sorted(prompts.keys()):
        # Show first 60 chars of prompt
        prompt_text = prompts[key]['prompt']
        preview = prompt_text[:60].replace('\n', ' ')
        print(f"  {key:40} | {preview}...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on prompts from test_prompts.md',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qa/run_inference.py --prompt 'parent overstepping' --checkpoint 'path/to/checkpoint.pt'
  qa/run_inference.py --prompt 'simple romantic miscommunication' --checkpoint 'checkpoints/model.pt'
  qa/run_inference.py --prompt 'value difference around ambition' --checkpoint 'checkpoints/model.pt'
  qa/run_inference.py --list  # List all available prompts
        """
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='Shorthand name of the prompt to run (e.g., "parent overstepping")'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help='Path to the checkpoint file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=["gpt2", "openai_backend", "from_scratch"], 
        default="gpt2"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available prompts and exit'
    )
    parser.add_argument(
        '--use-stem',
        action='store_true',
        help='Include the STEM (first sentence of ideal answer) after ANSWER: to nudge the model'
    )
    
    args = parser.parse_args()
    
    # Load prompts
    prompts_path = SCRIPT_DIR / 'test_prompts.md'
    prompts = parse_prompts_file(prompts_path)
    
    if args.list:
        list_available_prompts(prompts)
        sys.exit(0)
    
    if not args.prompt:
        print("❌ Error: --prompt is required (use --list to see available prompts)")
        sys.exit(1)
    
    if not args.checkpoint and args.model_type not in ["gpt2", "openai_backend"]:
        print("❌ Error: --checkpoint is required when not using a built-in model type")
        sys.exit(1)
    
    # Find the prompt
    matched_key, prompt_dict = find_prompt_by_shorthand(args.prompt, prompts)
    
    if not matched_key:
        print(f"❌ Error: Prompt '{args.prompt}' not found")
        print("\nAvailable prompts:")
        for key in sorted(prompts.keys()):
            print(f"  - {key}")
        sys.exit(1)
    
    prompt_text = prompt_dict['prompt']
    stem_text = prompt_dict.get('stem', '')
    
    print(f"✅ Found prompt: {matched_key}")
    if args.checkpoint:
        print(f"   Using checkpoint: {args.checkpoint}")
    print(f"   Using model type: {args.model_type.upper()}")
    
    if args.use_stem:
        if stem_text:
            print(f"   Using STEM: {stem_text[:60]}...")
        else:
            print(f"   ⚠️  Warning: --use-stem specified but no STEM found for this prompt")
    
    # Check if checkpoint exists (only if provided)
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        
        if not checkpoint_path.exists():
            print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
    
    # Load .env file
    env_path = SCRIPT_DIR / '.env'
    env_vars = load_env_file(env_path)
    
    # Prepare environment variables
    env = os.environ.copy()
    
    # Load variables from .env file
    for key, value in env_vars.items():
        env[key] = value
    
    # Clean the prompt text (strip any extra whitespace)
    prompt_text_clean = prompt_text.strip()
    
    # Format prompt to match training data format: QUESTION: <text>\n\nANSWER:
    if args.use_stem and stem_text:
        # Include stem after ANSWER:
        stem_clean = stem_text.strip()
        formatted_prompt = f"QUESTION: {prompt_text_clean}\n\nANSWER: {stem_clean}"
    else:
        # Standard format without stem
        formatted_prompt = f"QUESTION: {prompt_text_clean}\n\nANSWER:"
    
    # Override with command-line arguments
    if checkpoint_path:
        env['CHECKPOINT_PATH'] = str(checkpoint_path)
    env['PROMPT'] = formatted_prompt
    env['MODE'] = 'inference'  # Ensure MODE is set
    env['MODEL_TYPE'] = args.model_type
    
    # Print what we're running
    print(f"\n📏 Prompt preview (first 100 chars):")
    print(f"   {prompt_text_clean[:100]}...")
    print(f"\n📋 Full formatted prompt (first 150 chars):")
    print(f"   {formatted_prompt[:150]}...")
    print(f"\n🚀 Running inference...")
    if checkpoint_path:
        print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Environment variables from .env: {', '.join(env_vars.keys())}")
    
    # Run the inference script
    inference_script = PROJECT_ROOT / 'scripts' / 'load_checkpoint.py'
    
    if not inference_script.exists():
        print(f"❌ Error: Inference script not found at {inference_script}")
        sys.exit(1)
    
    try:
        # Run the script with the prepared environment
        result = subprocess.run(
            [sys.executable, str(inference_script)],
            env=env,
            cwd=str(PROJECT_ROOT),
            check=False
        )
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

