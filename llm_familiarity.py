#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai",
#   "transformers",
#   "torch",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///

import argparse
import os
import sys
import math
import torch
import re
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, Optional, List, Union, Tuple

DEBUG = False

# Prompt as used by Brysbaert et al. (2025)
WORD_FAMILIARITY = """
Familiarity is a measure of how familiar something is. A word is very FAMILIAR if you see/hear it often and it is easily recognisable. 
In contrast, a word is very UNFAMILIAR if you rarely see/hear it and it is relatively unrecognisable. Please indicate how familiar you 
think each word is on a scale from 1 (VERY UNFAMILIAR) to 7 (VERY FAMILIAR), with the midpoint representing moderate familiarity. 
The word is: {word}. Only answer a number from 1 to 7. Please limit your answer to numbers.
"""

def get_device() -> str:
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def process_word_openai(
        client, word: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> Tuple[str, Dict[str, float]]:
    """Process a word using OpenAI API and return results."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": WORD_FAMILIARITY.format(word=word)
            }
        ],
        max_completion_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=20,
    )

    result = response.choices[0].message.content.strip()

    # Initialize a dictionary to store logprobs for each possible answer
    answer_logprobs = {str(i): float('-inf') for i in range(1, 8)}

    # Get the logprobs from the response
    logprobs = response.choices[0].logprobs.content[0].top_logprobs

    # Update the answer_logprobs dictionary
    for n, logprob in enumerate(logprobs, start=1):
        token = logprob.token.strip()
        if DEBUG:
            print(f"Word: {word}\tn={n}\tResponse: {token:>10}\t{logprob.logprob}")

        if token in answer_logprobs:
            answer_logprobs[token] = logprob.logprob

    return result, answer_logprobs

def calculate_transformer_logprobs(
        model,
        tokenizer,
        prompt: str
    ) -> Tuple[str, Dict[str, float]]:

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_token_logits = outputs.logits[0, -1, :]

    next_token_id = torch.argmax(last_token_logits).item()

    # Decode the single token. Use skip_special_tokens=True to avoid printing things like <|endoftext|>
    # Add clean_up_tokenization_spaces=False if spaces are important for debugging, but for result True is usually better
    result_token_text = tokenizer.decode([next_token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    result = result_token_text.strip() # Use the stripped decoded token as the result

    if DEBUG:
        print(f"Top predicted next token ID: {next_token_id}, Decoded: '{result_token_text}', Stripped Result: '{result}'")
        # Optional: Print top few tokens for context
        top_k_logits, top_k_indices = torch.topk(last_token_logits, 5)
        print("Top 5 predicted next tokens:")
        for i in range(5):
            token_id = top_k_indices[i].item()
            token_text = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            log_prob = torch.log_softmax(last_token_logits, dim=-1)[token_id].item()
            print(f"  - '{token_text}' (ID: {token_id}), LogProb: {log_prob:.4f}")

    # --- Now, calculate the specific log probabilities for ratings 1-7 ---
    # Convert logits to log probabilities (do this only once)
    log_probs = torch.log_softmax(last_token_logits, dim=-1).cpu() # Move to CPU for easier handling

    rating_logprobs = {}
    target_token_ids = {}

    # Iterate over the ratings 1-7 and find the best logprob for each
    for i in range(1, 8):
        rating_str = str(i)
        current_best_logprob = float('-inf')
        selected_token_id = -1
        candidate_token_ids = []
        token_debug_info = [] # For debug printing

        # Check tokenization with and without space prefix
        tokens_with_space = tokenizer.tokenize(" " + rating_str)
        if len(tokens_with_space) == 1:
            tid_space = tokenizer.convert_tokens_to_ids(tokens_with_space[0])
            if tid_space != tokenizer.unk_token_id:
                candidate_token_ids.append(tid_space)

        tokens_without_space = tokenizer.tokenize(rating_str)
        if len(tokens_without_space) == 1:
            tid_normal = tokenizer.convert_tokens_to_ids(tokens_without_space[0])
            if tid_normal != tokenizer.unk_token_id:
                if tid_normal not in candidate_token_ids:
                    candidate_token_ids.append(tid_normal)

        if not candidate_token_ids:
            if DEBUG:
                print(f"Warning: Could not find single token ID for rating '{rating_str}' or ' {rating_str}'")
            rating_logprobs[rating_str] = float('-inf')
            continue

        # Find the highest logprob among the valid token IDs for this rating
        for token_id in candidate_token_ids:
             # Ensure token_id is within the valid range of the vocabulary size used in log_probs
            if 0 <= token_id < log_probs.shape[0]:
                logprob = log_probs[token_id].item()
                token_debug_info.append(f"{tokenizer.decode([token_id])}({token_id}): {logprob:.4f}")
                if logprob > current_best_logprob:
                    current_best_logprob = logprob
                    selected_token_id = token_id

        rating_logprobs[rating_str] = current_best_logprob
        target_token_ids[rating_str] = selected_token_id # Store the ID that gave the best logprob

        # Consolidate debug printing for rating logprobs
        if DEBUG:
           token_repr = ", ".join(token_debug_info)
           print(f"Rating {rating_str}: Found tokens=[{token_repr}], Best logprob={current_best_logprob:.4f} (using token ID {selected_token_id})")

    # Ensure all ratings 1-7 have an entry in the final dictionary
    final_logprobs = {str(i): rating_logprobs.get(str(i), float('-inf')) for i in range(1, 8)}

    if DEBUG and result not in [str(i) for i in range(1, 8)]:
        print(f"Warning: The top predicted token result ('{result}') is not one of '1'-'7'. The weighted mean will be based on calculated logprobs for '1'-'7'.")

    # Return the actual decoded top token as 'result', and the calculated logprobs for ratings 1-7
    return result, final_logprobs

def process_word_transformers(
        model,
        tokenizer,
        word: str
    ) -> Tuple[str, Dict[str, float]]:
    """Process a word using transformers model and return results."""
    prompt = WORD_FAMILIARITY.format(word=word)

    try:
        result, answer_logprobs = calculate_transformer_logprobs(model, tokenizer, prompt)

        if DEBUG:
            print(f"Word: {word}, Result: {result}")
            for i in range(1, 8):
                print(f"  Rating {i}: logprob={answer_logprobs.get(str(i), float('-inf')):.4f}")

    except Exception as e:
        print(f"Error processing word {word}: {str(e)}")
        answer_logprobs = None

    return result, answer_logprobs

def setup_transformers_model(
        model_name_or_path: str
    ) -> None:
    """Load transformers model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_name_or_path}...")

    # Determine device
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Ensure we have padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a padding token if needed
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load model with parameters appropriate for the device
    model_kwargs = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    # If not using device_map, manually move to device
    if "device_map" not in model_kwargs:
        model = model.to(device)

    model.eval()

    return model, tokenizer

def process_file(
        file_path: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        max_tokens: int = 100,
        temperature: float = 0.0,
        hf_model_name: Optional[str] = None
    ) -> None:
    """Process a file line by line using either OpenAI's API or Transformers."""

    # Initialize the appropriate client based on the chosen backend
    if hf_model_name:
        # Set up transformers model
        hf_model, hf_tokenizer = setup_transformers_model(hf_model_name)
        client = None
        print(f"Using Transformers with model: {hf_model_name}")
    else:
        # Use OpenAI API
        client = OpenAI(api_key=api_key)
        hf_model = hf_tokenizer = None
        print(f"Using OpenAI API with model: {model}")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            words = [line.strip() for line in file if line.strip()]

        print(f"{'nr'}\t{'word':<10}\t{'highest'}\t{'mean'}\t1\t2\t3\t4\t5\t6\t7")

        for line_number, word in enumerate(words, 1):
            try:
                # Process using either OpenAI or transformers
                if hf_model:
                    result, answer_logprobs = process_word_transformers(
                                                    hf_model, hf_tokenizer, word
                                                )
                else:
                    result, answer_logprobs = process_word_openai(
                                                    client, word, model, max_tokens, temperature
                                                )
                if answer_logprobs is None:
                    print(f"Error: No logprobs returned for word '{word}'")
                    continue

                # Convert logprobs to probabilities  
                valid_logprobs = {k: v for k, v in answer_logprobs.items() if v > float('-inf')}
                if valid_logprobs:
                    # Normalize probabilities
                    max_logprob = max(valid_logprobs.values())
                    # Subtract max for numerical stability
                    scaled_logprobs = {k: v - max_logprob for k, v in valid_logprobs.items()}

                    # Convert to probabilities
                    exps = {k: math.exp(v) for k, v in scaled_logprobs.items()}
                    total = sum(exps.values())
                    answer_probs = {k: v / total for k, v in exps.items()}

                    # Fill in any missing values with very small probabilities
                    for i in range(1, 8):
                        rating = str(i)
                        if rating not in answer_probs:
                            answer_probs[rating] = 1e-10
                else:
                    # If no valid logprobs, assign equal probabilities
                    answer_probs = {str(i): 1.0/7.0 for i in range(1, 8)}

                if DEBUG:
                    print(f"answer_logprobs: {answer_logprobs}")
                    print(f"answer_probs: {answer_probs}")

                # Calculate weighted average rating
                weighted_rating = sum(int(rating) * prob for rating, prob in answer_probs.items())

                print(f"{line_number}\t{word:<10}\t{result}\t{weighted_rating:.3f}\t", end="")
                for rating in range(1, 8):
                    print(f"{answer_probs[str(rating)]:.4f}", end="\t")
                print()

            except Exception as e:
                print(f"Error processing line {line_number}: {str(e)}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Process a file line by line using OpenAI's GPT-4 or Hugging Face Transformers"
    )
    parser.add_argument("file_path", help="Path to the input file")
    parser.add_argument("--api_key", help="OpenAI API key (not needed for transformers)")
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-08-06", 
        help="OpenAI model to use (default: gpt-4o-2024-08-06)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens in the response (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for response generation (default: 0.0)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    # Transformers options
    hf_group = parser.add_argument_group('Hugging Face Transformers options')
    hf_group.add_argument(
        "--hf_model",
        help="Hugging Face model name or path (e.g., 'Qwen/Qwen2.5-3B-Instruct')"
    )

    args = parser.parse_args()

    # Set debug flag
    global DEBUG
    DEBUG = args.debug

    # Determine which backend to use
    using_transformers = args.hf_model is not None

    # Check if API key is provided when using OpenAI
    if not using_transformers and not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            print("Error: OpenAI API key is required when not using transformers. Provide it using --api_key or set the OPENAI_API_KEY environment variable.")
            sys.exit(1)

    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")

    process_file(
        args.file_path,
        args.api_key,
        args.model,
        args.max_tokens,
        args.temperature,
        args.hf_model,
    )

if __name__ == "__main__":
    main()
