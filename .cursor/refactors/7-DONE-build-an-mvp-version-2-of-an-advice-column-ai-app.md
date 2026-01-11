# Build An Mvp Version 2 Of An Advice Column Ai App

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents
I'm trying to build an advice column app that takes in a user question and spits out an LLM-generated response. The question can be any interpersonal question, and the response needs to be in my own voice as closely as possible.

I've already built version 1 in the current repo. V1 simply took user input from a command line, and spat out LLM input. This is based on @run_inference.py  using version 1. I also built different functionality that takes in a user question and a response from the base LLM. It then spits out a score of the base LLM response, strengths and weaknesses of that response, and a revised version that's more in my voice and that addresses the weaknesses. This is what happens with version 3 in @run_inference.py. Want to put these two together into an MVP application.

The application should accept a user question from the command line, run that question through the base model of an LLM (GPT-4.1 mini), take the output of this base model and in turn run the original question in that output through my fine-tuned model to then generate a refined and improved response. The fine-tuned LLM will score the base response and give strengths and weaknesses, but the end user should only see the revised response that is emitted.

I want to build a GUI around this, but right now, I just want to build off the backend first. So I want to build a back-end that takes in a user question and then goes through the process that I just mentioned.

Outline how you would do it. Please give me a set of steps and a proposal first before undertaking any of the work.

## Acceptance criteria
- A back-end system that accepts user input, puts it through the base LLM, puts the base LLM response through a fine-tuned LLM, and then finally spits out a strengthened revised response for the user.
- The user should only enter the original response and then see the revised response. But under the hood, the program is taking the base response model, putting it through the fine-tune model, getting a score of strengths and weaknesses, and finally the revised response.
- Application should use both models in the same flow. Right now, I have to manually break up this flow by first running a question to the base model and then taking that response and running it to the fine-tune model. I want my application to connect those two things the only thing the user has to do manually is enter in a question.
- On @openai_backend.py line 8, you can see that the model to use is brought in through an environment variable. But so right now when I run this, it only is able to use one model at a time, but I want my MVP to have access to both models to do the flow that I just mentioned.

## Cursor agent proposal.

```markdown
# MVP Version 2 Implementation Proposal

## Overview
Build a backend system that chains two LLM calls: base model (GPT-4.1-mini) → fine-tuned model (ft:gpt-4.1-mini-2025-04-14:personal:friendly-advice-01092026:CwGsaVcA) to generate refined advice column responses.

## Implementation Steps

### Step 1: Update `models/openai_backend.py`
**Purpose**: Support multiple models in the same flow

**Changes**:
- Add support for `BASE_MODEL` and `FINE_TUNED_MODEL` environment variables
- Keep backward compatibility with `OPENAI_MODEL` (defaults to fine-tuned model)
- Modify `generate_answer()` to accept an optional `model` parameter to override the default
- Default behavior: `OPENAI_MODEL` = `FINE_TUNED_MODEL` if not set, falls back to `"ft:gpt-4.1-mini"`

**New function signature**:
```python
def generate_answer(input: str, version: str = "v1", model: str = None) -> str:
    # If model is provided, use it; otherwise use OPENAI_MODEL env var
```

### Step 2: Create `qa/advice_mvp.py`
**Purpose**: Main entry point for the MVP application

**Features**:
1. **CLI Interface**:
   - `--question` or `--prompt`: User's question (required)
   - `--output-dir`: Directory to save output (default: `outputs/inference/mvp`)
   - `--save-output`: Flag to save output (default: True)
   - `--max-retries`: Number of retries for base model (default: 2)
   - `--verbose`: Show debug information including scores, strengths, weaknesses

2. **Core Flow**:
   ```
   User Question → Base Model (v1) → Draft Response
   → Fine-tuned Model (v3) → Score/Strengths/Weaknesses/Revised Response
   → Extract & Display Revised Response
   ```

3. **Error Handling**:
   - Retry logic for base model calls (1-2 retries with exponential backoff)
   - If all retries fail, exit with clear error message
   - Log all errors with context

4. **Output Handling**:
   - **Stdout**: Only display `REVISED_RESPONSE` section (clean output for user)
   - **Logging**: Log full response including SCORE, STRENGTHS, WEAKNESSES for debugging
   - **File Output**: Save complete response to file (including all sections) with timestamp

5. **File Output Format**:
   - Filename: `mvp_output_YYYYMMDD_HHMMSS.txt`
   - Contents:
     - Timestamp
     - Original question
     - Base model draft response
     - Full fine-tuned model response (SCORE, STRENGTHS, WEAKNESSES, REVISED_RESPONSE)
     - Extracted revised response (what user sees)

### Step 3: Add Response Parsing Utilities
**Purpose**: Extract sections from v3 response format

**New module**: `qa/mvp_utils.py` (or add to existing utils)

**Functions**:
- `parse_v3_response(response: str) -> dict`: Parse SCORE, STRENGTHS, WEAKNESSES, REVISED_RESPONSE
- `extract_revised_response(response: str) -> str`: Extract just the REVISED_RESPONSE section
- Handle edge cases (missing sections, malformed responses)

### Step 4: Environment Variable Configuration
**Purpose**: Support separate model configuration

**Environment Variables**:
- `BASE_MODEL`: Default `"gpt-4.1-mini"` (base OpenAI model)
- `FINE_TUNED_MODEL`: Default `"ft:gpt-4.1-mini-2025-04-14:personal:friendly-advice-01092026:CwGsaVcA"`
- `OPENAI_API_KEY`: Required (existing)
- `TEMPERATURE`, `TOP_P`, `MAX_NEW_TOKENS`: Existing (used for both models)

### Step 5: Integration with Existing Code
**Purpose**: Reuse existing functionality

**Reuse**:
- `models/openai_backend.py`: `generate_answer()` function
- `models/prompts.py`: `ADVICE_COLUMNIST_SYSTEM_PROMPT` (v1) and `SYSTEM_PROMPT_V3` (v3)
- Output saving pattern from `scripts/load_checkpoint.py`

**New**:
- Chaining logic in `qa/advice_mvp.py`
- Response parsing utilities

## File Structure

```
build_llm_karpathy/
├── models/
│   ├── openai_backend.py          # Updated to support model parameter
│   └── prompts.py                 # Existing (no changes)
├── qa/
│   ├── advice_mvp.py              # NEW: Main MVP script
│   ├── mvp_utils.py               # NEW: Response parsing utilities
│   ├── run_inference.py           # Existing (no changes)
│   └── test_prompts_v3.md         # Existing (reference)
└── outputs/
    └── inference/
        └── mvp/                   # NEW: MVP output directory
            └── mvp_output_*.txt
```

## Usage Example

```bash
# Basic usage
python qa/advice_mvp.py --question "My partner and I are having communication issues"

# With custom output directory
python qa/advice_mvp.py --question "How do I set boundaries?" --output-dir outputs/custom

# With verbose logging (shows scores, strengths, weaknesses)
python qa/advice_mvp.py --question "Family conflict" --verbose

# Disable file saving
python qa/advice_mvp.py --question "Question here" --no-save-output
```

## Error Scenarios Handled

1. **Base Model Failure**:
   - Retry up to 2 times with exponential backoff
   - If all retries fail: Exit with error, don't show partial response

2. **Fine-tuned Model Failure**:
   - Single retry attempt
   - If fails: Exit with error (don't show base model response)

3. **Malformed v3 Response**:
   - Attempt to extract REVISED_RESPONSE even if format is slightly off
   - Log warning if sections are missing
   - Fallback: Use entire response if parsing fails

4. **Missing Environment Variables**:
   - Clear error messages indicating which variables are missing
   - Suggest defaults

## Testing Considerations

1. **Unit Tests** (future):
   - Test response parsing with various formats
   - Test retry logic
   - Test error handling

2. **Manual Testing**:
   - Test with various question types
   - Test retry scenarios (simulate failures)
   - Verify output format and file saving

## Dependencies

- Existing: `openai` library (already in use)
- Existing: `models/prompts.py` (no new dependencies)
- No new external dependencies required

## Backward Compatibility

- `openai_backend.py` changes maintain backward compatibility
- Existing `run_inference.py` continues to work unchanged
- Environment variable `OPENAI_MODEL` still works (defaults to fine-tuned model)

## Next Steps (Post-MVP)

- GUI wrapper (as mentioned in requirements)
- Additional error recovery strategies
- Caching base model responses
- Batch processing capabilities
- API endpoint (if needed for GUI)
```

<!-- DONE -->
