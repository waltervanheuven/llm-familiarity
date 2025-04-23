# LLM familiarity estimates

Use LLM to obtain word familiarity ratings. Script is based on method described in Brysbaert et al. (2025).

## Installation

<details>

<summary>Click to expand/collapse</summary>

### macOS

Install [brew](https://brew.sh).

Next install `Python` and `uv` using the [Terminal](https://support.apple.com/en-gb/guide/terminal/welcome/mac)

```sh
brew install python@3.12
brew install uv
```

### Windows

Install [scoop](https://scoop.sh).

Next install `Python` and `uv` using the [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.5).

```powershell
scoop bucket add versions
scoop install versions/python312
scoop bucket add main
scoop install main/uv
```

### Clone repository

```sh
git clone https://github.com/waltervanheuven/llm-familiarity.git
```

</details>

To use OpenAI API KEY you can set the `OPENAI_API_KEY` environment variable.
[How to set API KEY](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety). Alternatively, you can set it using `--api_key`.

## Examples

Process file with words (one on each line) using OpenAI model (default `gpt-4o-2024-08-06`, set different model with `--model`).

```sh
uv run llm_familiarity.py words.txt
```

Use a transformer model from [huggingface](https://huggingface.co).

```sh
uv run llm_familiarity.py words.txt --hf_model "Qwen/Qwen2.5-3B-Instruct"
```

Show command line options.

```sh
uv run llm_familiarity.py -h
```

## References

Brysbaert, M., Martínez, G., & Reviriego, P. (2025). Moving beyond word frequency based on tally counting: AI-generated familiarity estimates of words and phrases are a better index of language knowledge. *Behavior Research Methods*. [https://doi.org/10.31234/osf.io/kgevy](https://doi.org/10.31234/osf.io/kgevy)
