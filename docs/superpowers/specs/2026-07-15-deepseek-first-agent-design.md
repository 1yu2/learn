# DeepSeek First Agent Design

## Goal

Make `examples/01_first_agent/01_agent.py` run against the official DeepSeek API with the repository's installed Agno version.

## Scope

Only the target example will change. Shared runtime configuration and other examples remain untouched.

## Configuration

- Load the repository `.env` file with `python-dotenv`.
- Read the model ID from `AGNO_MODEL_ID`, defaulting to `deepseek-v4-flash`.
- Read the API key from `DEEPSEEK_API_KEY` first.
- Fall back to `OPENAI_API_KEY` for compatibility with the repository's current local configuration.
- Fail before making a request when neither key is configured, with an error that names the supported variables.

## Agent Construction

Replace the undefined `Ollama` model with Agno's native `DeepSeek` model. Keep the existing agent name, description, Markdown output, prompt, and streaming behavior.

## Verification

1. A regression check must fail against the current file because it references `Ollama` instead of constructing `DeepSeek`.
2. After the change, verify configuration loading and `DeepSeek` construction without sending a network request.
3. Run Ruff against the target file.
4. When a valid key and network access are available, run the example once against `https://api.deepseek.com`.

## Security

Do not print or commit API key values. A key currently present in `.env.example` should be rotated and replaced with an empty placeholder separately because that file is intended to be committed.
