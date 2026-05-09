# [RAG](https://www.boot.dev/courses/learn-retrieval-augmented-generation)

## prerequisites

- `uv`
- `python`

## initialize the virtual environment

```bash
uv venv
```

## activate the environment

```bash
source .venv/bin/activate
```

## install the required packages

```bash
uv pip install -r requirements.txt
```

> [!TIP]
> to update your `requirements.txt` file, run this command:
> ```bash
> uv pip freeze > requirements.txt

## add your api key

1. Create an account on [Google AI Studio](https://aistudio.google.com/) if you don't already have one.
2. Click the "Create API Key" button. Here are the [docs](https://ai.google.dev/gemini-api/docs/api-key) if you get lost.
3. Copy the API key, then paste it into a new .env file in your project directory. The file should look like this:

```env
GEMINI_API_KEY=your_api_key_here
```

> [!IMPORTANT]
> paste your api key without the quotation marks (e.g., GEMINI_API_KEY=fans890dfnb7923nf1nf90813nf791rbf)

4. Add the `.env` file to your [.gitignore](https://github.com/theantigone/retrieval_augmented_generation-python-bootdev/blob/a29a13ed31a8df243cc9e959dbd60f22cb521431/.gitignore)!
