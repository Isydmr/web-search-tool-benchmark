# Fake News Benchmark for Web-Search Models

This project benchmarks how well web-search-grounded models detect and repair subtle fake-news variants. Each run fetches recent news, injects a controlled false claim, evaluates model repairs, and publishes a public-safe snapshot for the static viewer.

Live demo: [GitHub Pages viewer](https://isydmr.github.io/web-search-tool-benchmark/viewer/index.html)

## System Overview

[![High-level system diagram](docs/high-level-system.svg)](docs/high-level-system.drawio)

Editable source: [docs/high-level-system.drawio](docs/high-level-system.drawio)

## What It Does

- Fetches recent stories from major news sources
- Creates one controlled fake variant per article
- Runs web-search-enabled models to verify and repair the article
- Scores repair quality and publishes a static viewer snapshot

Default model families: OpenAI, Perplexity, Anthropic, and Gemini.

## Run Locally

1. Clone the repo and install dependencies:

```bash
git clone https://github.com/Isydmr/web-search-tool-benchmark.git
cd web-search-tool-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

2. Create a `.env` file:

```env
OPENAI_API_KEY=...
PERPLEXITY_API_KEY=...    # optional
ANTHROPIC_API_KEY=...     # optional
GEMINI_API_KEY=...        # optional
ENABLED_WEB_SEARCH_MODELS=gpt-5.4,perplexity:sonar,claude-sonnet-4-6,gemini-3.1-pro-preview
```

`OPENAI_API_KEY` is required to generate the manipulated variants. Add any provider keys you want to benchmark. `GEMINI_API_KEY` and `GOOGLE_API_KEY` are both accepted. Set `ENABLED_WEB_SEARCH_MODELS=all` to run every configured search model.

3. Run the full benchmark and serve the viewer locally:

```bash
./scripts/run_end_to_end.sh --serve
```

Then open [http://localhost:8877/viewer/index.html](http://localhost:8877/viewer/index.html).

## Outputs

- Local run artifacts: `runs/<run_id>/`
- Public viewer dataset: `data/latest/`
- Static viewer entrypoint: `viewer/index.html`

## License

MIT
