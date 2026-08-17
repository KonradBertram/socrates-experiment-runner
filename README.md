# Socrates Experiment Runner

A Streamlit app for pre-testing up to five behavioral intervention variants with `socratesft/socrates-qwen2.5-14b-sft` through the Featherless API.

## What the app does

- Captures experiment context: line of business, customer journey type, and free-text context.
- Compares 2–5 touchpoint/intervention variants.
- Uses researcher-defined answer options and one selected target behavior.
- Generates individual respondent profiles from Socrates-supported demographic dimensions.
- Offers an approximate representative U.S. marginal preset or custom percentages.
- Reproduces each selected demographic marginal separately within every variant.
- Randomly interleaves variant queries and counterbalances response-code positions.
- Shows exact prompt previews and a cost estimate before the paid run.
- Reports overall response distributions, target-behavior effects versus control, and effects by demographic segment.
- Exports setup, overall results, segment results, profile balance, and raw simulation attempts to Excel.

## Repository files

```text
streamlit_app.py
socrates_core.py
requirements.txt
logo.png                       # optional; copy from the Centaur repository
.streamlit/config.toml
tests/test_socrates_core.py
```

The app runs without `logo.png`; it displays the logo only when that file exists.

## Deploy on Streamlit Community Cloud

Use these values:

- Repository: `konradbertram/socrates-experiment-runner`
- Branch: `main`
- Main file path: `streamlit_app.py`
- App URL: optional, for example `socrates-experiment-runner`

After deployment, open the app settings and add this secret:

```toml
FEATHERLESS_API_KEY = "paste-your-key-here"
```

Never commit the API key to GitHub.

## Git upload workflow

From a GitHub Codespace or local clone:

```bash
git add streamlit_app.py socrates_core.py requirements.txt .streamlit/config.toml README.md tests/test_socrates_core.py logo.png
git commit -m "Add Socrates Experiment Runner"
git pull --rebase origin main
git push
git status
```

Omit `logo.png` from `git add` when you have not copied it into the repository.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FEATHERLESS_API_KEY="your-key"
streamlit run streamlit_app.py
```

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Methodological notes

- The representative U.S. preset consists of rounded marginal distributions. It does not preserve joint correlations between demographics.
- The app sends one individual Featherless query per planned respondent slot.
- A failed or invalid respondent slot is retried up to three times with the same profile, assigned variant, and response-code mapping.
- Results are stochastic Socrates simulations for hypothesis screening and experiment design, not human observations or proof of a real-world treatment effect.
- RAG and historical-experiment calibration are intentionally outside this MVP.
