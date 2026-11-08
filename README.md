# nlp-project-tinyGpt

Semester project for **Introduction to NLP**. We clean and analyse a corpus of children's short stories, classify each story's narrative type, and train a small GPT-style model from scratch that writes new stories.

## Team

| Name | Roll number |
|---|---|
| Shamaas Munir | L1F23BSCS0640 |
| Hassan Shahzad | L1F23BSCS0169 |
| Hassan Nasarullah | L1F23BSCS0128 |

## Dataset

- **Name:** [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) (Eldan & Li, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759))
- **Licence:** CDLA-Sharing-1.0
- **Size:** a 15,000-story sample (5,000 per class, about 2.6M words) from the 2.6 GB corpus
- **Target:** `narrative_type`, one of `Dialogue`, `MoralValue` or `Conflict`, taken from each story's `Features:` tag

Raw data is not committed. Run the download script below to regenerate it.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/shamaas-munir/nlp-project-tinyGpt.git
cd nlp-project-tinyGpt
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python data/download_tinystories.py   # -> data/raw_tinystories.csv
python src/clean_pipeline.py          # -> data/cleaned_tinystories.csv
jupyter notebook notebooks/01_phase1_cleaning_eda.ipynb
```

## Project structure

```text
data/        download script (CSV files are git-ignored)
src/         cleaning pipeline and later model code
notebooks/   exploration and analysis notebooks
figures/     saved plots
reports/     phase reports
```

## Phases

- [ ] **Phase 1:** data acquisition and cleaning (`assignment-1`)
- [ ] **Phase 2:** BoW, TF-IDF, Word2Vec and fastText (`assignment-2`)
- [ ] **Phase 3:** classical ML classifiers
- [ ] **Phase 4:** Story-GPT, a Transformer language model

## Acknowledgements

Parts of the code and explanations were developed with help from Claude (Anthropic). All code was run and verified by the team.

## Licence

MIT. The dataset keeps its own CDLA-Sharing-1.0 licence.