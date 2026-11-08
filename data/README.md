# Data

The data files are **not** committed. `.gitignore` excludes every `data/*.csv` file. Regenerate them with the scripts below.

| File | Made by | Contents |
|---|---|---|
| `raw_tinystories.csv` | `python data/download_tinystories.py` | 15,000 raw stories (5,000 each of Dialogue, MoralValue and Conflict) with their metadata and the untouched raw text |
| `cleaned_tinystories.csv` | `python src/clean_pipeline.py` | The cleaned text and lemmas used by Phases 2-4 |

**Source:** [roneneldan/TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) on Hugging Face. The script streams the file `TinyStories-Instruct-train.txt` over HTTP and reads only the first ~180,000 stories, so the full 2.6 GB file is never stored on disk.

**Licence:** CDLA-Sharing-1.0, by Ronen Eldan and Yuanzhi Li (Microsoft Research).

**Citation:** Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
