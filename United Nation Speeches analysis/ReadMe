# The Sound of Power: An NLP Analysis of United Nations Security Council Speeches on Crisis Management

An NLP research project analyzing how member states of the United Nations Security Council use language to justify their positions on crisis and conflict management — focusing on debates surrounding **sanctions**, **humanitarian aid**, and **ceasefires**.

> **Authors:** Delvin Marimo, Pamela Quartson, Massa Coulibaly — UC Berkeley

---

## Overview

While prior literature has focused on *what* topics emerge in UNSC discourse, this project investigates *how* countries strategically use rhetoric to argue their positions. We apply a multi-method computational framework to extract over **450 linguistic features per speech** from **2,069 meeting transcripts** spanning 1950–2025.

**Key Finding:** Linguistic variability in pre-vote speeches is driven primarily by **resolution type** rather than country identity or bloc membership. Speeches cluster around meeting topics, indicating that UNSC diplomatic discourse is highly regimented and formulaic — shaped by institutional conventions regardless of geopolitical orientation.

## Methodology

The framework integrates five complementary analytical components operating at different linguistic levels (lexical, syntactic-proxy, thematic, and semantic):

1. Stance Detection — Rule-based lexical scoring on a 5-point ordinal scale to classify support vs. opposition
2. Rhetorical Strategy Analysis — Normalized keyword/phrase frequencies across 10 persuasive strategies
3. Topic Modeling — Dual approach using LDA (5 topics) and BERTopic to discover latent thematic structure
4. Framing Analysis — Hybrid lexical-intensity scoring across 10 interpretive frames
5. Transformer Embeddings — 384-dim vectors via SentenceTransformers for semantic similarity, clustering, and UMAP/t-SNE visualization

## Dataset

The dataset consists of **2,069 verbatim meeting transcripts** (S/PV records) downloaded from the [United Nations Digital Library](https://digitallibrary.un.org/):

- **Humanitarian Aid:** 601 records
- **Sanctions:** 942 records
- **Ceasefire:** 526 records

Each PDF transcript is parsed to extract individual country speeches, producing a structured DataFrame of speaker-level records.

> **Note:** The raw PDF datasets are not included in this repository due to size constraints. The data was sourced from the UN Digital Library and can be reconstructed by downloading S/PV meeting records for the three topic categories above.

## Project Structure

```
├── README.md
├── The_Sound_of_Power_...pdf          # Research paper
├── UN_Crisis_Speeches_Analysis_proj.ipynb  # Full analysis notebook
└── data/                               # (not included — see Dataset section due to space concerns)
    ├── humanitarian_assistance/
    ├── sanctions/
    └── ceasefire/
```

## Setup & Installation

The notebook was developed in **Google Colab**. To run locally:

```bash
pip install pdfplumber bertopic sentence-transformers transformers
pip install pandas numpy matplotlib seaborn scikit-learn
pip install nltk torch umap-learn
```

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## Key Results

- **Rhetorical Strategy:** Appeal to Authority (94.5%), Assertive Language (93.7%), and Dialogue Emphasis (93.6%) are the most prevalent strategies. Both Western/Non-Western and P5/non-P5 groupings show similar usage patterns.
- **Topic Modeling:** Five distinct topics emerged — Sovereignty & Non-Interference, Humanitarian Crisis & Protection, International Law & Charter Violations, Peaceful Resolution & Dialogue, and Security Threats & Sanctions. High topic-assignment confidence indicates clear thematic separation.
- **Semantic Similarity:** Cosine similarity centered around 0.42, with brighter bands for humanitarian assistance indicating more standardized rhetorical templates.
- **Embeddings Clustering:** Speech embeddings cluster by **meeting type** (ceasefire, humanitarian, sanctions) far more strongly than by country or geopolitical bloc.

## Citation

```bibtex
@article{marimo2025soundofpower,
  title={The Sound of Power: An NLP Analysis of United Nations Security Council Speeches on Crisis Management},
  author={Marimo, Delvin and Quartson, Pamela and Coulibaly, Massa},
  year={2025},
  institution={UC Berkeley}
}
```

## License

This project is for academic and research purposes. Please cite the paper if you use this work.
