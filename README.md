# Intelligent News Analyzer

An NLP-based application for analyzing news articles and presenting insights through an interactive Streamlit dashboard. The intended pipeline accepts news text or PDF files, processes the text, and runs analysis such as categorization, fake-news prediction, summarization, keyword extraction, named entity recognition, sentiment analysis, semantic similarity, and explainability.

> **Important:** Model names below reflect the proposed architecture. Verify exact model checkpoints and execution paths in your source code before describing them as implemented. A fake-news prediction or credibility score is an indicator, not proof that an article is true or false.

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Modules and Models](#modules-and-models)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [How to Use](#how-to-use)
- [Credibility Score](#credibility-score)
- [Database](#database)
- [Language Support](#language-support)
- [Testing](#testing)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [License and Model Notes](#license-and-model-notes)

## Project Overview
Online news is published in large volumes and may contain factual claims, conflicting statements, emotionally charged language, or information that needs verification. The Intelligent News Analyzer aims to help users examine news by extracting key information, classifying topics, summarizing articles, analyzing sentiment, identifying entities, and comparing articles. Where implemented, it may also provide fake-news predictions, claim analysis, and model explanations.

The application is a decision-support tool. It does not replace professional journalism, fact-checking, or verification against reliable primary sources.

## Objectives
- Accept news text and/or PDF files.
- Extract and preprocess text for analysis.
- Predict the likely news category.
- Summarize long articles.
- Extract keywords and named entities.
- Analyze sentiment.
- Predict whether an article resembles the classes learned by a fake-news classifier, if configured.
- Compare articles for semantic similarity.
- Explain selected predictions where explainability is supported.
- Display results and optionally save analysis history.

## Features

| Feature | Description |
|---|---|
| Text/PDF input | Accepts pasted text or uploaded PDF files, depending on enabled input handlers. |
| PDF text extraction | Extracts machine-readable text. Scanned PDFs may require OCR. |
| Preprocessing | Cleans and prepares text for downstream NLP tasks. |
| News categorization | Predicts a topic using the configured classifier. |
| Fake-news prediction | Estimates whether text resembles classes learned from training data. |
| Summarization | Produces a shorter article summary when a summarization model is configured. |
| Keyword extraction | Identifies important words and phrases. |
| Named entity recognition (NER) | Identifies people, organizations, locations, dates, and other entities. |
| Sentiment analysis | Estimates positive, negative, or neutral sentiment, depending on model labels. |
| Semantic similarity | Compares article embeddings to find related content. |
| Explainable AI | Uses LIME, if integrated, to explain an individual prediction locally. |
| Credibility indicator | Combines selected indicators according to the implemented scoring rules. |
| Dashboard | Displays results through Streamlit. |
| History and user data | Can persist records when database operations are enabled. |

## System Architecture

```text
User Input (Text / PDF Upload)
              |
              v
PDF Text Extraction / Preprocessing
              |
              v
        NLP Processing
              |
      +-------+--------+
      |       |        |
      v       v        v
 News      Fake-News  Summarization
Category   Prediction
      |       |        |
      +-------+--------+
              |
      +-------+--------+
      |       |        |
      v       v        v
 Keywords   Named     Sentiment
            Entities  Analysis
      |       |        |
      +-------+--------+
              |
              v
       Semantic Similarity
              |
              v
       Explainable AI (LIME)
              |
              v
      Streamlit Dashboard
              |
              v
       SQLite Database
       (History / User Data)
```

This is a simplified conceptual diagram. Some analysis modules can run independently or in parallel; the actual execution order depends on the source code.

## Modules and Models

The model names below come from the proposed architecture. Confirm exact checkpoints in the source before claiming that a specific model is active.

| Module | Technology/model shown | Role |
|---|---|---|
| User interface | Streamlit | Upload controls, results, and dashboard pages. |
| PDF extraction | PyMuPDF (`fitz`) | Extracts text from text-based PDFs. |
| Text preprocessing | Python, regular expressions; other NLP libraries if configured | Normalizes and prepares text. |
| News categorization | DistilBERT | Predicts a category when a suitable classification checkpoint is configured. |
| Fake-news detection | TF-IDF + Logistic Regression | Uses text features and a trained classifier. |
| Summarization | BART | Generates a summary when a suitable checkpoint is configured. |
| Keyword extraction | YAKE | Ranks candidate keywords using statistical text features. |
| Named entity recognition | BERT-based NER model | Identifies entities when an NER checkpoint is configured. |
| Sentiment analysis | DistilBERT-based sentiment model | Predicts sentiment when a suitable checkpoint is configured. |
| Semantic similarity | Sentence Transformers + cosine similarity | Compares text embeddings and ranks similar articles. |
| Explainability | LIME | Provides a local explanation for a selected prediction. |
| Persistence | SQLite | Stores records when database functions are implemented. |
| Dashboard | Streamlit and configured charting libraries | Displays results and statistics. |

### Abbreviations
- **AI** — Artificial Intelligence
- **NLP** — Natural Language Processing
- **PDF** — Portable Document Format
- **TF-IDF** — Term Frequency–Inverse Document Frequency
- **BERT** — Bidirectional Encoder Representations from Transformers
- **DistilBERT** — A distilled, smaller version of BERT
- **BART** — Bidirectional and Auto-Regressive Transformers
- **NER** — Named Entity Recognition
- **LIME** — Local Interpretable Model-agnostic Explanations
- **YAKE** — Yet Another Keyword Extractor

## Technology Stack
Technologies used or explored in the project context include:
- **Language:** Python
- **Web application:** Streamlit
- **PDF extraction:** PyMuPDF
- **Keyword extraction:** YAKE
- **Translation experiments:** Argos Translate and Hugging Face NLLB-200
- **Machine learning/NLP:** scikit-learn and/or Hugging Face Transformers, depending on the active module
- **Deep-learning runtime:** PyTorch, where required
- **Tokenization support:** SentencePiece, where required
- **Explainability:** LIME, if enabled
- **Database:** SQLite

Not every library listed is necessarily used by every version of the app. Check `requirements.txt`, imports, and model-loading statements in your current source.

## Project Structure
The exact layout may differ in your repository. A possible organization is:

```text
intelligent-news-analyzer/
├── app.py                 # Streamlit application entry point
├── utils.py               # Shared text-processing/analysis helpers
├── database.py            # Database functions, if separated
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── models/                # Local model files, if used
├── data/                  # Optional sample data
└── news_analyzer.db       # SQLite database, if created
```

Use filenames that actually exist in your project; this structure is illustrative.

## Installation

### Prerequisites
- Python and `pip`
- Git, if cloning from a repository
- Disk space and internet access if pretrained models need to be downloaded
- OCR software such as Tesseract only if scanned-document OCR is used

### 1. Get the project
```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```
Alternatively, open your existing project folder.

### 2. Create and activate a virtual environment

**Windows Command Prompt**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
If `requirements.txt` exists:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not have a complete requirements file, install only packages required by the actual imports. Common packages in this project context include:
```bash
python -m pip install streamlit pymupdf yake scikit-learn
```
Install additional packages such as `transformers`, `torch`, `sentence-transformers`, `lime`, `pytesseract`, `argostranslate`, or `sentencepiece` only if required by the active code. Model downloads can be large.

## Run the Application
From the project root, activate the virtual environment and run:
```bash
python -m streamlit run app.py
```
Open the local URL printed in the terminal, usually `http://localhost:8501`. If your entry-point file has another name, use that filename.

To check syntax for existing files:
```bash
python -m py_compile app.py utils.py
```

## How to Use
1. Start the Streamlit application.
2. Sign in or register if authentication is enabled.
3. Open the news-analysis page.
4. Paste news text or upload a supported PDF.
5. Start the analysis.
6. Review the outputs enabled in your version, such as category, summary, keywords, entities, sentiment, similarity, and predictions.
7. Review explanations and credibility indicators carefully.
8. Save or revisit the analysis if history is implemented.

## Credibility Score
The dashboard may display a score from 0 to 100 and a label such as **Low**, **Medium**, or **High**. The exact formula and thresholds must be documented from the application's scoring implementation. If the score combines a fake-news prediction and sentiment indicators, document the precise weights, transformations, and threshold boundaries used in code.

Sentiment is not evidence of truth: a positive article can be false, and a negative article can be accurate. A robust credibility assessment should consider source reliability, verifiable claims, corroborating sources, publication context, and uncertainty. Do not call the score a probability of truth unless it has been designed and calibrated for that interpretation.

## Database
The architecture uses SQLite for local persistence. Depending on the actual schema, it may store:
- User/account records
- Article text or metadata
- Analysis results
- Category, sentiment, and prediction outputs
- Analysis timestamps and user history

The database initialization code defines the real tables and fields. Review it before documenting the schema. Store passwords using a secure password-hashing method; never store plaintext passwords.

## Language Support
The project has explored multilingual processing and translation. NLLB-200 (`facebook/nllb-200-distilled-600M`) was tested for Telugu-to-English translation in the development context. Argos Translate was also explored, but available language pairs depend on installed packages.

Language support depends on the translation packages, model checkpoints, tokenizer configuration, and downstream pipeline. Do not assume every analysis model supports every language directly. Translation errors can affect subsequent analysis.

## Testing
Use varied sample articles to evaluate each module:
1. **Strong claims:** Include quantities, percentages, dates, and predictions.
2. **Contradictions:** Include two statements that cannot both be true in the same context.
3. **Positive sentiment:** Include praise or successful outcomes.
4. **Negative sentiment:** Include complaints, losses, or service failures.
5. **Neutral reporting:** Include factual descriptions without strong emotion.
6. **Multilingual input:** Test each language the app claims to support.
7. **Similar news:** Compare articles about the same event with different wording.
8. **PDF input:** Test text-based PDFs and scanned PDFs if OCR is supported.
9. **Credibility indicator:** Verify the score calculation and its displayed explanation.

For fair evaluation, use labeled test data, record model versions, and measure precision, recall, F1-score, and confusion matrices for classification tasks. High performance on synthetic or limited data does not establish real-world accuracy.

## Limitations and Responsible Use
- Model outputs can be incorrect, incomplete, or biased.
- A fake-news classifier identifies patterns learned from training data; it does not independently verify facts.
- Summaries can omit qualifications or introduce errors.
- NER can miss entities or assign incorrect labels.
- Sentiment models may misinterpret sarcasm, quotations, and multilingual text.
- Similarity scores indicate similarity, not necessarily that articles describe the same event.
- LIME explanations are local approximations, not proof of causal reasoning.
- Translation may alter names, idioms, or specialized meaning.
- Scanned PDFs require OCR, and OCR errors can affect all downstream modules.
- Credibility levels should be treated as indicators unless validated against suitable ground truth.

Verify important claims using reliable primary sources and independent reporting.

## Troubleshooting

### `ModuleNotFoundError`
Activate the intended virtual environment and install the missing package into that environment:
```bash
python -m pip install <package-name>
```

### Streamlit command not found
Run Streamlit through Python:
```bash
python -m streamlit run app.py
```

### PDF text is empty
The PDF may contain scanned images rather than selectable text. If OCR is supported, confirm that the OCR engine is installed and configured.

### Model download or memory errors
Check the model identifier, internet access for initial downloads, available disk space, RAM, and whether the selected model is appropriate for your machine.

### Unexpected category or sentiment
Check the model checkpoint, label mapping, preprocessing, language, and training-data domain. Evaluate against labeled examples instead of relying on a single article.

### Credibility score seems incorrect
Inspect the scoring function, component weights, thresholds, and treatment of missing indicators. Ensure the UI description matches the calculation.

## Future Enhancements
- Evidence-based claim verification using trusted sources.
- Contradiction detection using a validated Natural Language Inference model.
- Improved multilingual analysis and language-specific evaluation.
- Source citations for claims requiring verification.
- Calibrated credibility indicators with transparent component scores.
- Graphs for connections among articles, events, people, organizations, and topics.
- Duplicate-news detection across publishers.
- Model evaluation reports using documented datasets and metrics.
- Better PDF OCR quality checks.
- Secure authentication, role-based access, and database backup/retention controls.

These are possible future enhancements, not claims that every feature is currently implemented.

## License and Model Notes
Choose a license appropriate for your own source code and dependencies. Pretrained models and datasets may have separate licenses and restrictions. Review the license and model card before redistributing or using them commercially.

In particular, the NLLB-200 distilled model explored during development has a non-commercial, research-oriented license. Check the current model card and terms before production or commercial use.

---

## Project Status
This README describes the intended architecture and known development context. Confirm active modules, model checkpoints, database schema, supported languages, and the scoring formula against the current source code before submitting it as a final implementation report.
