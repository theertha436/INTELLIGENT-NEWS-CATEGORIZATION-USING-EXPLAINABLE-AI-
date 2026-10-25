import re

import numpy as np

import pandas as pd

import yake

import joblib

from pypdf import PdfReader

from sklearn.metrics.pairwise import cosine_similarity

from lime.lime_text import LimeTextExplainer

from langdetect import detect

from deep_translator import GoogleTranslator, MyMemoryTranslator

import time
from functools import lru_cache

# Optional local/offline translator. Install with: python -m pip install argostranslate

try:

    import argostranslate.translate as argos_translate

except ImportError:

    argos_translate = None

import spacy

from transformers import (

    pipeline,

    AutoTokenizer,

    AutoModel,

    AutoModelForSequenceClassification

)

from sentence_transformers import SentenceTransformer

# --------------------------------------------------

# 1. LOAD MODELS

# --------------------------------------------------

print("Loading NLP models...")

# News classification

classifier = pipeline(

    "text-classification",

    model="textattack/distilbert-base-uncased-ag-news",

    top_k=None

)

# Sentiment

sentiment_model = pipeline(

    "sentiment-analysis",

    model="distilbert-base-uncased-finetuned-sst-2-english"

)

# Named Entity Recognition

ner_model = pipeline(

    "ner",

    model="dslim/bert-base-NER",

    aggregation_strategy="simple"

)

# Summarization

summarizer = pipeline(

    "summarization",

    model="facebook/bart-large-cnn"

)

# Semantic embeddings

embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)

# spaCy

nlp = spacy.load("en_core_web_sm")

# Contradiction / Natural Language Inference model.

# Used to compare pairs of statements as entailment,

# contradiction, or neutral.

contradiction_model = pipeline(

    "text-classification",

    model="cross-encoder/nli-deberta-v3-small",

    top_k=None

)

print("Models loaded successfully!")

fake_news_model = joblib.load(

    "models/fake_news_model.pkl"

)

fake_news_vectorizer = joblib.load(

    "models/fake_news_vectorizer.pkl"

)

# --------------------------------------------------

# 2. PDF TEXT EXTRACTION

# --------------------------------------------------

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text

# --------------------------------------------------

# 3. TEXT PREPROCESSING

# --------------------------------------------------

def preprocess_text(text):

    # Remove extra spaces

    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary characters

    text = re.sub(r"[^\w\s.,!?'-]", "", text)

    return text.strip()

# --------------------------------------------------

# 4. NEWS CATEGORY

# --------------------------------------------------

# --------------------------------------------------

# 5. KEYWORD EXTRACTION

# --------------------------------------------------

def extract_keywords(text, number=10):

    kw_extractor = yake.KeywordExtractor(

        lan="en",

        n=2,

        dedupLim=0.9,

        top=number

    )

    keywords = kw_extractor.extract_keywords(text)

    return [

        keyword

        for keyword, score in keywords

    ]

# --------------------------------------------------

# 6. NAMED ENTITY RECOGNITION

# --------------------------------------------------

def extract_entities(text):

    entities = ner_model(text[:4000])

    result = []

    for entity in entities:

        result.append({

            "text": entity["word"],

            "type": entity["entity_group"],

            "score": round(entity["score"], 3)

        })

    return result

# --------------------------------------------------

# 7. SENTIMENT ANALYSIS

# --------------------------------------------------

def analyze_sentiment(text):

    result = sentiment_model(

        text[:4000]

    )[0]

    return {

        "sentiment": result["label"],

        "score": result["score"]

    }

# --------------------------------------------------

# 8. TEXT SUMMARIZATION

# --------------------------------------------------

def summarize_text(text):

    # BART has input limitations.

    # We restrict the text for the first version.

    text = text[:5000]

    if len(text.split()) < 50:

        return text

    summary = summarizer(

        text,

        max_length=120,

        min_length=30,

        do_sample=False

    )

    return summary[0]["summary_text"]

# --------------------------------------------------

# 9. CREATE EMBEDDING

# --------------------------------------------------

def create_embedding(text):

    embedding = embedding_model.encode(

        text,

        normalize_embeddings=True

    )

    return embedding

# --------------------------------------------------

# 10. FIND SIMILAR NEWS

# --------------------------------------------------

def find_similar_news(

    query,

    news_dataframe,

    top_n=5

):

    query_embedding = create_embedding(query)

    article_embeddings = embedding_model.encode(

        news_dataframe["text"].tolist(),

        normalize_embeddings=True

    )

    similarities = cosine_similarity(

        [query_embedding],

        article_embeddings

    )[0]

    news_dataframe = news_dataframe.copy()

    news_dataframe["similarity"] = similarities

    result = news_dataframe.sort_values(

        "similarity",

        ascending=False

    ).head(top_n)

    return result

# --------------------------------------------------

# EXPLAINABLE AI - LIME

# --------------------------------------------------

# --------------------------------------------------

# EXPLAINABLE AI - LIME

# --------------------------------------------------

# ============================================================

# DOMAIN-AWARE NEWS CLASSIFICATION
# The AG News model has only World, Sports, Business and Science/Technology.
# These transparent keyword signals add Environment, Education and Agriculture.

ENVIRONMENT_TERMS = {
    "environment": 1.5, "environmental": 1.5, "climate": 1.5,
    "climate change": 2.0, "global warming": 2.0, "pollution": 1.8,
    "air pollution": 2.0, "water pollution": 2.0, "soil pollution": 1.8,
    "forest": 1.0, "forests": 1.0, "deforestation": 2.0,
    "tree planting": 2.5, "planting trees": 2.5, "tree plantation": 2.5,
    "trees": 0.8, "wildlife": 1.5, "biodiversity": 2.0,
    "ecosystem": 1.5, "ecosystems": 1.5, "green energy": 1.8,
    "renewable energy": 2.0, "solar energy": 1.0, "wind energy": 1.0,
    "carbon emissions": 2.0, "emissions": 1.0, "recycling": 1.5,
    "plastic waste": 2.0, "conservation": 1.8, "drought": 1.3,
    "flooding": 1.0, "heatwave": 1.5, "greenhouse gases": 2.0,
    "sustainability": 1.5, "clean energy": 1.5, "natural resources": 1.5,
    "reforestation": 2.2, "afforestation": 2.2, "environmental protection": 2.5,
}

EDUCATION_TERMS = {
    "education": 2.0, "educational": 1.8, "school": 1.0, "schools": 1.0,
    "student": 1.0, "students": 1.0, "teacher": 1.0, "teachers": 1.0,
    "university": 1.2, "universities": 1.2, "college": 1.0, "colleges": 1.0,
    "digital education": 2.8, "online learning": 2.8, "e-learning": 2.5,
    "learning resources": 2.0, "educational resources": 2.3,
    "rural students": 2.0, "curriculum": 1.2, "scholarship": 1.5,
    "scholarships": 1.5, "literacy": 1.5, "classroom": 1.0,
    "teaching": 1.0, "exam": 0.8, "examination": 0.8,
}

AGRICULTURE_TERMS = {
    "agriculture": 2.0, "agricultural": 2.0, "farming": 1.8,
    "farmer": 1.5, "farmers": 1.5, "farm": 1.0, "farms": 1.0,
    "crop": 1.5, "crops": 1.5, "crop yield": 2.5, "crop yields": 2.5,
    "harvest": 1.3, "harvesting": 1.3, "irrigation": 1.5,
    "fertilizer": 1.2, "fertiliser": 1.2, "seeds": 1.0,
    "livestock": 1.2, "dairy farming": 1.5, "agriculture scheme": 2.5,
    "agricultural scheme": 2.5, "agriculture policy": 2.0,
    "farmers scheme": 2.0, "rural farming": 1.8, "soil health": 1.5,
    "pesticide": 1.0, "monsoon": 0.8, "minimum support price": 2.0,
}

HEALTH_TERMS = {
    "health": 1.2, "hospital": 1.8, "hospitals": 1.8,
    "doctor": 1.2, "doctors": 1.2, "patient": 1.2, "patients": 1.2,
    "medical": 1.8, "medicine": 1.5, "healthcare": 2.2,
    "health care": 2.2, "disease": 1.5, "diseases": 1.5,
    "diagnosis": 1.5, "treatment": 1.2, "vaccine": 1.8,
    "vaccination": 1.8, "clinical trial": 2.0, "screening program": 2.5,
    "health screening": 2.5, "diabetes": 1.8, "blood pressure": 1.5,
    "mental health": 2.0, "public health": 2.2, "outbreak": 1.5,
    "epidemic": 1.8, "pandemic": 1.8, "hospital launched": 1.5,
    "free screenings": 1.8,
}

GOVERNMENT_TERMS = {
    "government": 1.8, "minister": 1.5, "ministry": 1.5,
    "parliament": 1.8, "election": 1.8, "elections": 1.8,
    "political party": 2.0, "political parties": 2.0,
    "policy": 1.2, "public policy": 1.8, "state government": 2.0,
    "announced a scheme": 1.5, "government scheme": 2.2,
    "funding": 1.0, "allocated": 1.0, "budget": 1.5,
    "roads": 0.8, "road repair": 1.5, "rural development": 1.8,
    "villages": 0.8, "public works": 1.8, "approved": 0.8,
    "official statement": 1.5, "government announced": 2.0,
}

DOMAIN_TERM_SETS = {
    "Environment": ENVIRONMENT_TERMS,
    "Education": EDUCATION_TERMS,
    "Agriculture": AGRICULTURE_TERMS,
    "Health": HEALTH_TERMS,
    "Politics/Government": GOVERNMENT_TERMS,
}


def calculate_domain_score(text, terms):
    """Return a transparent 0-100 domain score based on matching terms."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lower_text = re.sub(r"\s+", " ", text.lower())
    matched = []
    total = 0.0
    for term, weight in sorted(terms.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", lower_text):
            matched.append(term)
            total += weight
    if len(matched) >= 2:
        total += 1.5
    if len(matched) >= 4:
        total += 2.0
    return round(min(100.0, total * 10.0), 2)


def calculate_environment_score(text):
    """Compatibility wrapper retained for existing code that uses this function."""
    return calculate_domain_score(text, ENVIRONMENT_TERMS)


def _category_index_from_label(label):
    """Map Hugging Face labels to the four AG News class positions."""
    value = str(label).strip().upper().replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    label_id = re.fullmatch(r"LABEL\s*(\d+)", value)
    if label_id:
        idx = int(label_id.group(1))
        return idx if idx in range(4) else None
    if value in {"WORLD", "WORLD NEWS", "POLITICS", "INTERNATIONAL"}:
        return 0
    if value in {"SPORT", "SPORTS"}:
        return 1
    if value in {"BUSINESS", "ECONOMY", "FINANCE"}:
        return 2
    if value in {"SCI TECH", "SCI/TECH", "SCIENCE TECH", "SCIENCE/TECHNOLOGY", "SCIENCE TECHNOLOGY", "TECH", "TECHNOLOGY", "SCIENCE AND TECHNOLOGY", "SCIENCE & TECHNOLOGY"}:
        return 3
    return None


def _normalise_pipeline_results(raw):
    """Normalize single-text and batched text-classification output."""
    if not raw:
        return []
    if isinstance(raw, dict):
        return [[raw]]
    if isinstance(raw[0], dict):
        return [raw]
    normalized = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append([item])
        elif isinstance(item, list):
            normalized.append(item[0] if item and isinstance(item[0], list) else item)
        else:
            normalized.append([])
    return normalized


def _scores_from_results(results):
    scores = [0.0, 0.0, 0.0, 0.0]
    for result in results:
        if not isinstance(result, dict):
            continue
        idx = _category_index_from_label(result.get("label", ""))
        if idx is not None:
            scores[idx] = float(result.get("score", 0.0))
    total = sum(scores)
    if total <= 0:
        raise RuntimeError(
            "Could not map the news model's output labels to AG News categories. "
            f"Model labels received: {[r.get('label') for r in results if isinstance(r, dict)]}"
        )
    return [score / total for score in scores]


def _get_base_category_scores(text):
    """Return base scores ordered World, Sports, Business, Science/Technology."""
    raw = classifier(str(text), top_k=None, truncation=True)
    normalized = _normalise_pipeline_results(raw)
    if not normalized:
        raise RuntimeError("The news classifier returned no predictions.")
    return _scores_from_results(normalized[0])


def _classification_probabilities(text, base_scores=None):
    """Combine AG News scores with transparent domain-specific signals."""
    if base_scores is None:
        base_scores = _get_base_category_scores(text)
    base_scores = list(base_scores)
    domain_scores = {
        category: calculate_domain_score(text, terms)
        for category, terms in DOMAIN_TERM_SETS.items()
    }
    domain, score = max(domain_scores.items(), key=lambda item: item[1])

    # A dedicated domain is selected only when several relevant terms match.
    # Strong topic evidence then takes priority over the four broad AG News labels.
    if score >= 25.0:
        domain_weight = min(0.96, max(0.72, score / 100.0))
        base_total = sum(base_scores) or 1.0
        base_scores = [
            (value / base_total) * (1.0 - domain_weight)
            for value in base_scores
        ]
        domain_scores = {name: 0.0 for name in DOMAIN_TERM_SETS}
        domain_scores[domain] = domain_weight
    else:
        domain_scores = {name: 0.0 for name in DOMAIN_TERM_SETS}

    return base_scores + [domain_scores[name] for name in DOMAIN_TERM_SETS]


def classify_news(text):
    """Classify news into AG News labels plus domain-aware categories."""
    if not isinstance(text, str) or not text.strip():
        return []
    scores = _classification_probabilities(text[:4000])
    label_names = [
        "World", "Sports", "Business", "Science/Technology",
        *DOMAIN_TERM_SETS.keys(),
    ]
    results = [
        {"category": category, "score": float(score)}
        for category, score in zip(label_names, scores)
    ]
    results.sort(key=lambda item: item["score"], reverse=True)
    return results

# Replace the previous LIME function so it supports the

# fifth Environment category.

# ------------------------------------------------------------

def explain_prediction(text, num_samples=80):

    """Return input-specific feature weights for the predicted news category.

    First attempts LIME. If LIME fails for a library/model compatibility reason,

    falls back to batched word-ablation importance using the same classifier.

    Returned weights are signed: positive supports the predicted class, negative

    weakens it. These are model explanations, not causal proof.

    """

    if not text or not str(text).strip():

        return []

    text = re.sub(r"\s+", " ", str(text)).strip()

    lime_text = text[:1600]

    words = re.findall(r"\b[\w'-]+\b", lime_text)

    if len(words) < 3:

        return []

    label_names = ["World", "Sports", "Business", "Science/Technology", *DOMAIN_TERM_SETS.keys()]

    def probabilities_for(texts):

        inputs = [str(t).strip()[:1600] or "news" for t in texts]

        raw = classifier(inputs, top_k=None, batch_size=8, truncation=True)

        batches = _normalise_pipeline_results(raw)

        rows = []

        for item, results in zip(inputs, batches):

            base = _scores_from_results(results)
            combined = _classification_probabilities(item, base)
            total = sum(combined)
            rows.append([p / total for p in combined] if total > 0 else [1.0 / (4 + len(DOMAIN_TERM_SETS))] * (4 + len(DOMAIN_TERM_SETS)))

        if len(rows) != len(inputs):

            raise RuntimeError(f"Classifier returned {len(rows)} rows for {len(inputs)} texts")

        return np.asarray(rows, dtype=np.float64)

    try:

        current = classify_news(lime_text)

        if not current:

            return []

        category = current[0]["category"]

        category_index = label_names.index(category)

        explainer = LimeTextExplainer(class_names=label_names, random_state=17, bow=True)

        try:

            explanation = explainer.explain_instance(

                lime_text,

                probabilities_for,

                labels=(category_index,),

                num_features=min(10, max(5, len(words))),

                num_samples=max(40, min(int(num_samples), 100))

            )

            weights = explanation.as_list(label=category_index)

            if weights:

                return [(str(word), float(weight)) for word, weight in weights]

        except Exception as lime_exc:

            print(f"LIME explanation error: {type(lime_exc).__name__}: {lime_exc}")

        # Reliable fallback: compare target-category score with each token removed.

        baseline = float(probabilities_for([lime_text])[0][category_index])

        unique_words = []

        seen = set()

        for word in words:

            key = word.lower()

            if len(key) > 2 and key not in seen:

                seen.add(key)

                unique_words.append(word)

        unique_words = unique_words[:18]

        variants = []

        for target in unique_words:

            pattern = re.compile(r"(?<!\w)" + re.escape(target) + r"(?!\w)", re.IGNORECASE)

            changed = pattern.sub(" ", lime_text, count=1)

            variants.append(re.sub(r"\s+", " ", changed).strip() or "news")

        if not variants:

            return []

        variant_scores = probabilities_for(variants)

        weights = [(word, baseline - float(row[category_index])) for word, row in zip(unique_words, variant_scores)]

        weights.sort(key=lambda pair: abs(pair[1]), reverse=True)

        return weights[:10]

    except Exception as exc:

        print(f"Explainable AI error: {type(exc).__name__}: {exc}")

        return []

# ==========================================

# FAKE NEWS MODEL

# ==========================================

def detect_fake_news(text):

    """Classify an article using the fitted TF-IDF fake-news estimator.

    The meaning of each numeric prediction is derived from model.classes_,

    rather than assuming that class 0 always means real and class 1 always fake.

    """

    if not text or not str(text).strip():

        return "UNKNOWN", 0.0

    vector = fake_news_vectorizer.transform([str(text)])

    predicted = fake_news_model.predict(vector)[0]

    probabilities = fake_news_model.predict_proba(vector)[0]

    classes = list(getattr(fake_news_model, "classes_", []))

    # Infer semantics from textual labels when possible. For numeric 0/1 labels,

    # preserve the project's original training convention: 0=real, 1=fake.

    label_text = str(predicted).strip().lower()

    if any(word in label_text for word in ("fake", "false", "unreliable")):

        result = "POTENTIALLY FAKE"

    elif any(word in label_text for word in ("real", "true", "reliable")):

        result = "LIKELY REAL"

    elif predicted in (0, 0.0):

        result = "LIKELY REAL"

    elif predicted in (1, 1.0):

        result = "POTENTIALLY FAKE"

    else:

        result = "UNKNOWN"

    predicted_index = classes.index(predicted) if classes and predicted in classes else int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_index])

    return result, confidence

def detect_language(text):

    """

    Detect the language of the input text.

    """

    if not text or not text.strip():

        return "unknown"

    try:

        language = detect(text)

        language_names = {

            "en": "English",

            "ta": "Tamil",

            "hi": "Hindi",

            "te": "Telugu",

            "ml": "Malayalam",

            "kn": "Kannada",

            "bn": "Bengali",

            "mr": "Marathi"

        }

        return language_names.get(

            language,

            language

        )

    except Exception:

        return "unknown"

# Translation cache for repeated articles during the current app session.

# Translation cache for successful translations during the current app session.
_translation_cache = {}

# NLLB supports these language codes. The model is loaded only when required.
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_LANGUAGE_CODES = {
    "ml": "mal_Mlym",  # Malayalam
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "hi": "hin_Deva",  # Hindi
    "bn": "ben_Beng",  # Bengali
    "kn": "kan_Knda",  # Kannada
    "mr": "mar_Deva",  # Marathi
}


@lru_cache(maxsize=1)
def _load_nllb_model():
    """Load NLLB once and reuse it across translations."""
    import torch
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    model.eval()
    return model


@lru_cache(maxsize=8)
def _load_nllb_tokenizer(source_language):
    """Cache the tokenizer configured for each supported source language."""
    from transformers import AutoTokenizer

    language_code = NLLB_LANGUAGE_CODES[source_language]
    return AutoTokenizer.from_pretrained(
        NLLB_MODEL_NAME,
        src_lang=language_code,
    )


def _translate_with_argos(text, source_language):
    """Translate locally with an installed Argos pair; return None if unavailable."""
    if argos_translate is None or not source_language or source_language == "en":
        return None

    try:
        # Avoid the known Bengali/Stanza sentence-splitter issue.
        if source_language == "bn":
            import argostranslate.settings as argos_settings
            argos_settings.chunk_type = argos_settings.ChunkType.MINISBD
            cache_clear = getattr(
                argos_translate.get_installed_languages, "cache_clear", None
            )
            if callable(cache_clear):
                cache_clear()

        languages = argos_translate.get_installed_languages()
        source_lang = next(
            (language for language in languages if language.code == source_language),
            None,
        )
        target_lang = next(
            (language for language in languages if language.code == "en"), None
        )
        if source_lang is None or target_lang is None:
            return None

        translator = source_lang.get_translation(target_lang)
        if translator is None:
            return None

        result = translator.translate(text)
        return result.strip() if isinstance(result, str) and result.strip() else None
    except Exception as exc:
        print(f"Argos local translation failed ({source_language}->en): {exc}")
        return None


def _translate_with_nllb(text, source_language):
    """Translate supported Indian languages to English using the local NLLB model."""
    if source_language not in NLLB_LANGUAGE_CODES:
        return None

    try:
        import torch

        tokenizer = _load_nllb_tokenizer(source_language)
        model = _load_nllb_model()
        target_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        if target_token_id is None or target_token_id == tokenizer.unk_token_id:
            print("NLLB tokenizer could not resolve the English target token.")
            return None

        # Tokenize by sentences where possible, then translate each manageable chunk.
        sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
        chunks = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            # Keep each chunk below the model's input token limit.
            if current and len(candidate) > 700:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        if not chunks:
            chunks = [text[:700]]

        translated_chunks = []
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    max_new_tokens=256,
                    num_beams=2,
                )
            translated = tokenizer.decode(
                output[0], skip_special_tokens=True
            ).strip()
            if not translated:
                return None
            translated_chunks.append(translated)

        result = " ".join(translated_chunks).strip()
        return result or None
    except Exception as exc:
        print(f"NLLB translation failed ({source_language}->en): {type(exc).__name__}: {exc}")
        return None


def translate_to_english(text):
    """Translate text into English using local Argos/NLLB before online fallbacks.

    Returns an empty string only when all available translation methods fail.
    Successful translations are cached for the current app session.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.strip()
    if text in _translation_cache:
        return _translation_cache[text]

    try:
        source_language = detect(text)
    except Exception as exc:
        print(f"Language detection before translation failed: {exc}")
        return ""

    if source_language == "en":
        _translation_cache[text] = text
        return text

    # 1. Installed Argos package: fast and fully local where available.
    argos_result = _translate_with_argos(text, source_language)
    if argos_result:
        _translation_cache[text] = argos_result
        return argos_result

    # 2. NLLB local model: works even when online services are rate-limited.
    nllb_result = _translate_with_nllb(text, source_language)
    if nllb_result:
        _translation_cache[text] = nllb_result
        return nllb_result

    # 3. Online fallbacks for languages not supported by the local model,
    # or if local model translation failed. Do not make repeated retries.
    for translator_class in (MyMemoryTranslator, GoogleTranslator):
        try:
            result = translator_class(
                source=source_language,
                target="en",
            ).translate(text)
            if isinstance(result, str) and result.strip():
                result = result.strip()
                _translation_cache[text] = result
                return result
        except Exception as exc:
            print(
                f"{translator_class.__name__} translation failed "
                f"({source_language}->en): {type(exc).__name__}: {exc}"
            )

    print(f"Translation failed for language '{source_language}'.")
    return ""


def calculate_credibility_score(

    fake_confidence,

    fake_prediction,

    sentiment_score=None,

    bias_score=0,

    claim_score=100

):

    """

    Calculate an AI-based news credibility score from 0 to 100.

    Higher score = stronger credibility indicators.

    Parameters

    ----------

    fake_confidence : float

        Confidence returned by the fake-news classifier, from 0 to 1.

    fake_prediction : str

        "LIKELY REAL" or "POTENTIALLY FAKE".

    sentiment_score : float, optional

        Sentiment model confidence, from 0 to 1.

    bias_score : float, optional

        Bias score from 0 to 100. Higher means more biased.

    claim_score : float, optional

        Claim/evidence score from 0 to 100. For the current

        version, 100 is used when claim analysis is not yet enabled.

    Returns

    -------

    float

        Credibility score between 0 and 100.

    """

    # Keep values inside valid ranges.

    fake_confidence = max(0.0, min(1.0, float(fake_confidence or 0.0)))

    bias_score = max(0.0, min(100.0, float(bias_score or 0.0)))

    claim_score = max(0.0, min(100.0, float(claim_score or 0.0)))

    # Fake-news component.

    if fake_prediction == "LIKELY REAL":

        fake_component = fake_confidence * 100

    elif fake_prediction == "POTENTIALLY FAKE":

        fake_component = (1.0 - fake_confidence) * 100

    else:

        fake_component = 50.0

    # Sentiment component.

    # Strongly emotional language is treated as a small credibility

    # penalty. This is only an indicator, not proof of misinformation.

    if sentiment_score is not None:

        sentiment_confidence = max(

            0.0,

            min(1.0, float(sentiment_score))

        )

        sentiment_component = 100 - (sentiment_confidence * 30)

    else:

        sentiment_component = 50.0

    # Lower bias = higher credibility contribution.

    bias_component = 100 - bias_score

    # Weighted overall score.

    credibility_score = (

        fake_component * 0.50 +

        bias_component * 0.20 +

        claim_score * 0.20 +

        sentiment_component * 0.10

    )

    return round(

        max(0.0, min(100.0, credibility_score)),

        2

    )

def get_credibility_level(score):

    """

    Convert a credibility score into a human-readable level.

    """

    score = float(score)

    if score >= 80:

        return "Highly Credible"

    if score >= 60:

        return "Moderately Credible"

    if score >= 40:

        return "Low Credibility"

    return "Very Low Credibility"

# ============================================================

# 17. BIAS DETECTION

# ============================================================

# Transparent linguistic indicators. These are screening signals,

# not proof that an article or source is objectively biased.

LOADED_BIAS_WORDS = {

    "shocking", "disastrous", "outrageous", "horrible",

    "terrible", "awful", "ridiculous", "absurd",

    "pathetic", "corrupt", "greedy", "evil",

    "dangerous", "reckless", "incompetent", "unfair",

    "dishonest", "scandalous", "radical", "extreme",

    "brutal", "devastating", "miraculous", "perfect",

    "heroic", "fantastic", "wonderful", "amazing",

    "disgusting"

}

INTENSIFIER_WORDS = {

    "very", "extremely", "completely", "totally",

    "absolutely", "clearly", "obviously", "undeniably",

    "massively", "dramatically", "highly", "deeply",

    "certainly", "definitely", "never", "always"

}

OPINION_MARKERS = {

    "i think", "i believe", "in my opinion",

    "we believe", "we think", "it seems",

    "it is obvious", "obviously", "clearly",

    "should", "must", "undoubtedly"

}

FEAR_OR_ALARM_WORDS = {

    "crisis", "disaster", "threat", "panic",

    "fear", "danger", "warning", "catastrophe",

    "collapse", "chaos", "terror", "alarming"

}

def detect_bias(text):

    """

    Detect potentially biased linguistic patterns.

    Returns a dictionary with:

        score       : 0-100, higher means more potential bias

        level       : Low / Moderate / High / Very High

        terms       : detected words/phrases

        indicators  : detected bias categories

        sentences   : sentences containing indicators

    """

    if not text or not text.strip():

        return {

            "score": 0.0,

            "level": "Low",

            "terms": [],

            "indicators": [],

            "sentences": []

        }

    clean_text = re.sub(

        r"\s+",

        " ",

        text

    ).strip()

    lower_text = clean_text.lower()

    words = re.findall(

        r"\b[a-zA-Z][a-zA-Z'-]*\b",

        lower_text

    )

    if not words:

        return {

            "score": 0.0,

            "level": "Low",

            "terms": [],

            "indicators": [],

            "sentences": []

        }

    terms = []

    indicators = []

    points = 0.0

    # Emotionally loaded words

    loaded_found = sorted(

        set(words).intersection(

            LOADED_BIAS_WORDS

        )

    )

    if loaded_found:

        terms.extend(loaded_found)

        indicators.append(

            "Emotionally loaded language"

        )

        points += min(

            40,

            len(loaded_found) * 8

        )

    # Intensifiers

    intensifier_found = sorted(

        set(words).intersection(

            INTENSIFIER_WORDS

        )

    )

    if intensifier_found:

        terms.extend(intensifier_found)

        indicators.append(

            "Intensifying or absolute language"

        )

        points += min(

            20,

            len(intensifier_found) * 4

        )

    # Opinion markers

    opinion_found = []

    for phrase in OPINION_MARKERS:

        if re.search(

            r"\b" + re.escape(phrase) + r"\b",

            lower_text

        ):

            opinion_found.append(phrase)

    if opinion_found:

        terms.extend(opinion_found)

        indicators.append(

            "Opinion or subjective wording"

        )

        points += min(

            20,

            len(opinion_found) * 8

        )

    # Fear/alarm wording

    alarm_found = sorted(

        set(words).intersection(

            FEAR_OR_ALARM_WORDS

        )

    )

    if alarm_found:

        terms.extend(alarm_found)

        indicators.append(

            "Fear or alarm language"

        )

        points += min(

            20,

            len(alarm_found) * 5

        )

    # Excessive exclamation marks

    exclamation_count = clean_text.count("!")

    if exclamation_count >= 2:

        indicators.append(

            "Excessive exclamation marks"

        )

        points += min(

            10,

            exclamation_count * 2

        )

    # Excessive capitalization

    capitalized_words = re.findall(

        r"\b[A-Z]{3,}\b",

        text

    )

    if len(capitalized_words) >= 2:

        indicators.append(

            "Excessive capitalization"

        )

        points += min(

            10,

            len(capitalized_words) * 2

        )

    # Reduce effect for very short inputs.

    if len(words) < 30:

        points *= 0.75

    score = round(

        min(

            100.0,

            points

        ),

        2

    )

    if score >= 70:

        level = "Very High"

    elif score >= 45:

        level = "High"

    elif score >= 20:

        level = "Moderate"

    else:

        level = "Low"

    unique_terms = list(

        dict.fromkeys(terms)

    )

    # Sentence-level evidence

    sentence_results = []

    sentences = re.split(

        r"(?<=[.!?])\s+",

        clean_text

    )

    for sentence in sentences:

        sentence_lower = sentence.lower()

        sentence_words = re.findall(

            r"\b[a-zA-Z][a-zA-Z'-]*\b",

            sentence_lower

        )

        sentence_terms = []

        for word in sentence_words:

            if word in LOADED_BIAS_WORDS:

                sentence_terms.append(word)

            elif word in INTENSIFIER_WORDS:

                sentence_terms.append(word)

            elif word in FEAR_OR_ALARM_WORDS:

                sentence_terms.append(word)

        for phrase in OPINION_MARKERS:

            if phrase in sentence_lower:

                sentence_terms.append(phrase)

        sentence_terms = list(

            dict.fromkeys(sentence_terms)

        )

        if sentence_terms:

            sentence_results.append({

                "sentence": sentence,

                "terms": sentence_terms

            })

    return {

        "score": score,

        "level": level,

        "terms": unique_terms,

        "indicators": list(

            dict.fromkeys(indicators)

        ),

        "sentences": sentence_results

    }

# ============================================================

# 18. CLAIM DETECTION

# ============================================================

# Claim Detection is a linguistic screening module.

# It identifies sentences that are likely to contain

# factual/verifiable assertions. It does not verify whether

# the claim is actually true.

CLAIM_CUE_WORDS = {

    "announced", "reported", "confirmed", "said", "stated",

    "according", "officials", "researchers", "scientists",

    "study", "studies", "found", "shows", "showed",

    "revealed", "increased", "decreased", "reached",

    "caused", "causes", "will", "plans", "planned",

    "expects", "expected", "launched", "introduced",

    "approved", "rejected", "won", "lost", "created",

    "discovered", "developed", "produced", "recorded",

    "estimated", "measured", "proved", "evidence"

}


CLAIM_QUANTIFIER_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:%|percent|million|billion|thousand|crore|lakh|"
    r"km|kg|years?|months?|days?|people|students?|"
    r"companies?|dollars?|rupees?|₹|\$|€|£)?\b",
    re.IGNORECASE
)


CLAIM_TEMPORAL_WORDS = {

    "today", "yesterday", "tomorrow", "recently",

    "currently", "now", "this year", "last year",

    "next year", "monday", "tuesday", "wednesday",

    "thursday", "friday", "saturday", "sunday"

}


def _prepare_text_for_english_analysis(text):
    """Translate non-English input before English-only NLP rules/models run.

    Returns (analysis_text, source_language, translated). If translation fails,
    the original text is retained rather than silently discarded.
    """
    if not isinstance(text, str) or not text.strip():
        return "", "unknown", False

    original = text.strip()
    try:
        source_language = detect(original)
    except Exception as exc:
        print(f"Language detection for claim analysis failed: {exc}")
        return original, "unknown", False

    if source_language == "en":
        return original, source_language, False

    try:
        translated = translate_to_english(original)
    except Exception as exc:
        print(f"Translation before claim analysis failed: {exc}")
        translated = ""

    if isinstance(translated, str) and translated.strip():
        return translated.strip(), source_language, True

    print(
        f"Could not translate {source_language} text for claim analysis; "
        "English-based claim rules may return incomplete results."
    )
    return original, source_language, False

def detect_claims(text, max_claims=10):

    """

    Identify sentences that are likely to contain factual claims.

    The module uses transparent linguistic features:

    - claim cue words

    - named entities

    - numbers/quantities

    - dates/temporal expressions

    - assertive sentence structure

    It returns claim candidates, not truth verification.

    Returns:

        {

            "total_sentences": int,

            "claim_count": int,

            "claim_score": float,       # 0-100 evidence/coverage score

            "level": str,

            "claims": list

        }

    """

    if not text or not text.strip():

        return {

            "total_sentences": 0,

            "claim_count": 0,

            "claim_score": 0.0,

            "level": "No Claims Detected",

            "claims": []

        }

    # This module's claim cues and NER model are English-based. Translate
    # non-English input first so Kannada/Telugu/Malayalam articles are analysed.
    text, source_language, was_translated = _prepare_text_for_english_analysis(text)

    clean_text = re.sub(

        r"\s+",

        " ",

        text

    ).strip()

    # Split into sentences.

    sentences = [

        sentence.strip()

        for sentence in re.split(

            r"(?<=[.!?])\s+",

            clean_text

        )

        if sentence.strip()

    ]

    if not sentences:

        return {

            "total_sentences": 0,

            "claim_count": 0,

            "claim_score": 0.0,

            "level": "No Claims Detected",

            "claims": []

        }

    # Avoid running NER on a huge article at once.

    ner_text = clean_text[:4000]

    try:

        ner_entities = ner_model(ner_text)

    except Exception:

        ner_entities = []

    entity_words = set()

    for entity in ner_entities:

        entity_text = str(

            entity.get("word", "")

        ).strip().lower()

        if entity_text:

            entity_words.add(entity_text)

    claim_candidates = []

    for sentence in sentences:

        lower_sentence = sentence.lower()

        words = re.findall(

            r"\b[a-zA-Z][a-zA-Z'-]*\b",

            lower_sentence

        )

        if not words:

            continue

        cue_matches = sorted(

            set(words).intersection(

                CLAIM_CUE_WORDS

            )

        )

        number_matches = CLAIM_QUANTIFIER_PATTERN.findall(

            sentence

        )

        temporal_matches = []

        for temporal in CLAIM_TEMPORAL_WORDS:

            if temporal in lower_sentence:

                temporal_matches.append(

                    temporal

                )

        sentence_entities = []

        for entity_word in entity_words:

            if entity_word in lower_sentence:

                sentence_entities.append(

                    entity_word

                )

        # Strong signals of a factual/verifiable assertion.

        points = 0.0

        if cue_matches:

            points += min(

                45.0,

                len(cue_matches) * 15.0

            )

        if number_matches:

            points += min(

                25.0,

                len(number_matches) * 10.0

            )

        if temporal_matches:

            points += min(

                15.0,

                len(temporal_matches) * 7.5

            )

        if sentence_entities:

            points += min(

                15.0,

                len(sentence_entities) * 5.0

            )

        # Assertive verbs make a sentence more likely to be

        # a verifiable statement.

        assertive_patterns = [

            r"\b(is|are|was|were|has|have|had)\b",

            r"\b(will|can|could|may|might)\b",

            r"\b(according to|officials said|researchers said)\b"

        ]

        assertive_found = any(

            re.search(

                pattern,

                lower_sentence

            )

            for pattern in assertive_patterns

        )

        if assertive_found:

            points += 10.0

        # A sentence is a candidate when it has enough

        # evidence from these transparent signals.

        if points >= 20:

            confidence = min(

                100.0,

                round(points, 2)

            )

            reasons = []

            if cue_matches:

                reasons.append(

                    "Claim cue words"

                )

            if number_matches:

                reasons.append(

                    "Quantitative information"

                )

            if temporal_matches:

                reasons.append(

                    "Temporal information"

                )

            if sentence_entities:

                reasons.append(

                    "Named entities"

                )

            if assertive_found:

                reasons.append(

                    "Assertive statement structure"

                )

            claim_candidates.append({

                "claim": sentence,

                "confidence": confidence,

                "reasons": reasons,

                "cue_words": cue_matches,

                "numbers": number_matches,

                "entities": sentence_entities

            })

    # Sort strongest candidates first.

    claim_candidates.sort(

        key=lambda item: item["confidence"],

        reverse=True

    )

    selected_claims = claim_candidates[

        :max_claims

    ]

    total_sentences = len(sentences)

    claim_count = len(claim_candidates)

    # Coverage-based claim score:

    # more identifiable factual assertions = stronger claim presence.

    if total_sentences:

        coverage = (

            claim_count /

            total_sentences

        ) * 100

    else:

        coverage = 0.0

    average_confidence = (

        sum(

            item["confidence"]

            for item in claim_candidates

        ) / claim_count

        if claim_count

        else 0.0

    )

    # Combine claim coverage and detection confidence.

    claim_score = round(

        min(

            100.0,

            (coverage * 0.60) +

            (average_confidence * 0.40)

        ),

        2

    )

    if claim_count == 0:

        level = "No Claims Detected"

    elif claim_score >= 70:

        level = "Strong Claim Presence"

    elif claim_score >= 40:

        level = "Moderate Claim Presence"

    else:

        level = "Limited Claim Presence"

    return {

        "total_sentences": total_sentences,

        "claim_count": claim_count,

        "claim_score": claim_score,

        "level": level,

        "claims": selected_claims

    }

# ============================================================

# CONTRADICTION DETECTION

# ============================================================

def _normalize_nli_results(results):

    """

    Normalize Transformers pipeline output into a flat list.

    """

    if results and isinstance(results[0], list):

        results = results[0]

    return results or []

def _get_nli_scores(premise, hypothesis):

    """

    Run the NLI model on a pair of statements and return

    normalized entailment/contradiction/neutral scores.

    """

    try:

        results = contradiction_model(

            {

                "text": premise,

                "text_pair": hypothesis

            },

            top_k=None

        )

    except Exception as e:

        print(

            "NLI prediction error:",

            e

        )

        return {

            "entailment": 0.0,

            "contradiction": 0.0,

            "neutral": 0.0

        }

    results = _normalize_nli_results(

        results

    )

    scores = {

        "entailment": 0.0,

        "contradiction": 0.0,

        "neutral": 0.0

    }

    for item in results:

        label = str(

            item.get("label", "")

        ).upper()

        score = float(

            item.get("score", 0.0)

        )

        if "CONTRADICTION" in label:

            scores["contradiction"] = score

        elif "ENTAILMENT" in label:

            scores["entailment"] = score

        elif "NEUTRAL" in label:

            scores["neutral"] = score

    return scores

def detect_contradictions(

    text,

    max_sentences=10,

    max_pairs=45,

    threshold=0.70

):

    """

    Detect likely contradictions between important statements.

    The function first identifies claim-like sentences using the

    existing Claim Detection module, then compares a limited

    number of sentence pairs with a Natural Language Inference

    model.

    Returns:

        {

            "pairs_checked": int,

            "contradiction_count": int,

            "contradiction_score": float,

            "level": str,

            "contradictions": list

        }

    """

    if not text or not text.strip():

        return {

            "pairs_checked": 0,

            "contradiction_count": 0,

            "contradiction_score": 0.0,

            "level": "No Contradictions Detected",

            "contradictions": []

        }

    # NLI model is English-based; translate before extracting/comparing sentences.
    text, source_language, was_translated = _prepare_text_for_english_analysis(text)

    clean_text = re.sub(

        r"\s+",

        " ",

        text

    ).strip()

    # Use claim candidates first.

    try:

        claim_result = detect_claims(

            clean_text,

            max_claims=max_sentences

        )

        claim_items = claim_result.get(

            "claims",

            []

        )

    except Exception as e:

        print(

            "Claim extraction for contradiction detection failed:",

            e

        )

        claim_items = []

    sentences = []

    for item in claim_items:

        sentence = item.get(

            "claim",

            ""

        ).strip()

        if sentence:

            sentences.append(sentence)

    # Fallback if Claim Detection finds too few candidates.

    if len(sentences) < 2:

        sentences = [

            sentence.strip()

            for sentence in re.split(

                r"(?<=[.!?])\s+",

                clean_text

            )

            if sentence.strip()

        ]

    # Remove duplicates while preserving order.

    sentences = list(

        dict.fromkeys(sentences)

    )

    # Limit pairwise comparisons for CPU efficiency.

    sentences = sentences[

        :max_sentences

    ]

    pairs = []

    for i in range(len(sentences)):

        for j in range(i + 1, len(sentences)):

            pairs.append(

                (

                    sentences[i],

                    sentences[j]

                )

            )

            if len(pairs) >= max_pairs:

                break

        if len(pairs) >= max_pairs:

            break

    contradictions = []

    for premise, hypothesis in pairs:

        nli_scores = _get_nli_scores(

            premise,

            hypothesis

        )

        contradiction_probability = (

            nli_scores["contradiction"]

        )

        if contradiction_probability >= threshold:

            contradictions.append({

                "statement_1": premise,

                "statement_2": hypothesis,

                "confidence": round(

                    contradiction_probability * 100,

                    2

                ),

                "entailment_score": round(

                    nli_scores["entailment"] * 100,

                    2

                ),

                "neutral_score": round(

                    nli_scores["neutral"] * 100,

                    2

                )

            })

    contradictions.sort(

        key=lambda item: item["confidence"],

        reverse=True

    )

    # Convert number of strong contradictions into a

    # bounded risk score.

    if pairs:

        contradiction_ratio = (

            len(contradictions) /

            len(pairs)

        )

    else:

        contradiction_ratio = 0.0

    contradiction_score = round(

        min(

            100.0,

            contradiction_ratio * 100.0

        ),

        2

    )

    if not contradictions:

        level = "No Contradictions Detected"

    elif contradiction_score >= 30:

        level = "High Contradiction"

    elif contradiction_score >= 10:

        level = "Moderate Contradiction"

    else:

        level = "Low Contradiction"

    return {

        "pairs_checked": len(pairs),

        "contradiction_count": len(contradictions),

        "contradiction_score": contradiction_score,

        "level": level,

        "contradictions": contradictions

    }
