from typing import Mapping, Union
from bs4 import BeautifulSoup
from huggingface_hub.inference._generated.types import feature_extraction
from nltk.stem import PorterStemmer
from scipy.special import ber
import typer
import gzip
import base64
import json
import re

from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import logging


@dataclass
class ProcessingConfig:
    input_dir: Path
    output_dir: Path
    num_topics: int = 3
    min_word_length: int = 3
    raw_data_dir: str = "raw"
    features_dir: str = "features"


@dataclass
class Error:
    message: str


def setup_directories(config: ProcessingConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / config.raw_data_dir).mkdir(exist_ok=True)
    (config.output_dir / config.features_dir).mkdir(exist_ok=True)


def clean_text(html_content):
    # Create BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove all script and style elements
    for script in soup(["script", "style", "code", "meta", "link"]):
        script.decompose()

    # Get text content
    text = soup.get_text()

    # Remove special characters and extra whitespace
    clean = re.sub(r"[^\w\s.,!?]", "", text)
    clean = " ".join(clean.split())

    return clean.lower()


def preprocess_words(words: list[str]):
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in words]
    return stemmed


def extract_features_with_bertopic(texts: list[str]) -> dict | Error:
    cleaned_texts = [clean_text(text) for text in texts]
    try:
        vectorizer = CountVectorizer(
            ngram_range=(1, 2), stop_words=set(stopwords.words("english"))
        )

        model_id = "MaartenGr/BERTopic_ArXiv"
        topic_model = BERTopic.load(
            model_id,
        )

        topics, probabilities = topic_model.transform(cleaned_texts)

        topic_words = {}
        for topic_id in topics:
            words = topic_model.get_topic(topic_id)
            if isinstance(words, Mapping):
                topic_words[topic_id] = [word for word, _ in words]
            if isinstance(words, list):
                topic_words[topic_id] = [word for word, _ in words]

        formatted_topics = [
            {"topic": int(topic_id), "words": words}
            for topic_id, words in topic_words.items()
        ]

        feature_extraction = {
            "topics": formatted_topics,
            "document_count": len(texts),
        }
        print(feature_extraction)
        return feature_extraction
    except Exception as e:
        return Error(f"Failed to analyze data {e}")


def extract_lda_features(texts: list[str], config: ProcessingConfig) -> dict | Error:
    stop_words = set(stopwords.words("english"))
    cleaned_texts = [clean_text(text) for text in texts]
    try:
        tokenized_texts = [
            [
                word
                for word in word_tokenize(text)
                if word not in stop_words and len(word) >= config.min_word_length
            ]
            for text in cleaned_texts
        ]

        processed_texts = [preprocess_words(words) for words in tokenized_texts]

        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        lda_model = models.LdaModel(
            corpus, num_topics=config.num_topics, id2word=dictionary
        )

        return {
            "topics": [
                {"topic": i, "words": [word for word, probaility in words]}
                for i, words in lda_model.show_topics(formatted=False)
            ],
            "vocabulary_size": len(dictionary),
            "document_count": len(texts),
        }
    except Exception as e:
        return Error(f"Failed to analyze data {e}")


def generate_count_report(counter: Counter):
    total_words = sum(counter.values())
    report = {
        "total_words": total_words,
        "unique_words": len(counter),
        "most_common": counter.most_common(200),
        "frequency_distribution": {
            word: count / total_words for word, count in counter.items()
        },
    }
    return report


def process_files(config: ProcessingConfig) -> None:
    json_files = list(config.input_dir.glob("**/*.json"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Processing JSON files", total=len(json_files))
        all_words_set = Counter()

        for json_file in json_files:

            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            if data.get("base64") and data["content_encoding"] == "application/gzip":
                compressed_content = base64.b64decode(data["content"])
                decompressed_content = gzip.decompress(compressed_content).decode(
                    "utf-8"
                )

                lda_features = extract_lda_features([decompressed_content], config)
                if isinstance(lda_features, Error):
                    logging.error(lda_features)
                    continue

                bert_features = extract_features_with_bertopic([decompressed_content])
                if isinstance(bert_features, Error):
                    logging.error(bert_features)
                    continue

                features_results = {
                    "metadata": data["metadata"],
                    "features": {
                        "lda_features": lda_features,
                        "bert_features": bert_features,
                    },
                }
                print(features_results)

                for topic in lda_features["topics"]:
                    all_words_set.update(topic["words"])

                features_dir = config.output_dir / data["metadata"].get("domain")
                features_dir.mkdir(exist_ok=True)

                output_file = features_dir / json_file.name
                with open(output_file, "w", encoding="utf-8") as file:
                    json.dump(features_results, file, indent=4)

                progress.advance(task)
        with open(config.output_dir / "word_list.json", "w", encoding="utf-8") as wl:
            json.dump(generate_count_report(all_words_set), wl, indent=4)


def main(
    input_dir: str = typer.Argument(..., help="Directory containing HTML files"),
    output_dir: str = typer.Argument(..., help="Directory for output files"),
    num_topics: int = typer.Option(3, help="Number of topics for LDA"),
    min_word_length: int = typer.Option(3, help="Minimum word length to consider"),
) -> None:
    """Process HTML files and extract features."""
    config = ProcessingConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        num_topics=num_topics,
        min_word_length=min_word_length,
    )

    setup_directories(config)
    process_files(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    typer.run(main)
