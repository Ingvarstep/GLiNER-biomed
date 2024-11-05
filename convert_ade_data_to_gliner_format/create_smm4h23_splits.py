import os
import csv
from typing import Dict, List, Tuple
import regex
from tqdm.auto import tqdm
import random
import spacy
import json


def read_tweet_tsv_file(path: str) -> Dict[str, str]:
    """
    Read a TSV file and return its data in a dictionary format.
    """
    data_dict = {}
    with open(path, "r", encoding="utf-8") as file:
        text = file.read().strip()
        rows = text.split("\n")
        for row in rows:
            parts = row.split("\t")
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                data_dict[key] = value
    return data_dict


def read_span_tsv_file(path: str) -> Dict[str, List[Tuple[int, int, str, str]]]:
    """
    Read a TSV file and return its data in a dictionary format.
    """
    data_dict = {}
    with open(path, "r", encoding="utf-8") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_reader:
            key = row[0]
            span_data = (
                int(row[2]),
                int(row[3]),
                row[4],
                row[5],
            )  # Start, end, text, meddra_llt
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(span_data)
    return data_dict


def merge_tweets_spans(
    tweets: Dict[str, str], spans: Dict[str, List[Tuple[int, int, str, str]]]
) -> Dict[str, Dict[str, object]]:
    """
    Merge tweet text and spans into a single dictionary.
    """
    merged_dict = {}
    for tweet_id, tweet_text in tweets.items():
        merged_dict[tweet_id] = {"text": tweet_text, "spans": spans.get(tweet_id, [])}
    return merged_dict


def is_within_word(text, start, end):
    """
    Check if the span starts or ends within a word.
    """
    if start > 0 and text[start - 1].isalnum():
        return True
    if end < len(text) and text[end].isalnum():
        return True
    return False


def adjust_span_to_word_boundary(text, start, end):
    """
    Adjusts the span to align with word boundaries.
    """
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))

    while start < end and not text[start].isalnum():
        start += 1
    start = min(start, len(text))

    while end > start and not text[end - 1].isalnum():
        end -= 1

    while start > 0 and text[start - 1].isalnum():
        start -= 1

    while end < len(text) and text[end].isalnum():
        end += 1
    end = min(end, len(text))

    return start, end


def process_data_to_json(
    data_dict: Dict[str, Dict[str, object]],
    output_file: str,
    split_type: str,
    all_entity_types: set,
):
    """
    Process the data dictionary and save as JSON file in the desired format.
    """
    nlp = spacy.load("en_core_web_sm")

    data = []
    total_entities = 0

    for tweet_id, tweet_data in tqdm(
        data_dict.items(), desc=f"Processing {split_type} set"
    ):
        # Clean the tweet text
        cleaned_content = regex.sub(r"[^\p{L}\p{N}\p{P}]", " ", tweet_data["text"])
        cleaned_content = regex.sub(r"\p{Z}", " ", cleaned_content)

        # Tokenize the text
        doc = nlp(cleaned_content)
        tokens = [token.text for token in doc]

        # Process entities
        entities = []
        file_entity_types = set()

        for span in tweet_data["spans"]:
            START, END = span[0], span[1]
            if is_within_word(cleaned_content, START, END):
                START, END = adjust_span_to_word_boundary(cleaned_content, START, END)

            entity_type = "Adverse drug event"
            file_entity_types.add(entity_type)
            total_entities += 1

            # Map character offsets to token indices using doc.char_span
            span_obj = doc.char_span(START, END, alignment_mode="expand")
            if span_obj is not None:
                start_token_index = span_obj.start
                end_token_index = span_obj.end - 1  # end index inclusive
                entities.append([start_token_index, end_token_index, entity_type])
            else:
                print(
                    f"Warning: could not create token span for entity in tweet {tweet_id} at chars {START}-{END}"
                )

        # Identify absent entity types in this file
        negative_entity_types = list(all_entity_types - file_entity_types)

        to_append = {
                "tokenized_text": tokens,
                "ner": entities,
                "negatives": negative_entity_types,
            }
        
        assert negative_entity_types or entities

        if not negative_entity_types:
            del to_append["negatives"]

        data.append(to_append)

    # Save data to output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print statistics
    print(f"Processed {len(data_dict)} tweets for {split_type} set.")
    print(f"Total entities processed: {total_entities}")


def main() -> None:
    """
    Create JSON data with new splitting strategy.
    """
    # Set a seed for reproducibility
    seed = 42
    random.seed(seed)

    src = "./raw_data/smm4h23/Task5_train_validation"
    dst = "./data/smm4h23"

    # Read train and development datasets
    train_tweets = read_tweet_tsv_file(os.path.join(src, "Train/train_tweets.tsv"))
    train_spans = read_span_tsv_file(os.path.join(src, "Train/train_spans_norm.tsv"))
    dev_tweets = read_tweet_tsv_file(os.path.join(src, "Dev/tweets.tsv"))
    dev_spans = read_span_tsv_file(os.path.join(src, "Dev/spans_norm.tsv"))

    # Merge train dataset and shuffle
    merged_train_data = merge_tweets_spans(train_tweets, train_spans)
    combined_train_data = list(merged_train_data.items())
    random.shuffle(combined_train_data)

    # Split into new train and val datasets
    split_index = int(0.9 * len(combined_train_data))
    new_train_data = dict(combined_train_data[:split_index])
    new_val_data = dict(combined_train_data[split_index:])

    # Generate JSON data for the new train and val datasets
    all_entity_types = {"Adverse drug event"}

    process_data_to_json(
        new_train_data, os.path.join(dst, "train.json"), "train", all_entity_types
    )
    process_data_to_json(
        new_val_data, os.path.join(dst, "val.json"), "val", all_entity_types
    )

    # Process Dev dataset as test
    merged_dev_data = merge_tweets_spans(dev_tweets, dev_spans)
    process_data_to_json(
        merged_dev_data, os.path.join(dst, "test.json"), "test", all_entity_types
    )


if __name__ == "__main__":
    main()