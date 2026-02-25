from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_reduced_output_path(tokenizer_path: Path, vocab_size: int) -> Path:
    return tokenizer_path.with_name(f"{tokenizer_path.stem}reduced{vocab_size}.json")


def _token_from_merge(merge: Any) -> str:
    if isinstance(merge, str):
        return "".join(merge.split())
    if isinstance(merge, list):
        return "".join(str(p) for p in merge)
    raise ValueError(f"Unsupported merge entry type: {type(merge).__name__}")


def downsize_tokenizer(tokenizer_json_path: str, vocab_size: int) -> str:
    path = Path(tokenizer_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    model = data.get("model")
    if not isinstance(model, dict):
        raise ValueError("Invalid tokenizer file: missing 'model' object.")

    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("Invalid tokenizer file: expected model.vocab to be a dict token->id.")

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")

    max_vocab_size = len(vocab)
    if vocab_size > max_vocab_size:
        raise ValueError(
            f"Requested vocab_size={vocab_size}, but tokenizer only has {max_vocab_size} tokens."
        )

    kept_vocab_items = [(token, token_id) for token, token_id in vocab.items() if token_id < vocab_size]
    if len(kept_vocab_items) != vocab_size:
        raise ValueError(
            "Tokenizer vocab ids are not contiguous from 0..N-1; cannot safely keep first K tokens by id."
        )

    kept_vocab = dict(sorted(kept_vocab_items, key=lambda x: x[1]))
    model["vocab"] = kept_vocab
    kept_tokens = set(kept_vocab.keys())

    merges = model.get("merges")
    if isinstance(merges, list):
        filtered_merges = []
        for merge in merges:
            produced_token = _token_from_merge(merge)
            if produced_token in kept_tokens:
                filtered_merges.append(merge)
        model["merges"] = filtered_merges

    added_tokens = data.get("added_tokens")
    if isinstance(added_tokens, list):
        data["added_tokens"] = [tok for tok in added_tokens if int(tok.get("id", -1)) < vocab_size]

    output_path = _build_reduced_output_path(path, vocab_size)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reduced tokenizer with first K vocabulary entries.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--vocab-size", required=True, type=int, help="Target reduced vocabulary size K")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved = downsize_tokenizer(args.tokenizer, args.vocab_size)
    print(f"Saved reduced tokenizer to: {saved}")
