from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import datasets
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers import pre_tokenizers
from tokenizers.trainers import BneTrainer


def _batch_iterator(dataset_split, text_column: str, batch_size: int) -> Iterable[list[str]]:
    for batch in dataset_split.iter(batch_size=batch_size):
        yield batch[text_column]


def _build_pretokenizer(pretok_name: str):
    name = pretok_name.strip().lower()
    factories = {
        "bert": lambda: pre_tokenizers.BertPreTokenizer(),
        "byte_level": lambda: pre_tokenizers.ByteLevel(),
        "digits": lambda: pre_tokenizers.Digits(),
        "fixed_length": lambda: pre_tokenizers.FixedLength(),
        "metaspace": lambda: pre_tokenizers.Metaspace(),
        "punctuation": lambda: pre_tokenizers.Punctuation(),
        "split_byte_level": lambda: pre_tokenizers.SplitByteLevel(),
        "unicode_scripts": lambda: pre_tokenizers.UnicodeScripts(),
        "whitespace": lambda: pre_tokenizers.Whitespace(),
        "whitespace_split": lambda: pre_tokenizers.WhitespaceSplit(),
    }
    unsupported = {"delimiter", "split", "sequence"}

    if name in factories:
        return factories[name]()
    if name in unsupported:
        raise ValueError(
            f"pretok '{pretok_name}' requires extra parameters and is not supported by this script. "
            f"Use one of: {', '.join(sorted(factories.keys()))}"
        )
    raise ValueError(f"Unknown pretok '{pretok_name}'. Supported: {', '.join(sorted(factories.keys()))}")


def _infer_local_dataset_builder(local_path: Path) -> str:
    suffix = local_path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return "text"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        f"Could not infer dataset format from extension '{suffix}'. "
        "Use --dataset-format (text/json/csv/parquet)."
    )


def _extensions_for_builder(builder: str) -> tuple[str, ...]:
    if builder == "text":
        return (".txt", ".text")
    if builder == "json":
        return (".json", ".jsonl")
    if builder == "csv":
        return (".csv", ".tsv")
    if builder == "parquet":
        return (".parquet",)
    raise ValueError(f"Unsupported dataset format '{builder}'.")


def _infer_builder_from_directory(local_dir: Path) -> str:
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    suffixes = {p.suffix.lower() for p in files}

    known = {
        "parquet": {".parquet"},
        "json": {".json", ".jsonl"},
        "csv": {".csv", ".tsv"},
        "text": {".txt", ".text"},
    }
    matching = [builder for builder, exts in known.items() if suffixes & exts]
    if len(matching) == 1:
        return matching[0]
    if len(matching) == 0:
        raise ValueError(
            f"Could not infer dataset format from directory '{local_dir}'. "
            "Use --dataset-format (text/json/csv/parquet)."
        )
    raise ValueError(
        f"Directory '{local_dir}' contains multiple supported formats ({matching}). "
        "Specify one with --dataset-format."
    )


def _collect_files_in_tree(local_dir: Path, builder: str) -> list[str]:
    allowed_exts = _extensions_for_builder(builder)
    files = [str(p) for p in local_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
    if not files:
        raise ValueError(f"No files with extensions {allowed_exts} found under '{local_dir}'.")
    return sorted(files)


def _select_parquet_text_column(
    parquet_file: Path,
    preferred_column: str,
    fallback_columns: tuple[str, ...] = ("content",),
) -> Optional[str]:
    schema = pq.read_schema(parquet_file)
    names = set(schema.names)
    if preferred_column in names:
        return preferred_column
    for col in fallback_columns:
        if col in names:
            return col
    return None


def _load_local_parquet_with_fallback_columns(
    data_files: list[str],
    split: str,
    text_column: str,
) -> datasets.Dataset:
    grouped_files: dict[str, list[str]] = {}
    skipped = 0
    for file_str in data_files:
        source_col = _select_parquet_text_column(Path(file_str), preferred_column=text_column)
        if source_col is None:
            skipped += 1
            continue
        grouped_files.setdefault(source_col, []).append(file_str)

    if not grouped_files:
        raise ValueError(
            f"None of the parquet files contained '{text_column}' or fallback columns."
        )

    loaded_parts = []
    for source_col, files in grouped_files.items():
        ds_part = datasets.load_dataset("parquet", data_files=files, split=split, columns=[source_col])
        if source_col != text_column:
            ds_part = ds_part.rename_column(source_col, text_column)
        ds_part = ds_part.cast_column(text_column, datasets.Value("string"))
        loaded_parts.append(ds_part)

    if len(loaded_parts) == 1:
        result = loaded_parts[0]
    else:
        result = datasets.concatenate_datasets(loaded_parts)

    if skipped > 0:
        print(
            f"Warning: skipped {skipped} parquet files without '{text_column}' "
            "or fallback text columns."
        )
    return result


def _load_dataset_split(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    dataset_format: Optional[str] = None,
):
    dataset_path = Path(dataset_name)
    if dataset_path.is_file():
        if dataset_config is not None:
            raise ValueError("When --dataset points to a local file, do not provide --config.")
        builder = dataset_format or _infer_local_dataset_builder(dataset_path)
        if builder == "parquet":
            source_col = _select_parquet_text_column(dataset_path, preferred_column=text_column)
            if source_col is None:
                raise ValueError(
                    f"Parquet file '{dataset_path}' has neither '{text_column}' nor fallback columns (e.g. content)."
                )
            ds = datasets.load_dataset(builder, data_files=str(dataset_path), split=split, columns=[source_col])
            if source_col != text_column:
                ds = ds.rename_column(source_col, text_column)
            ds = ds.cast_column(text_column, datasets.Value("string"))
            return ds
        return datasets.load_dataset(builder, data_files=str(dataset_path), split=split)
    if dataset_path.is_dir():
        if dataset_config is not None:
            raise ValueError("When --dataset points to a local directory, do not provide --config.")
        builder = dataset_format or _infer_builder_from_directory(dataset_path)
        data_files = _collect_files_in_tree(dataset_path, builder)
        if builder == "parquet":
            return _load_local_parquet_with_fallback_columns(
                data_files=data_files,
                split=split,
                text_column=text_column,
            )
        return datasets.load_dataset(builder, data_files=data_files, split=split)

    if dataset_config is None:
        return datasets.load_dataset(dataset_name, split=split)
    return datasets.load_dataset(dataset_name, dataset_config, split=split)


def _sanitize_for_filename(value: str) -> str:
    return value.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _dataset_label_for_output(dataset_name: str) -> str:
    dataset_path = Path(dataset_name)
    if dataset_path.is_file():
        return dataset_path.stem
    if dataset_path.is_dir():
        return dataset_path.name

    # Handle path-like inputs even when the path doesn't exist yet.
    looks_like_local_path = (
        dataset_path.is_absolute()
        or dataset_name.startswith((".", "~", "\\\\"))
        or (len(dataset_name) > 1 and dataset_name[1] == ":")
    )
    if looks_like_local_path:
        return dataset_path.stem if dataset_path.suffix else dataset_path.name

    return dataset_name


def build_default_output_path(
    dataset_name: str,
    dataset_config: Optional[str],
    sample_pct: float,
    vocab_size: int,
    max_ngram_length: int,
    pretok: str,
    base_dir: str = "data",
) -> str:
    dataset_part = _sanitize_for_filename(_dataset_label_for_output(dataset_name))
    config_part = _sanitize_for_filename(dataset_config) if dataset_config else "none"
    sample_part = str(sample_pct).replace(".", "pt")
    pretok_part = _sanitize_for_filename(pretok)
    filename = (
        f"bne_{dataset_part}_{config_part}_{sample_part}pct_"
        f"v{vocab_size}_ng{max_ngram_length}_{pretok_part}.json"
    )
    return str(Path(base_dir) / filename)


def train_bne_tokenizer(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    output_path: Optional[str] = None,
    vocab_size: int = 8000,
    max_ngram_length: int = 2,
    batch_size: int = 1000,
    sample_pct: float = 100.0,
    pretok: str = "split_byte_level",
    dataset_format: Optional[str] = None,
) -> tuple[Tokenizer, str]:
    dataset_split = _load_dataset_split(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        text_column=text_column,
        dataset_format=dataset_format,
    )

    if text_column not in dataset_split.column_names:
        raise ValueError(f"Column '{text_column}' not found. Available: {dataset_split.column_names}")

    if sample_pct <= 0 or sample_pct > 100:
        raise ValueError(f"sample_pct must be in (0, 100], got {sample_pct}")

    total_size = len(dataset_split)
    if sample_pct < 100.0:
        sample_size = max(1, int(total_size * (sample_pct / 100.0)))
        dataset_split = dataset_split.shuffle().select(range(sample_size))

    tokenizer = Tokenizer(BNE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = _build_pretokenizer(pretok)
    trainer = BneTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=vocab_size,
        max_ngram_length=max_ngram_length,
    )

    tokenizer.train_from_iterator(
        _batch_iterator(dataset_split, text_column=text_column, batch_size=batch_size),
        trainer=trainer,
        length=len(dataset_split),
    )

    resolved_output_path = output_path or build_default_output_path(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        sample_pct=sample_pct,
        vocab_size=vocab_size,
        max_ngram_length=max_ngram_length,
        pretok=pretok,
    )
    output = Path(resolved_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer, str(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save a BNE tokenizer on a dataset.")
    parser.add_argument(
        "--dataset",
        default="JeanKaddour/minipile",
        help="Dataset name, e.g. JeanKaddour/minipile or wikitext",
    )
    parser.add_argument("--config", default=None, help="Dataset config, e.g. wikitext-103-raw-v1")
    parser.add_argument(
        "--dataset-format",
        default=None,
        help="Optional local dataset format override: text, json, csv, parquet",
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--text-column", default="text", help="Text column name")
    parser.add_argument("--output", default=None, help="Where to save tokenizer JSON")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--max-ngram-length", type=int, default=2, help="Maximum n-gram length")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for streaming training")
    parser.add_argument(
        "--pretok",
        default="split_byte_level",
        help="Pre-tokenizer name (file name without .rs), e.g. split_byte_level, byte_level, whitespace",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=100.0,
        help="Random sample size as percentage of the split (0 < value <= 100)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trained_tokenizer, resolved_output_path = train_bne_tokenizer(
        dataset_name=args.dataset,
        dataset_config=args.config,
        split=args.split,
        text_column=args.text_column,
        output_path=args.output,
        vocab_size=args.vocab_size,
        max_ngram_length=args.max_ngram_length,
        batch_size=args.batch_size,
        sample_pct=args.sample_pct,
        pretok=args.pretok,
        dataset_format=args.dataset_format,
    )
    print(f"Saved tokenizer to: {resolved_output_path}")
    encoded = trained_tokenizer.encode("Training BNE is very easy! I've added some numbers: 1234.")
    print(f"Sample token ids: {encoded.ids}")
    print(f"Sample tokens (escaped): {[t.encode('unicode_escape').decode('ascii') for t in encoded.tokens]}")
    decoded = trained_tokenizer.decode(encoded.ids)
    print(f"Decoded from ids (escaped): {decoded.encode('unicode_escape').decode('ascii')}")
