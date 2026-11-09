"""
Step 2.5: TSDAE bridge training from an MLM/DAPT encoder.

This script trains a TSDAE-style sentence embedding model from a plain
corpus (one sentence per line). The structure mirrors SimCSE_training.py
to make configuration and logging comparable.

How TSDAE is built here:
- DenoisingAutoEncoderDataset applies noise to inputs.
- The model encodes noisy inputs and decodes to reconstruct the original.
- CLS pooling is used for sentence embeddings (recommended for TSDAE).

Usage:
    python TSDAE_training.py \
        --base-model path/to/pretrained-model \
        --corpus-path data/corpus.txt \
        --output-path models/tsdae

The corpus must be a UTF-8 TXT file with one sentence per line. The trained
SentenceTransformer model and a JSON training summary are written to the
directory passed with --output-path. Use --device cpu when a CUDA GPU is not
available, and use --auto-config/--no-auto-config to control runtime defaults.
"""

import argparse
import json
import logging
import os
import random
import unicodedata
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer, datasets, losses, models
from torch.utils.data import DataLoader


class MaskedDenoisingAutoEncoderLoss(losses.DenoisingAutoEncoderLoss):
    """TSDAE loss variant that forwards decoder attention masks explicitly."""

    def forward(self, sentence_features, labels):
        source_features, target_features = tuple(sentence_features)
        if self.need_retokenization:
            target_features = self.retokenize(target_features)

        reps = self.encoder(source_features)["sentence_embedding"]
        target_length = target_features["input_ids"].shape[1]

        decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        decoder_attention_mask = target_features["attention_mask"][:, : target_length - 1]
        label_ids = target_features["input_ids"][:, 1:]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=reps[:, None],
            encoder_attention_mask=source_features["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        lm_logits = decoder_outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        return loss

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _get_base_runtime_config() -> dict:
    return {
        "batch_size": 8,
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": None,
        "persistent_workers": False,
        "learning_rate": 3e-5,
        "epochs": 1,
        "use_amp": False,
        "max_seq_length": 256,
    }


def _get_training_runtime_config(device: str) -> dict:
    """Returns runtime defaults tuned for modern NVIDIA laptop GPUs."""
    config = _get_base_runtime_config()

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        cpu_workers = max(2, min(8, (os.cpu_count() or 4) - 1))

        # RTX PRO 4000 Blackwell laptop class cards with 16GB VRAM.
        batch_size = config["batch_size"]
        max_seq_length = config["max_seq_length"]
        if vram_gb >= 15:
            batch_size = 16
            max_seq_length = 512

        config.update(
            {
                "batch_size": batch_size,
                "num_workers": cpu_workers,
                "pin_memory": True,
                "prefetch_factor": 4,
                "persistent_workers": True,
                "learning_rate": 2e-5,
                "epochs": 1,
                "use_amp": True,
                "max_seq_length": max_seq_length,
            }
        )

        # Good throughput defaults on Ada/Blackwell-generation NVIDIA GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    return config


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_runtime_config(args: argparse.Namespace, device: str) -> dict:
    config = _get_training_runtime_config(device) if args.auto_config else _get_base_runtime_config()
    overrides = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "use_amp": args.use_amp,
        "max_seq_length": args.max_seq_length,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def read_corpus_lines(
    path: str,
    min_chars: int,
    lowercase: bool,
    deduplicate: bool,
    max_sentences: Optional[int],
) -> List[str]:
    """Load one sentence per line from TXT and apply simple cleaning."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: {path}")

    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if lowercase:
                text = text.lower()
            if len(text) < min_chars:
                continue
            if text:
                sentences.append(text)
            if max_sentences is not None and len(sentences) >= max_sentences:
                break

    if deduplicate:
        seen = set()
        deduped: List[str] = []
        for s in sentences:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)
        sentences = deduped

    if len(sentences) < 2:
        raise ValueError("Need at least 2 valid corpus sentences after filtering.")

    logger.info("Loaded %d corpus sentences from %s", len(sentences), path)
    return sentences


# ── Greek text normalization ─────────────────────────────────────────────────
# Two Ancient Greek encoder families need opposite preprocessing (verified by
# inspecting their tokenizers):
#   * 'greberta' (bowphs/GreBerta, *PhilBerta — byte-level BPE, no normalizer):
#       case-sensitive and accent-aware, trained on precomposed text. Only apply
#       NFC + apostrophe unification; keep case and accents.
#   * 'agbert' (pranaydeeps/Ancient-Greek-BERT — WordPiece + BertNormalizer):
#       the tokenizer lowercases and strips accents internally, and the koronis
#       elision marks (U+1FBD/U+1FBF/U+02BC) are OOV -> [UNK]. We mirror it by
#       deaccenting + lowercasing and unifying apostrophes.

_CANON_APOSTROPHE = "’"  # ’ — in-vocab for Ancient-Greek-BERT, safe for BPE
_APOSTROPHE_VARIANTS = "'‘’ʼʹ᾽᾿`´"


def _unify_apostrophes(text: str) -> str:
    for ch in _APOSTROPHE_VARIANTS:
        if ch != _CANON_APOSTROPHE:
            text = text.replace(ch, _CANON_APOSTROPHE)
    return text


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_greek(text: str, target: str) -> str:
    """Normalize one Greek line for the target encoder family ('greberta'/'agbert')."""
    text = unicodedata.normalize("NFC", text)  # compose accents/breathings onto letters
    text = _unify_apostrophes(text)            # spacing elision marks -> canonical
    if target == "agbert":
        # mirror BertNormalizer: deaccent + lowercase, re-compose to NFC
        text = unicodedata.normalize("NFC", _strip_accents(text).lower())
    elif target != "greberta":
        raise ValueError(f"Unknown Greek normalization target: {target!r}")
    return text


def detect_greek_target(base_model: str) -> Optional[str]:
    """Infer the Greek normalization family from the base model id/path.

    Returns 'greberta', 'agbert', or None when the model is not a known Ancient
    Greek encoder (e.g. Latin models -> no Greek normalization).
    """
    name = base_model.lower()
    if "ancient-greek-bert" in name or "greek-bert" in name:
        return "agbert"
    if "greberta" in name or "philberta" in name:
        return "greberta"
    return None


def resolve_greek_target(mode: str, base_model: str) -> Optional[str]:
    """Resolve the --greek-normalization choice into a concrete target or None."""
    if mode == "none":
        return None
    if mode in ("greberta", "agbert"):
        return mode
    # mode == "auto"
    target = detect_greek_target(base_model)
    if target:
        logger.info("Auto-detected Greek normalization '%s' for base model %s", target, base_model)
    else:
        logger.info("No Greek normalization (base model %s not recognized as Greek).", base_model)
    return target


def apply_greek_normalization(
    sentences: List[str], target: str, deduplicate: bool
) -> List[str]:
    """Normalize every corpus line for ``target`` and optionally re-deduplicate.

    Re-dedup is needed because normalization (notably 'agbert' deaccenting) can
    collapse previously distinct lines into identical ones.
    """
    before = len(sentences)
    normalized = [normalize_greek(s, target) for s in sentences]

    if deduplicate:
        seen = set()
        deduped: List[str] = []
        for s in normalized:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)
        normalized = deduped

    logger.info(
        "Applied '%s' Greek normalization: %d -> %d sentences", target, before, len(normalized)
    )
    return normalized


def build_model(base_model: str, max_seq_length: int, device: str) -> SentenceTransformer:
    """Build TSDAE encoder with CLS pooling."""
    logger.info("Loading base transformer from: %s", base_model)
    word_embedding_model = models.Transformer(base_model, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)


def build_loss(
    model: SentenceTransformer,
    decoder_name_or_path: str,
    tie_encoder_decoder: bool,
) -> MaskedDenoisingAutoEncoderLoss:
    return MaskedDenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=decoder_name_or_path,
        tie_encoder_decoder=tie_encoder_decoder,
    )


def build_dataloader(train_sentences: List[str], runtime: dict) -> DataLoader:
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    dataloader_kwargs = {
        "batch_size": runtime["batch_size"],
        "shuffle": True,
        "num_workers": runtime["num_workers"],
    }
    if runtime["num_workers"] > 0:
        dataloader_kwargs["pin_memory"] = runtime["pin_memory"]
        dataloader_kwargs["persistent_workers"] = runtime["persistent_workers"]
        if runtime["prefetch_factor"] is not None:
            dataloader_kwargs["prefetch_factor"] = runtime["prefetch_factor"]
    return DataLoader(train_dataset, **dataloader_kwargs)


def write_training_summary(
    output_path: str,
    args: argparse.Namespace,
    runtime: dict,
    n_train_sentences: int,
    steps_per_epoch: int,
    warmup_steps: int,
    device: str,
    greek_target: Optional[str],
) -> None:
    payload = {
        "base_model": args.base_model,
        "corpus_path": args.corpus_path,
        "output_path": args.output_path,
        "epochs": runtime["epochs"],
        "batch_size": runtime["batch_size"],
        "learning_rate": runtime["learning_rate"],
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": runtime["max_seq_length"],
        "seed": args.seed,
        "device": device,
        "scheduler": args.scheduler,
        "use_amp": runtime["use_amp"],
        "num_workers": runtime["num_workers"],
        "pin_memory": runtime["pin_memory"],
        "prefetch_factor": runtime["prefetch_factor"],
        "persistent_workers": runtime["persistent_workers"],
        "auto_config": args.auto_config,
        "num_train_sentences": n_train_sentences,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "min_chars": args.min_chars,
        "lowercase": args.lowercase,
        "keep_duplicates": args.keep_duplicates,
        "max_sentences": args.max_sentences,
        "greek_normalization": args.greek_normalization,
        "greek_normalization_target": greek_target,
        "loss": "MaskedDenoisingAutoEncoderLoss",
        "tie_encoder_decoder": args.tie_encoder_decoder,
    }

    out_file = os.path.join(output_path, "tsdae_training_log.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_tsdae(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but no GPU is available. "
            "Ensure you run in a CUDA-enabled environment."
        )

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Using GPU: %s", gpu_name)

    runtime = resolve_runtime_config(args, device)
    logger.info(
        "Runtime config | batch_size=%s num_workers=%s use_amp=%s max_seq_length=%s lr=%s",
        runtime["batch_size"],
        runtime["num_workers"],
        runtime["use_amp"],
        runtime["max_seq_length"],
        runtime["learning_rate"],
    )

    train_sentences = read_corpus_lines(
        path=args.corpus_path,
        min_chars=args.min_chars,
        lowercase=args.lowercase,
        deduplicate=not args.keep_duplicates,
        max_sentences=args.max_sentences,
    )

    greek_target = resolve_greek_target(args.greek_normalization, args.base_model)
    if greek_target:
        if greek_target == "greberta" and args.lowercase:
            logger.warning(
                "--lowercase is set but '%s' is a case-sensitive encoder; "
                "lowercasing will hurt this model.",
                greek_target,
            )
        train_sentences = apply_greek_normalization(
            train_sentences, target=greek_target, deduplicate=not args.keep_duplicates
        )

    train_dataloader = build_dataloader(train_sentences, runtime)
    if len(train_dataloader) == 0:
        raise ValueError(
            "Training dataloader is empty. Decrease --batch-size or provide more training lines."
        )

    model = build_model(args.base_model, runtime["max_seq_length"], device)
    train_loss = build_loss(
        model=model,
        decoder_name_or_path=args.base_model,
        tie_encoder_decoder=args.tie_encoder_decoder,
    )

    warmup_steps = int(len(train_dataloader) * runtime["epochs"] * args.warmup_ratio)
    logger.info(
        "Training config: epochs=%d, batch_size=%d, warmup_steps=%d, steps/epoch=%d",
        runtime["epochs"],
        runtime["batch_size"],
        warmup_steps,
        len(train_dataloader),
    )

    os.makedirs(args.output_path, exist_ok=True)

    logger.info("Starting TSDAE training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=runtime["epochs"],
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        optimizer_params={"lr": runtime["learning_rate"]},
        warmup_steps=warmup_steps,
        use_amp=runtime["use_amp"],
        show_progress_bar=True,
        output_path=args.output_path,
    )

    logger.info("Training complete. Model saved to %s", args.output_path)

    write_training_summary(
        output_path=args.output_path,
        args=args,
        runtime=runtime,
        n_train_sentences=len(train_sentences),
        steps_per_epoch=len(train_dataloader),
        warmup_steps=warmup_steps,
        device=device,
        greek_target=greek_target,
    )


def train_tsdae_adaptation(
    base_model_name: str = "models/PatriBERT-lat-II_e8_lr5e-05_bs16_acc8",
    corpus_path: str = "preprocessed_corpus/SC_corpus.txt",
    output_path: str = "models/PatriBERT-tsdae",
    device: str = "cuda",
    max_sentences: Optional[int] = None,
) -> None:
    args = argparse.Namespace(
        base_model=base_model_name,
        corpus_path=corpus_path,
        output_path=output_path,
        device=device,
        max_seq_length=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        weight_decay=0.0,
        warmup_ratio=0.0,
        scheduler="constantlr",
        seed=42,
        min_chars=3,
        lowercase=False,
        keep_duplicates=False,
        max_sentences=max_sentences,
        greek_normalization="auto",
        auto_config=True,
        num_workers=None,
        pin_memory=None,
        prefetch_factor=None,
        persistent_workers=None,
        use_amp=None,
        tie_encoder_decoder=True,
    )
    train_tsdae(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TSDAE from a text corpus (one sentence per line)."
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="models/PatriBERT-lat-II_e8_lr5e-05_bs16_acc8",
        help="Path or HF id of pretrained model",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="preprocessed_corpus/SC_corpus.txt",
        help="Path to TXT corpus (one sentence per line)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/PatriBERT-tsdae",
        help="Directory to save trained model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="constantlr")

    parser.add_argument("--min-chars", type=int, default=3, help="Drop very short lines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument(
        "--greek-normalization",
        choices=["auto", "greberta", "agbert", "none"],
        default="auto",
        help=(
            "Greek text normalization: 'auto' detects the family from --base-model; "
            "'greberta'/'agbert' force one; 'none' disables it (use for Latin models)."
        ),
    )
    parser.add_argument("--lowercase", action="store_true", help="Lowercase all corpus lines")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate lines (default behavior is deduplication)",
    )

    parser.add_argument(
        "--auto-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GPU-tuned defaults when available",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--tie-encoder-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()
    train_tsdae(args)