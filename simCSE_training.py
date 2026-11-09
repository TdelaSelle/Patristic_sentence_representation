"""
Step 2.5: Unsupervised SimCSE bridge training from an MLM/DAPT encoder.

This script trains an unsupervised SimCSE-style sentence embedding model from
plain corpus sentences stored as one sentence per line in a TXT file.

How unsupervised SimCSE is built here:
- Each corpus sentence is duplicated as an InputExample(texts=[s, s]).
- Dropout noise in the encoder creates two different views of the same sentence.
- MultipleNegativesRankingLoss uses all other batch items as implicit negatives.

Usage:
    python SimCSE_training.py \
        --base-model path/to/pretrained-model \
        --corpus-path data/corpus.txt \
        --output-path models/simcse

The corpus must be a UTF-8 TXT file with one sentence per line. The trained
SentenceTransformer model and a JSON training summary are written to the
directory passed with --output-path. Use --dev-ratio to reserve a portion of
the corpus for pseudo-pair evaluation and best-model selection.

"""

import argparse
import json
import logging
import os
import random
import unicodedata
from typing import List, Optional, Tuple

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, Dataset


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class InputExampleDataset(Dataset):
    """Tiny dataset wrapper for InputExample objects."""

    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]


def read_corpus_lines(
    path: str,
    min_chars: int,
    lowercase: bool,
    deduplicate: bool,
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

    if deduplicate:
        # Stable deduplication while preserving order.
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
#   * 'agbert' (pranaydeeps/Ancient-Greek-BERT and princeton-logion/logion-bert-base
#       — WordPiece + BertNormalizer(lowercase=True, strip_accents=None)):
#       the tokenizer lowercases and strips accents internally (strip_accents=None
#       falls back to lowercase=True, verified: 'Ἐν ἀρχῇ' -> 'εν αρχη'). Spacing
#       elision marks (e.g. U+1FBD/U+1FBF/U+02BC) tokenize poorly (OOV/[UNK] on
#       Ancient-Greek-BERT, awkward '##' subwords on Logion), while the canonical
#       '’' (U+2019) is in-vocab on both. We mirror this by deaccenting +
#       lowercasing and unifying apostrophes.

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
    # Logion shares Ancient-Greek-BERT's WordPiece + BertNormalizer scheme
    # (lowercase + accent stripping), so it uses the same 'agbert' normalization.
    if "ancient-greek-bert" in name or "greek-bert" in name or "logion" in name:
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


def subsample_sentences(sentences: List[str], max_sentences: int, seed: int) -> List[str]:
    """Randomly keep at most ``max_sentences`` sentences from the corpus.

    Values <= 0 disable subsampling and return all sentences unchanged. When the
    corpus already has fewer sentences than the cap, it is returned as-is.
    """
    if max_sentences <= 0 or len(sentences) <= max_sentences:
        return sentences

    rng = random.Random(seed)
    sampled = rng.sample(sentences, max_sentences)
    logger.info(
        "Subsampled corpus from %d to %d sentences (max_sentences=%d)",
        len(sentences),
        len(sampled),
        max_sentences,
    )
    return sampled


def split_train_dev(sentences: List[str], dev_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Split corpus sentences into train/dev subsets."""
    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError("dev_ratio must be in [0.0, 1.0).")

    if dev_ratio == 0.0:
        return sentences, []

    random.seed(seed)
    shuffled = list(sentences)
    random.shuffle(shuffled)

    n_dev = max(1, int(len(shuffled) * dev_ratio))
    n_dev = min(n_dev, len(shuffled) - 1)

    dev_sentences = shuffled[:n_dev]
    train_sentences = shuffled[n_dev:]
    logger.info("Split sizes: train=%d, dev=%d", len(train_sentences), len(dev_sentences))
    return train_sentences, dev_sentences


def build_model(base_model: str, max_seq_length: int) -> SentenceTransformer:
    """Build SBERT-style encoder from a pretrained transformer checkpoint."""
    logger.info("Loading base transformer from: %s", base_model)
    word_embedding_model = models.Transformer(base_model, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        # pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def make_unsupervised_examples(sentences: List[str]) -> List[InputExample]:
    """Create duplicated sentence pairs expected by SimCSE unsupervised objective."""
    return [InputExample(texts=[s, s]) for s in sentences]


def build_dev_evaluator(
    dev_sentences: List[str],
    seed: int,
    max_pairs: int,
) -> EmbeddingSimilarityEvaluator:
    """Build a lightweight pseudo-labeled evaluator from dev sentences.

    Positives are identical sentence pairs (score=1.0).
    Negatives are random mismatched sentence pairs (score=0.0).
    """
    rng = random.Random(seed)
    positives = [InputExample(texts=[s, s], label=1.0) for s in dev_sentences]

    negatives: List[InputExample] = []
    if len(dev_sentences) > 1:
        for i, s1 in enumerate(dev_sentences):
            j = rng.randrange(len(dev_sentences) - 1)
            if j >= i:
                j += 1
            s2 = dev_sentences[j]
            negatives.append(InputExample(texts=[s1, s2], label=0.0))

    examples = positives + negatives
    if len(examples) > max_pairs:
        examples = rng.sample(examples, max_pairs)

    return EmbeddingSimilarityEvaluator.from_input_examples(examples, name="dev-unsup-simcse")


def build_loss(
    model: SentenceTransformer,
    use_cached_mnrl: bool,
    mini_batch_size: int,
):
    """Build contrastive loss; optionally use cached MNRL for small-memory setups."""
    if use_cached_mnrl:
        cached_cls = getattr(losses, "CachedMultipleNegativesRankingLoss", None)
        if cached_cls is None:
            logger.warning(
                "CachedMultipleNegativesRankingLoss is unavailable in this sentence-transformers version. "
                "Falling back to MultipleNegativesRankingLoss."
            )
            return losses.MultipleNegativesRankingLoss(model=model)
        logger.info("Using CachedMultipleNegativesRankingLoss (mini_batch_size=%d)", mini_batch_size)
        return cached_cls(model=model, mini_batch_size=mini_batch_size)

    return losses.MultipleNegativesRankingLoss(model=model)


def write_training_summary(
    output_path: str,
    args: argparse.Namespace,
    n_train_sentences: int,
    n_dev_sentences: int,
    used_cached_mnrl: bool,
    used_evaluator: bool,
    save_best_model: bool,
    effective_evaluation_steps: int,
    greek_target: Optional[str],
) -> None:
    """Persist a compact machine-readable training summary."""
    payload = {
        "base_model": args.base_model,
        "corpus_path": args.corpus_path,
        "output_path": args.output_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "max_sentences": args.max_sentences,
        "greek_normalization": args.greek_normalization,
        "greek_normalization_target": greek_target,
        "num_train_sentences": n_train_sentences,
        "num_dev_sentences": n_dev_sentences,
        "used_evaluator": used_evaluator,
        "save_best_model": save_best_model,
        "effective_evaluation_steps": effective_evaluation_steps,
        "loss": "CachedMultipleNegativesRankingLoss" if used_cached_mnrl else "MultipleNegativesRankingLoss",
        "cached_mnrl_mini_batch_size": args.cached_mnrl_mini_batch_size if used_cached_mnrl else None,
    }

    out_file = os.path.join(output_path, "unsupervised_simcse_training_log.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_unsupervised_simcse(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    sentences = read_corpus_lines(
        path=args.corpus_path,
        min_chars=args.min_chars,
        lowercase=args.lowercase,
        deduplicate=not args.keep_duplicates,
    )

    greek_target = resolve_greek_target(args.greek_normalization, args.base_model)
    if greek_target:
        if greek_target == "greberta" and args.lowercase:
            logger.warning(
                "--lowercase is set but '%s' is a case-sensitive encoder; "
                "lowercasing will hurt this model.",
                greek_target,
            )
        sentences = apply_greek_normalization(
            sentences, target=greek_target, deduplicate=not args.keep_duplicates
        )

    sentences = subsample_sentences(sentences, max_sentences=args.max_sentences, seed=args.seed)
    train_sentences, dev_sentences = split_train_dev(sentences, dev_ratio=args.dev_ratio, seed=args.seed)

    train_examples = make_unsupervised_examples(train_sentences)
    train_dataloader = DataLoader(
        InputExampleDataset(train_examples),
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )

    if len(train_dataloader) == 0:
        raise ValueError(
            "Training dataloader is empty. Decrease --batch-size or provide more training lines."
        )

    model = build_model(args.base_model, args.max_seq_length)
    train_loss = build_loss(
        model=model,
        use_cached_mnrl=args.use_cached_mnrl,
        mini_batch_size=args.cached_mnrl_mini_batch_size,
    )

    evaluator = None
    if dev_sentences:
        evaluator = build_dev_evaluator(
            dev_sentences=dev_sentences,
            seed=args.seed,
            max_pairs=args.dev_eval_max_pairs,
        )
        logger.info(
            "Built dev evaluator from %d sentences (max_pairs=%d)",
            len(dev_sentences),
            args.dev_eval_max_pairs,
        )

    effective_evaluation_steps = args.evaluation_steps
    save_best_model = evaluator is not None
    if evaluator is not None and effective_evaluation_steps <= 0:
        # Evaluate once per epoch when user leaves evaluation_steps at 0.
        effective_evaluation_steps = max(1, len(train_dataloader))
        logger.info(
            "evaluation_steps was 0; using %d to enable best-model tracking.",
            effective_evaluation_steps,
        )
    elif evaluator is None and args.evaluation_steps > 0:
        logger.warning(
            "evaluation_steps is set but no dev split is available. "
            "Training will save the final model (no best-model selection)."
        )

    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    logger.info(
        "Training config: epochs=%d, batch_size=%d, warmup_steps=%d, steps/epoch=%d",
        args.epochs,
        args.batch_size,
        warmup_steps,
        len(train_dataloader),
    )

    os.makedirs(args.output_path, exist_ok=True)

    # For unsupervised corpus training, we usually skip evaluator unless external
    # labeled dev data is available. We still keep the same fit path for consistency.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        evaluator=evaluator,
        warmup_steps=warmup_steps,
        output_path=args.output_path,
        optimizer_params={"lr": args.learning_rate},
        weight_decay=args.weight_decay,
        evaluation_steps=effective_evaluation_steps,
        save_best_model=save_best_model,
        show_progress_bar=True,
    )

    logger.info("Training complete. Model saved to %s", args.output_path)

    used_cached_mnrl = args.use_cached_mnrl and hasattr(losses, "CachedMultipleNegativesRankingLoss")
    write_training_summary(
        output_path=args.output_path,
        args=args,
        n_train_sentences=len(train_sentences),
        n_dev_sentences=len(dev_sentences),
        used_cached_mnrl=used_cached_mnrl,
        used_evaluator=evaluator is not None,
        save_best_model=save_best_model,
        effective_evaluation_steps=effective_evaluation_steps,
        greek_target=greek_target,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train unsupervised SimCSE from a text corpus (one sentence per line)."
    )

    parser.add_argument("--base-model", type=str, required=True, help="Path or HF id of pretrained model")
    parser.add_argument("--corpus-path", type=str, required=True, help="Path to TXT corpus (one sentence per line)")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save trained model")

    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=0,
        help="Eval interval; with --dev-ratio > 0 and value 0, defaults to once per epoch",
    )

    parser.add_argument("--min-chars", type=int, default=3, help="Drop very short lines")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Randomly keep at most K corpus sentences (<=0 keeps all)",
    )
    parser.add_argument(
        "--greek-normalization",
        choices=["auto", "greberta", "agbert", "none"],
        default="auto",
        help=(
            "Greek text normalization: 'auto' detects the family from --base-model; "
            "'greberta'/'agbert' force one; 'none' disables it (use for Latin models)."
        ),
    )
    parser.add_argument("--dev-ratio", type=float, default=0.0, help="Optional train/dev split ratio")
    parser.add_argument(
        "--dev-eval-max-pairs",
        type=int,
        default=5000,
        help="Max pseudo-labeled dev pairs used for evaluator",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lowercase", action="store_true", help="Lowercase all corpus lines")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate lines (default behavior is deduplication)",
    )

    parser.add_argument(
        "--use-cached-mnrl",
        action="store_true",
        help="Use CachedMultipleNegativesRankingLoss when available",
    )
    parser.add_argument(
        "--cached-mnrl-mini-batch-size",
        type=int,
        default=16,
        help="Mini-batch size used by CachedMultipleNegativesRankingLoss",
    )

    args = parser.parse_args()
    train_unsupervised_simcse(args)
