# Unsupervised Learning of Patristic Sentence Representations for Text Reuse Detection 

Code and data accompanying:

> **From transcription to semantic corpus analysis: unsupervised learning of
> sentence representations for ancient languages**

Paper: [arXiv:2607.24542](https://arxiv.org/abs/2607.24542)

This repository contains the materials used to adapt language models into
sentence encoders for biblical-reuse detection and retrieval in patristic
Latin and Ancient Greek. The experiments compare two unsupervised approaches:
TSDAE (denoising autoencoding) and SimCSE-style contrastive training.

## Repository contents

```text
.
├── TSDAE_training.py                 TSDAE sentence-encoder training
├── simCSE_training.py                unsupervised SimCSE training
├── prepare_corpus.py                 sentence segmentation and normalization
├── abbreviation_noise.py             medieval-Latin abbreviation noise model
├── htr_noise.py                      HTR-style character noise model
├── evaluation_reuses_detection&IR.ipynb
│                                      reuse detection and retrieval evaluation
└── data/
    ├── bibles/                       Vulgate and Greek Bible TSV files
    ├── reuses_test_set/              annotated patristic reuse test sets
    └── training_corpus/              raw sentence-training corpora
```

### Data

Data availability: restricted access. Most of the patristic data underlying this study are proprietary to Brepols Publishers (Library of Latin Texts / BREPOLiS) and were used under licence; they therefore cannot be shared publicly. Equivalent access may be obtained through a Brepols subscription. The biblical reference data used in this study come from the BiblIndex platform ( https://www.biblindex.org/en/bible-parser); although accessible via the BiblIndex interface and API, they are not freely redistributable under the platform's terms of use. Only the open-access data derived from Augustine of Hippo's Contra Faustum (public-domain text) are deposited in this Zenodo repository. 


## Workflow

1. Prepare and normalize sentence corpora with `prepare_corpus.py`.
2. Optionally add realistic medieval abbreviation or HTR noise with
   `abbreviation_noise.py` and `htr_noise.py`.
3. Train a sentence encoder from a pretrained token-level model with TSDAE or
   unsupervised SimCSE.
4. Run `evaluation_reuses_detection&IR.ipynb` to measure:
   - binary text-reuse detection (average precision and ROC/PR curves);
   - top-1 correspondence retrieval in the selected Bible corpus.

## Training

Both training scripts accept a Hugging Face model identifier or local model
directory, a UTF-8 text corpus with one sentence per line, and an output
directory.

```bash
python TSDAE_training.py \
  --base-model path/to/pretrained-model \
  --corpus-path data/training_corpus/ \
  --output-path models/tsdae

python simCSE_training.py \
  --base-model path/to/pretrained-model \
  --corpus-path data/training_corpus/ \
  --output-path models/simcse
```

Use `--help` for the complete set of training, sampling, normalization, and
device options. The scripts use CUDA automatically when available; pass
`--device cpu` when GPU execution is unavailable.

## Environment

The training scripts require Python with PyTorch and
[`sentence-transformers`](https://www.sbert.net/). The evaluation notebook
also uses Jupyter, pandas, NumPy, scikit-learn, matplotlib, and FAISS.
Install versions appropriate for the available hardware and CUDA runtime.

Some preprocessing paths and model configurations are intentionally project-
specific. Review the constants at the top of `prepare_corpus.py` and the
configuration cells in the notebook before running the full pipeline.

## License

The code in this repository is released under the [MIT License](./LICENSE).