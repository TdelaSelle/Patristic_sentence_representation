"""
HTR (Handwritten Text Recognition) noise model for medieval Latin texts.

Models realistic character-level errors observed in HTR systems applied to
Latin manuscripts.

Default target CER here = 8% (realistic out-of-domain, lightly abbreviated).

Error types modeled (frequencies relative to total errors):
  (a) Single-character substitution (paleographic confusion)  ~70%
  (b) Multi-character substitution (e.g. m -> in, in -> m)    ~15%
  (c) Character deletion                                       ~7%
  (d) Character insertion                                      ~4%
  (e) Word-boundary error (space drop / add)                   ~3%
  (f) Abbreviation expansion error                             ~1%

Note: this is a *forward* noise model (clean -> noisy). It does NOT model
all real HTR error patterns (e.g. line-end hyphenation, signed-letter
preservation, capitalization quirks). For Latin in classical orthography
(u for v, no j, lowercase) it is a reasonable first-order approximation.
"""

import random
import re
import csv


# Single-character paleographic confusions. Each char maps to a list of
# plausible misreadings, weighted by approximate frequency in the literature.
# (Weights implicit via repetition — first item is most common confusion.)
CONFUSION_MATRIX = {
    # Minim confusion — by far the dominant HTR error in Latin minuscule
    'n': ['u', 'ii', 'ri', 'u', 'u'],         # n is 2 minims, easily read as u or ii
    'u': ['n', 'ii', 'a', 'n', 'n'],
    'm': ['in', 'ni', 'iu', 'ui', 'nn'],      # m is 3 minims
    'i': ['l', 'r', 'e', 'l'],                # undotted i confused with l
    'l': ['i', 'b', 't', 'i'],

    # Bowl/loop confusions
    'c': ['e', 't', 'o', 'e'],
    'e': ['c', 'o', 'e', 'c'],
    'o': ['a', 'e', 'c', 'o'],
    'a': ['o', 'u', 'd', 'o'],

    # Long-s and r confusions
    's': ['f', 'r', 's'],
    'f': ['s', 'f', 'P'],
    'r': ['n', 'i', 'r', 's'],

    # Ascender/descender confusions
    't': ['c', 'l', 'r', 't'],
    'b': ['h', 'l', 'b'],
    'h': ['b', 'k', 'h'],
    'd': ['cl', 'a', 'd'],
    'p': ['q', 'b', 'p'],
    'q': ['p', 'g', 'q'],
    'g': ['q', 'y', 'g'],

    # Less common
    'x': ['r', 'x'],
    'y': ['g', 'v', 'y'],
}

# Multi-character substitutions (also captures the reverse direction)
MULTI_CHAR_SUBS = [
    ('in', 'm'),    # in -> m (reverse of m -> in)
    ('ni', 'm'),
    ('iu', 'm'),
    ('ui', 'm'),
    ('ii', 'n'),
    ('ii', 'u'),
    ('cl', 'd'),
]

# Abbreviation expansion errors — common in HTR of abbreviated Latin manuscripts
# Direction: clean expanded form -> abbreviation that the HTR fails to expand
ABBREV_RAW = [
    (r'\bque\b',     ['q;', 'q3', 'qz']),
    (r'\bbus\b',     ['b;', 'b3']),
    (r'\beius\b',    ['ei9', 'eius', 'e9']),
    (r'\beorum\b',   ['eor', 'eoz']),
    (r'\bomnes\b',   ['oms', 'omes']),
    (r'\bomnium\b',  ['omim', 'omiu']),
    (r'\bomnia\b',   ['oia', 'omia']),
    (r'\bdeus\b',    ['ds', 'dus']),
    (r'\bdei\b',     ['di', 'di']),
    (r'\bdeum\b',    ['dm', 'deum']),
    (r'\bdomine\b',  ['dne', 'dmine']),
    (r'\bdominus\b', ['dns', 'dms']),
    (r'\bdomini\b',  ['dni', 'dnii']),
    (r'\bspiritus\b',['sps', 'spc', 'sps']),
    (r'\bsanctus\b', ['scs', 'sancs']),
    (r'\bsancti\b',  ['sci', 'sancti']),
    (r'\bchristus\b',['xps', 'xpc']),
    (r'\bchristi\b', ['xpi', 'xpi']),
    (r'\biesus\b',   ['ihs', 'ihc']),
    (r'\biesu\b',    ['ihu', 'ihu']),
    (r'\bnoster\b',  ['nr', 'nost']),
    (r'\bnostrum\b', ['nrm', 'nstr']),
    (r'\bnostri\b',  ['nri', 'nstri']),
    (r'\bautem\b',   ['aut', 'aute']),
    (r'\benim\b',    ['en', 'em']),
    (r'\bquod\b',    ['qd', 'qd']),
    (r'\bquia\b',    ['qa', 'qa']),
    (r'\bqui\b',     ['q', 'qi']),
    (r'\bpater\b',   ['pat', 'patr']),
    (r'\bpatris\b',  ['patris', 'ptris']),
    (r'\bfilius\b',  ['flius', 'fls']),
    (r'\bfilii\b',   ['fli', 'flii']),
]

ABBREV_RE = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ABBREV_RAW]


def apply_htr_noise(text: str, cer: float = 0.08,
                    seed: int | None = None) -> str:
    """
    Apply HTR noise to text with target Character Error Rate ≈ cer.

    The cer parameter controls the per-character corruption probability for
    substitutions. Other error types (delete/insert/boundary/abbrev) scale
    proportionally.

    Note: actual measured CER on output will be slightly above cer because
    multi-char substitutions and insertions add to edit distance.

    Args
    ----
    text : input string (clean Latin)
    cer  : target character error rate (default 0.08 = 8%)
    seed : RNG seed for reproducibility

    Returns
    -------
    Noised string.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Stage 1: stochastic abbreviation introduction
    out = text
    for pattern, repls in ABBREV_RE:
        if rng.random() < cer * 0.15:  # ~1% per match site
            repl = rng.choice(repls)
            out = pattern.sub(repl, out, count=1)

    # Stage 2: character-level noise pass
    chars = list(out)
    i = 0
    result_chars = []
    while i < len(chars):
        ch = chars[i]
        cl = ch.lower()

        # Multi-character substitution attempt: look at 2-char window
        if i + 1 < len(chars):
            bigram = (cl + chars[i+1].lower())
            for src, tgt in MULTI_CHAR_SUBS:
                if bigram == src and rng.random() < cer * 0.10:
                    # apply multi-char sub
                    sub = tgt.upper() if ch.isupper() else tgt
                    result_chars.append(sub)
                    i += 2
                    break
            else:
                # No multi-char sub fired — fall through to single-char
                pass
            if i >= len(chars) or (i > 0 and len(result_chars) > 0
                                    and i != _last_idx_marker(chars, i)):
                # Simplified: if i advanced via multi-char sub, continue
                if len(result_chars) > 0 and result_chars[-1] in [tgt for _, tgt in MULTI_CHAR_SUBS]:
                    continue

        # Single-character substitution
        if cl.isalpha() and cl in CONFUSION_MATRIX:
            if rng.random() < cer:
                sub = rng.choice(CONFUSION_MATRIX[cl])
                if ch.isupper():
                    sub = sub.capitalize()
                result_chars.append(sub)
                i += 1
                continue

        # Character deletion (rare)
        if cl.isalpha() and rng.random() < cer * 0.10:
            i += 1
            continue

        # Character insertion (rare): duplicate a minim-class letter
        if cl in 'inumlcoe' and rng.random() < cer * 0.06:
            result_chars.append(ch)
            result_chars.append(ch)
            i += 1
            continue

        # Word boundary error: drop space (merge) or split a word
        if ch == ' ':
            r = rng.random()
            if r < cer * 0.04:
                # drop space (merge words)
                i += 1
                continue
            elif r < cer * 0.06:
                # split a word: skip space, add space later
                result_chars.append('  ')
                i += 1
                continue

        result_chars.append(ch)
        i += 1

    return ''.join(result_chars)


def _last_idx_marker(chars, i):
    """Helper to detect multi-char advance (sentinel)."""
    return i


def compute_cer(orig: str, noisy: str) -> float:
    """Levenshtein-based Character Error Rate."""
    # Use a memory-efficient Levenshtein
    if not orig:
        return 0.0
    m, n = len(orig), len(noisy)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            if orig[i - 1] == noisy[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev, cur = cur, prev
    return prev[n] / max(1, m)


if __name__ == '__main__':
    # ===== Apply HTR noise to the input CSV =====
    INPUT  = 'preprocessed_corpus/Aug_trin_sentences.txt'
    OUTPUT = 'preprocessed_corpus/Aug_trin_sentences_htr.tsv'

    with open(INPUT, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    SEED = 20251201    # fixed for reproducibility
    TARGET_CER = 0.05

    noised_rows = []
    cers = []
    for i, r in enumerate(rows):
        clean = r['text']
        # Seed deterministically per row so noise is reproducible AND independent
        noisy = apply_htr_noise(clean, cer=TARGET_CER, seed=SEED + i)
        cer = compute_cer(clean, noisy)
        cers.append(cer)
        noised_rows.append({
            'text_clean': clean,
            'text_htr':   noisy,
            'cer':        f"{cer:.4f}",
        })

    # Empirical CER stats
    import statistics
    print(f"Sentences: {len(noised_rows):,}")
    print(f"Target CER: {TARGET_CER}")
    print(f"Achieved mean CER:   {statistics.mean(cers):.4f}")
    print(f"Achieved median CER: {statistics.median(cers):.4f}")
    print(f"CER stdev:           {statistics.stdev(cers):.4f}")
    print(f"Quartiles: q1={sorted(cers)[len(cers)//4]:.4f}  "
          f"q3={sorted(cers)[3*len(cers)//4]:.4f}")

    # Write TSV (raw, no quoting — Latin text contains commas, never tabs)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write('text_clean\ttext_htr\tcer\n')
        for r in noised_rows:
            t1 = r['text_clean'].replace('\t', ' ').replace('\n', ' ')
            t2 = r['text_htr'].replace('\t', ' ').replace('\n', ' ')
            f.write(f"{t1}\t{t2}\t{r['cer']}\n")
    print(f"\nWritten: {OUTPUT}")

    # Sample
    print("\nSamples (clean -> noisy):")
    sample_indices = [50, 100, 500, 1000, 2000, 3000, 4000, 5000]
    for idx in sample_indices:
        if idx < len(noised_rows):
            print(f"\n[{idx}] CER={noised_rows[idx]['cer']}")
            print(f"  CLEAN: {noised_rows[idx]['text_clean'][:140]}")
            print(f"  HTR  : {noised_rows[idx]['text_htr'][:140]}")
