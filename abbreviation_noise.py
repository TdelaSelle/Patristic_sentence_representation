"""
Medieval-Latin abbreviation noise model.

Models the systematic abbreviation conventions of medieval Latin scribes.

Categories modelled (in order of application):

  1. NOMINA SACRA — the oldest and most stable abbreviation class, in
     uninterrupted use from late antiquity through the medieval period.
     Christian-vocabulary contractions with overline (here represented
     without the overline because plain text):
        deus -> ds, dei -> di, deum -> dm, domini -> dni, dominus -> dns,
        spiritus -> sps, sanctus -> scs, christus -> xps, iesus -> ihs ...

  2. STANDARD CONTRACTIONS — common high-frequency words abbreviated by
     dropping medial letters:
        noster -> nr, autem -> aut, etiam -> etia, est -> e, eius -> ei9,
        propter -> ppt, secundum -> scdm ...

  3. SUSPENSIONS — final letters dropped (typically -us, -is, -um, -em, -nt):
        bonus -> bon, omnis -> oms, omnium -> omim, gloriam -> gloria etc.
     The most common is the final -m/-n nasal suspension (a macron in
     manuscripts; here represented as a missing letter).

  4. TIRONIAN NOTES & SPECIAL MARKS — the Tironian et (7-shape) and the
     enclitic -que abbreviation (q; / q3):
        et   -> 7      (only some occurrences; not all)
        -que -> q;
        -bus -> b;
        per  -> p- (per with crossed stem; rendered "p~")
        pro  -> p-/  ; here "pp"  (we use a distinct token)

  5. SYNTAGM ABBREVIATIONS — frequent multi-word collocations:
        dominus noster      -> dns nr
        spiritus sanctus    -> sps scs
        dei filius          -> di fl
        iesus christus      -> ihs xps
        in nomine domini    -> in noie dni

This is a forward (clean -> abbreviated) noise model. Application is
stochastic: not every occurrence of every abbreviable form is abbreviated,
so abbreviated and full forms coexist within the same text (as in real
manuscripts where abbreviation density varies by scribe and book).
"""

import csv
import random
import re
import statistics


# ===== 1. Nomina sacra (very high frequency in patristic Latin) =====
# These are the most abbreviated forms in any Christian Latin manuscript.
NOMINA_SACRA = [
    # (full form regex, abbreviated form)
    (r'\bdeus\b',        'ds'),
    (r'\bdei\b',         'di'),
    (r'\bdeum\b',        'dm'),
    (r'\bdeo\b',         'do'),
    (r'\bdominus\b',     'dns'),
    (r'\bdomini\b',      'dni'),
    (r'\bdomino\b',      'dno'),
    (r'\bdominum\b',     'dnm'),
    (r'\bdomine\b',      'dne'),
    (r'\bdominorum\b',   'dnoz'),
    (r'\bspiritus\b',    'sps'),
    (r'\bspiritu\b',     'spu'),
    (r'\bspiritum\b',    'spm'),
    (r'\bspiritui\b',    'spui'),
    (r'\bsanctus\b',     'scs'),
    (r'\bsancti\b',      'sci'),
    (r'\bsancto\b',      'sco'),
    (r'\bsancta\b',      'sca'),
    (r'\bsancte\b',      'sce'),
    (r'\bsanctam\b',     'scam'),
    (r'\bsanctum\b',     'scm'),
    (r'\bchristus\b',    'xps'),
    (r'\bchristi\b',     'xpi'),
    (r'\bchristo\b',     'xpo'),
    (r'\bchristum\b',    'xpm'),
    (r'\biesus\b',       'ihs'),
    (r'\biesu\b',        'ihu'),
    (r'\biesum\b',       'ihm'),
    (r'\bpater\b',       'pat'),
    (r'\bpatris\b',      'ptris'),
    (r'\bpatrem\b',      'ptrem'),
    (r'\bpatri\b',       'ptri'),
    (r'\bpatre\b',       'ptre'),
    (r'\bfilius\b',      'fls'),
    (r'\bfilii\b',       'fli'),
    (r'\bfilio\b',       'flo'),
    (r'\bfilium\b',      'flm'),
    (r'\bfilium\b',      'flm'),
    (r'\bnoster\b',      'nr'),
    (r'\bnostri\b',      'nri'),
    (r'\bnostrum\b',     'nrm'),
    (r'\bnostra\b',      'nra'),
    (r'\bnostro\b',      'nro'),
    (r'\bnostros\b',     'nros'),
    (r'\bsalvator\b',    'salu'),
    (r'\bsalvatoris\b',  'saluis'),
    (r'\bisrael\b',      'isrl'),
    (r'\bisrahel\b',     'isrl'),
    (r'\bhierusalem\b',  'iherl'),
]

# ===== 2. Standard high-frequency contractions =====
# Cf. Cappelli §§ on word-internal contractions; Lindsay's "common" set.
CONTRACTIONS = [
    (r'\bautem\b',       'aut'),
    (r'\benim\b',        'em'),
    (r'\bergo\b',        'g'),         # very stylised in MSS, here "g"
    (r'\bigitur\b',      'igr'),
    (r'\bquod\b',        'qd'),
    (r'\bquae\b',        'qe'),
    (r'\bqui\b',         'q'),
    (r'\bquam\b',        'qm'),
    (r'\bquia\b',        'qa'),
    (r'\bquoniam\b',     'qm'),
    (r'\bquoque\b',      'qq'),
    (r'\bquondam\b',     'qdm'),
    (r'\bquemadmodum\b', 'qmdm'),
    (r'\bcum\b',         'cu'),
    (r'\btantum\b',      'tm'),
    (r'\btamen\b',       'tn'),
    (r'\bnon\b',         'n'),         # only some occurrences (see rate)
    (r'\bnec\b',         'n,'),
    (r'\bhabet\b',       'h~'),
    (r'\bhabere\b',      'here'),
    (r'\bsicut\b',       'sic'),
    (r'\bsuper\b',       'sup'),
    (r'\bsupra\b',       'spa'),
    (r'\binter\b',       'int'),
    (r'\bpropter\b',     'ppt'),
    (r'\bproprius\b',    'pprius'),
    (r'\bproprie\b',     'pprie'),
    (r'\bproprium\b',    'pprium'),
    (r'\bsecundum\b',    'scdm'),
    (r'\bprimus\b',      'pmus'),
    (r'\bprima\b',       'pma'),
    (r'\bprimum\b',      'pmum'),
    (r'\bsecundus\b',    'sdus'),
    (r'\bgloria\b',      'gla'),
    (r'\bgloriam\b',     'glam'),
    (r'\bgloriae\b',     'glae'),
    (r'\becclesia\b',    'eccla'),
    (r'\becclesiae\b',   'ecclae'),
    (r'\bsalus\b',       'sal'),
    (r'\bsalutis\b',     'sals'),
    (r'\banima\b',       'aia'),
    (r'\banimam\b',      'aiam'),
    (r'\banimae\b',      'aie'),
    (r'\bomnis\b',       'oms'),
    (r'\bomnium\b',      'omim'),
    (r'\bomnes\b',       'oes'),
    (r'\bomnia\b',       'oia'),
    (r'\bomne\b',        'oe'),
    (r'\bomnem\b',       'oem'),
    (r'\beius\b',        'ei9'),       # 9 = -us suffix sign
    (r'\beorum\b',       'eoz'),       # z = -orum suffix sign
    (r'\beius\b',        'ei9'),
    (r'\billius\b',      'illi9'),
    (r'\bipsius\b',      'ipi9'),
    (r'\bcuius\b',       'ci9'),
    (r'\bhuius\b',       'hi9'),
    (r'\bunius\b',       'uni9'),
    (r'\btotius\b',      'toti9'),
    (r'\bsapientia\b',   'sapia'),
    (r'\bsapientiae\b',  'sapiae'),
    (r'\baeternus\b',    'etn9'),
    (r'\baeternam\b',    'etna'),
    (r'\baeternae\b',    'etne'),
    (r'\baeterna\b',     'etna'),
    (r'\baeterni\b',     'etni'),
    (r'\baeterno\b',     'etno'),
    (r'\baeternum\b',    'etnm'),
    (r'\bueritas\b',     'uerit'),
    (r'\bueritatem\b',   'ueritem'),
    (r'\buirtus\b',      'uirt'),
    (r'\buirtutem\b',    'uirtem'),
    (r'\buirtute\b',     'uirte'),
    (r'\bmisericordia\b','mia'),
    (r'\bcaelum\b',      'celu'),
    (r'\bcaeli\b',       'celi'),
    (r'\bcaelo\b',       'celo'),
]

# ===== 3. Suffix / suspension abbreviations =====
# These apply at the END of words: final nasal (-m, -n), -us, -is, -tur, -ntur.
# Implemented as character-level substitutions on word endings.
# In manuscripts these are written as macrons, tildes, or special signs;
# here represented by removing the letter (the macron itself is dropped).

# Word-final -m: very common nasal suspension (above the preceding vowel)
SUSPEND_FINAL_M = re.compile(r'\b(\w*[aeiou])m\b', re.IGNORECASE)
# Word-final -us: replace with 9 (Tironian sign)
SUSPEND_FINAL_US = re.compile(r'\b(\w{3,})us\b', re.IGNORECASE)
# Word-final -orum / -arum: replace with z / r marks
SUSPEND_ORUM = re.compile(r'\b(\w+)orum\b', re.IGNORECASE)
SUSPEND_ARUM = re.compile(r'\b(\w+)arum\b', re.IGNORECASE)
# Word-final -tur / -ntur (passive endings)
SUSPEND_NTUR = re.compile(r'\b(\w+)ntur\b', re.IGNORECASE)
SUSPEND_TUR  = re.compile(r'\b(\w+)tur\b',  re.IGNORECASE)
# Word-final -bus
SUSPEND_BUS  = re.compile(r'\b(\w+)bus\b',  re.IGNORECASE)
# Word-final -que enclitic
SUSPEND_QUE  = re.compile(r'(\w{2,})que\b', re.IGNORECASE)

# ===== 4. Per-/Pro-/Prae- prefix signs =====
# These prefixes are written with a special crossed-stem or hooked sign
# in manuscripts. Here represented as 'p-' / 'pp-' tokens.
PREFIX_PER  = re.compile(r'\bper(\w+)', re.IGNORECASE)
PREFIX_PRO  = re.compile(r'\bpro(\w+)', re.IGNORECASE)
PREFIX_PRAE = re.compile(r'\bprae(\w+)', re.IGNORECASE)

# ===== 5. Multi-word syntagms =====
# These are applied BEFORE single-word abbreviations so they win the race.
SYNTAGMS = [
    (r'\bspiritus sanctus\b',    'sps scs'),
    (r'\bspiritus sancti\b',     'sps sci'),
    (r'\bspiritu sancto\b',      'spu sco'),
    (r'\bspiritum sanctum\b',    'spm scm'),
    (r'\bdei filius\b',          'di fls'),
    (r'\bdei filii\b',           'di fli'),
    (r'\bfilius dei\b',          'fls di'),
    (r'\bfilii dei\b',           'fli di'),
    (r'\bfilio dei\b',           'flo do'),
    (r'\biesus christus\b',      'ihs xps'),
    (r'\biesu christo\b',        'ihu xpo'),
    (r'\biesu christi\b',        'ihu xpi'),
    (r'\bdominus noster\b',      'dns nr'),
    (r'\bdomini nostri\b',       'dni nri'),
    (r'\bdomino nostro\b',       'dno nro'),
    (r'\bdeus noster\b',         'ds nr'),
    (r'\bdei nostri\b',          'di nri'),
    (r'\bin nomine domini\b',    'in noie dni'),
    (r'\bin nomine dei\b',       'in noie di'),
    (r'\bgloria patri\b',        'gla ptri'),
    (r'\bin saecula saeculorum\b','in scla scloz'),
    (r'\bpater noster\b',        'pat nr'),
    (r'\bregnum dei\b',          'regnm di'),
    (r'\bregni dei\b',           'regni di'),
    (r'\bregno dei\b',           'regno do'),
    (r'\bsanctus spiritus\b',    'scs sps'),
    (r'\bsancti spiritus\b',     'sci sps'),
    (r'\bcaelum et terra\b',     'celu 7 terra'),
    (r'\bcaeli et terrae\b',     'celi 7 terre'),
]


# Compile all patterns once
NOMINA_SACRA_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in NOMINA_SACRA]
CONTRACTIONS_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in CONTRACTIONS]
SYNTAGMS_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in SYNTAGMS]


def _stochastic_sub(pattern_compiled, repl, text, prob, rng):
    """Replace each match of `pattern_compiled` with `repl` with probability `prob`."""
    def _swap(m):
        return repl if rng.random() < prob else m.group(0)
    return pattern_compiled.sub(_swap, text)


def _stochastic_sub_callable(pattern_compiled, repl_func, text, prob, rng):
    """Like _stochastic_sub but repl is a function of match -> str."""
    def _swap(m):
        return repl_func(m) if rng.random() < prob else m.group(0)
    return pattern_compiled.sub(_swap, text)


def apply_abbreviation_noise(text: str,
                              p_nomina_sacra: float = 0.85,
                              p_contraction:  float = 0.55,
                              p_syntagm:      float = 0.75,
                              p_final_m:      float = 0.30,
                              p_final_us:     float = 0.20,
                              p_orum:         float = 0.70,
                              p_arum:         float = 0.50,
                              p_ntur:         float = 0.35,
                              p_tur:          float = 0.20,
                              p_bus:          float = 0.65,
                              p_que:          float = 0.50,
                              p_prefix_per:   float = 0.35,
                              p_prefix_pro:   float = 0.30,
                              p_prefix_prae:  float = 0.40,
                              p_tironian_et:  float = 0.30,
                              seed: int | None = None) -> str:
    """
    Apply medieval-Latin abbreviation conventions stochastically.

    Per-class abbreviation rates calibrated so the OVERALL token-abbreviation
    rate is ~25-35% (typical of a moderately abbreviated Caroline minuscule
    hand). To simulate Gothic-period heavy abbreviation, multiply all rates
    by ~1.3-1.5.

    Parameters
    ----------
    p_nomina_sacra : abbreviation probability for nomina sacra words
        (deus, dominus, christus, iesus, sanctus, spiritus, etc.). Default
        0.85 reflects the near-universal abbreviation of these forms.
    p_contraction : probability for standard medial contractions.
    p_syntagm : probability for multi-word syntagms.
    p_final_m, p_final_us, ... : probabilities for suspension classes.
    p_tironian_et : probability for "et" -> "7" (Tironian).

    Returns
    -------
    Abbreviated text (string).
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    out = text

    # Stage 1: SYNTAGMS (must run first to win the race against single-word subs)
    for pat, repl in SYNTAGMS_RE:
        out = _stochastic_sub(pat, repl, out, p_syntagm, rng)

    # Stage 2: NOMINA SACRA
    for pat, repl in NOMINA_SACRA_RE:
        out = _stochastic_sub(pat, repl, out, p_nomina_sacra, rng)

    # Stage 3: STANDARD CONTRACTIONS (high-frequency words)
    for pat, repl in CONTRACTIONS_RE:
        out = _stochastic_sub(pat, repl, out, p_contraction, rng)

    # Stage 4: PREFIXES per-/pro-/prae-
    out = _stochastic_sub_callable(
        PREFIX_PER,  lambda m: 'p~' + m.group(1),  out, p_prefix_per, rng)
    out = _stochastic_sub_callable(
        PREFIX_PRO,  lambda m: 'pp~' + m.group(1), out, p_prefix_pro, rng)
    out = _stochastic_sub_callable(
        PREFIX_PRAE, lambda m: 'pre' + m.group(1), out, p_prefix_prae, rng)

    # Stage 5: SUSPENSIONS / final-letter suspensions
    # -orum / -arum first (more specific)
    out = _stochastic_sub_callable(
        SUSPEND_ORUM, lambda m: m.group(1) + 'oz',  out, p_orum, rng)
    out = _stochastic_sub_callable(
        SUSPEND_ARUM, lambda m: m.group(1) + 'arz', out, p_arum, rng)
    # -ntur / -tur
    out = _stochastic_sub_callable(
        SUSPEND_NTUR, lambda m: m.group(1) + 'nr',  out, p_ntur, rng)
    out = _stochastic_sub_callable(
        SUSPEND_TUR,  lambda m: m.group(1) + 't~',  out, p_tur, rng)
    # -bus / -que enclitic
    out = _stochastic_sub_callable(
        SUSPEND_BUS,  lambda m: m.group(1) + 'b;',  out, p_bus, rng)
    out = _stochastic_sub_callable(
        SUSPEND_QUE,  lambda m: m.group(1) + 'q;',  out, p_que, rng)
    # Final -us (Tironian 9): only on longer words (≥3 chars before -us) to
    # avoid abbreviating very short words like "eus" which doesn't exist anyway
    out = _stochastic_sub_callable(
        SUSPEND_FINAL_US, lambda m: m.group(1) + '9', out, p_final_us, rng)
    # Final -m: very common nasal suspension. Replace -Vm -> -V~ (macron).
    out = _stochastic_sub_callable(
        SUSPEND_FINAL_M, lambda m: m.group(1) + '~', out, p_final_m, rng)

    # Stage 6: Tironian et
    out = _stochastic_sub(re.compile(r'\bet\b', re.IGNORECASE),
                          '7', out, p_tironian_et, rng)

    return out


def count_abbreviation_density(orig: str, abbr: str) -> dict:
    """Diagnostic: how 'abbreviated' is the result vs. the original?"""
    orig_tokens = re.findall(r"\b\w+\b", orig)
    abbr_tokens = re.findall(r"\b\S+\b", abbr)
    return {
        'orig_chars':       len(orig),
        'abbr_chars':       len(abbr),
        'char_reduction':   1.0 - (len(abbr) / max(1, len(orig))),
        'orig_tokens':      len(orig_tokens),
        'abbr_tokens':      len(abbr_tokens),
    }


if __name__ == '__main__':
    INPUT  = 'Aug_trin.tsv'
    OUTPUT = 'Aug_trin_abbr.tsv'

    with open(INPUT, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))

    SEED = 20251201
    char_reductions = []
    out_rows = []
    for i, r in enumerate(rows):
        clean = r['text']
        abbreviated = apply_abbreviation_noise(clean, seed=SEED + i)
        stats = count_abbreviation_density(clean, abbreviated)
        char_reductions.append(stats['char_reduction'])
        out_rows.append({
            'text_clean':      clean,
            'text_abbr':       abbreviated,
            'char_reduction':  f"{stats['char_reduction']:.4f}",
        })

    # Empirical stats
    print(f"Sentences:                  {len(out_rows):,}")
    print(f"Mean character reduction:   {statistics.mean(char_reductions):.4f}")
    print(f"Median character reduction: {statistics.median(char_reductions):.4f}")
    print(f"Quartiles: q1={sorted(char_reductions)[len(char_reductions)//4]:.4f}  "
          f"q3={sorted(char_reductions)[3*len(char_reductions)//4]:.4f}")

    # Write TSV
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write('text_clean\ttext_abbr\tchar_reduction\n')
        for r in out_rows:
            t1 = r['text_clean'].replace('\t', ' ').replace('\n', ' ')
            t2 = r['text_abbr'].replace('\t', ' ').replace('\n', ' ')
            f.write(f"{t1}\t{t2}\t{r['char_reduction']}\n")
    print(f"\nWritten: {OUTPUT}")

    # Samples
    print("\nSamples (clean -> abbreviated):")
    for idx in [50, 200, 500, 1000, 2000, 3000, 4000, 5000]:
        if idx < len(out_rows):
            print(f"\n[{idx}] reduction={out_rows[idx]['char_reduction']}")
            print(f"  CLEAN: {out_rows[idx]['text_clean'][:180]}")
            print(f"  ABBR : {out_rows[idx]['text_abbr'][:180]}")
