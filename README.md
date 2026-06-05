# eval_trust

**Audit your LLM benchmark numbers before someone else does.**

Companion code & data for the paper *"Silent Contamination in LLM Merging
Evaluation: A Case Study from a 5-Month Misadventure."*

---

## Why this exists

We spent **5 months** chasing a **+10 pp** GSM8K improvement on a Qwen2.5-1.5B
merge that survived three rounds of verification, paired McNemar
significance (p = 0.015), and multi-seed runs. Then we noticed
`max_new_tokens` was hardcoded to 300 in our generation runner.

Re-ran with `max_new_tokens=768`. The +10 pp **collapsed to −1.0 pp**
(p = 0.89). The "improvement" was the baseline being silently truncated
more often than the merge candidate.

This toolkit is the audit we wish we had run on day 1.

📄 **Read the paper:** [`papers/eval_trust/draft.pdf`](papers/eval_trust/draft.pdf)
(30 pages, 3 figures, 23 references — tells the full story).

---

## 30-second demo: reproduce the case-study flip

```bash
git clone https://github.com/telleroutlook/evomerge-framework
cd evomerge-framework
pip install scipy
python - <<'PY'
import json
from scipy.stats import binomtest

# Real raw logs from the case study (n=199, both runs evaluated greedy on
# the same GSM8K dev items, only max_new_tokens differs):
with open("data/winner_max_new768.json") as f:
    winner = json.load(f)
with open("data/instruct_max_new768.json") as f:
    instruct = json.load(f)

wm = {r["id"]: r["correct"] for r in winner["results"] if "correct" in r}
im = {r["id"]: r["correct"] for r in instruct["results"] if "correct" in r}
common = sorted(set(wm) & set(im))
b = sum(1 for i in common if im[i] and not wm[i])  # Instruct-only correct
c = sum(1 for i in common if not im[i] and wm[i])  # winner-only correct
p = binomtest(min(b, c), b + c, p=0.5, alternative="two-sided").pvalue

print(f"AUDITED: n={len(common)}, b={b}, c={c}, p={p:.4f}")
# AUDITED: n=199, b=29, c=27, p=0.8939   ← the +10 pp wasn't real

# For comparison, here's the same paired test on the BROKEN protocol
# (max_new=300, the original Phase 13 numbers):
print("ORIGINAL: n=200, b=21, c=41, p=0.0151   ← was paired-significant")
PY
```

Output:

```
AUDITED:  n=199, b=29, c=27, p=0.8939   ← the +10 pp wasn't real
ORIGINAL: n=200, b=21, c=41, p=0.0151   ← was paired-significant
```

That's the entire case study, on real data, in one screenful.

---

## What's in the toolkit

```
eval_trust/
├── paired_stats.py       # McNemar exact, Wilson CI, paired bootstrap
├── conformal_ci.py       # split-conformal accuracy intervals (small n)
└── t0v2/
    ├── truncation_extract.py  # detect generations cut off by max_new_tokens
    └── aggregate.py            # multi-channel triage of "wrong" answers

data/                     # raw audit logs cited in the paper
├── winner_max_new768.json
├── instruct_max_new768.json
├── self_consistency_full.json    # SC-5 lottery rates
├── t0v2_aggregate.json
└── gsm8k_dev_200.json            # the 200-item dev split

papers/eval_trust/
├── draft.pdf             # ★ the paper
├── draft.md              # markdown source
├── refs.bib
├── figures/              # paired McNemar contingency, T0v2 channels, granularity Pareto
└── numbers_cross_check.json    # every number in the paper has a source file
```

Pure Python + NumPy + (optional) SciPy. **No GPU dependency. No model
inference.** All audits run on your existing log files.

---

## Three reasons to use it

**1. Cheap.** Audit cost ~milliseconds on 200 items. A `pytest` run
takes 0.04 s. The case study took 4 hours of laptop time end-to-end.

**2. Falsifiable.** Every channel is a deterministic predicate on
`(question, expected, gen_text)`. No probabilistic classifier, no
LLM-as-judge, no opaque model — you can replicate every label by hand.

**3. Embarrassing if you don't.** A small-delta merging claim on a
standard benchmark, without a paired McNemar `(b, c)` count and a
SC-5 lottery rate report, is consistent with a real improvement, with
a primitive contamination, and with several mechanisms in between. The
audit costs you nothing and removes the ambiguity. Reviewers will
increasingly expect it.

---

## Tests

```bash
pip install scipy pytest
PYTHONPATH=. pytest tests/ -q
```

---

## Citation

```bibtex
@misc{evaltrust2026,
  title  = {Silent Contamination in {LLM} Merging Evaluation:
            A Case Study from a 5-Month Misadventure},
  author = {{telleroutlook}},
  year   = {2026},
  url    = {https://github.com/telleroutlook/evomerge-framework},
}
```

## License

- **Code** (`eval_trust/`, `tests/`): Apache-2.0 (see `LICENSE`)
- **Paper** (`papers/eval_trust/draft.{pdf,md}`): CC BY 4.0
- **Data** (`data/`): CC BY 4.0 (these are evaluation logs of public Qwen
  models on GSM8K, both of which are publicly licensed)

---

## Status

Pre-arxiv preprint. The paper is camera-ready (PDF compiles cleanly via
pandoc + tectonic). Awaiting an arxiv endorsement to assign an arXiv ID.

If you'd like to **endorse** this submission for arxiv `cs.CL`, please open
an issue or DM — endorsement is a 30-second click, no paper review needed.
