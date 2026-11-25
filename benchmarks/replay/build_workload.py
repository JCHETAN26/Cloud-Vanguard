#!/usr/bin/env python3
"""Build a labelled customer-support request stream for the cache-replay benchmark.

Source: the Bitext customer-support dataset (26,872 utterances, 27 intents,
CDLA-Sharing-1.0), pinned to a revision and downloaded at run time; it is not
committed to this repo.

  https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

How the traffic is generated (seeded, so every run is identical):

1. Split. Utterances whose text appears under more than one intent are dropped
   (ambiguous), duplicates within an intent are merged, and each intent's
   utterances are shuffled and split in half: the *tune*
   half builds the stream used to pick the semantic threshold, the *test* half
   builds the held-out stream the results are reported on. No utterance text is
   shared between the two.
2. Answers. A request's correct answer is identified by its label = intent +
   the values filled into its slots ({{Order Number}}, {{Account Type}}, ...)
   + any number already written in the utterance ("my bill #12588") + any slot
   value already written in it ("create a gold account"). Values are compared
   case-insensitively and regardless of slot type, so "a gold account" written
   by the dataset and "a {{Account Category}}" filled with Gold get one label.
   This is conservative: a reuse across different values always counts as wrong.
   "cancel order 4417" and "cancel order 9032" share an intent but need
   different answers; so do "cancel my order" and "what is the cancellation fee".
3. Popularity. Intents follow a Zipf(1.0) popularity over a seeded random order;
   slot values come from small pools, so the same (intent, values) recurs.
4. Repeats. With probability --repeat a request re-sends the exact text of an
   earlier request with the same label (users retrying, FAQ buttons); otherwise
   it draws an utterance from its intent's half.

Each request is classed against everything sent before it:
  exact_repeat   identical text sent before         -> a reuse is correct
  paraphrase     same label, different text before  -> a reuse is correct
  new_answer     label not seen before              -> any reuse is wrong

Output: JSON lines {i, text, intent, slots, label, category, ref_words}.
ref_words = words in the dataset's reference answer for that utterance
(used to report answer volume avoided; not model tokens).

    python3 benchmarks/replay/build_workload.py --csv bitext.csv --split test --n 1000 --out test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict

DATASET_URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
               "/resolve/{rev}/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
DATASET_REV = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"

SLOT_VALUES = {
    "{{Order Number}}": ["4417", "9032", "1185", "7720", "3369", "5804"],
    "{{Invoice Number}}": ["INV-2201", "INV-3307", "INV-4410"],
    "{{Account Type}}": ["Free", "Premium", "Business"],
    "{{Account Category}}": ["Standard", "Gold", "Platinum"],
    "{{Person Name}}": ["Maria Lopez", "Wei Chen", "Sam Okafor"],
    "{{Refund Amount}}": ["25", "140", "380"],
    "{{Currency Symbol}}": ["$", "€"],
    "{{Delivery City}}": ["Denver", "Leeds", "Pune"],
    "{{Delivery Country}}": ["Canada", "Germany", "India"],
}
SLOT_RE = re.compile(r"\{\{[^}]+\}\}")
NUM_RE = re.compile(r"\d+")
LITERAL_VALUES = sorted({v.lower() for vs in SLOT_VALUES.values() for v in vs if v.isalpha() or " " in v})
LITERAL_RE = re.compile(r"\b(" + "|".join(re.escape(v) for v in LITERAL_VALUES) + r")\b", re.IGNORECASE)


def load(path: str) -> dict[str, list[dict]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    intents_of: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        intents_of[row["instruction"].strip().lower()].add(row["intent"])
    by_intent: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        text = row["instruction"].strip()
        key = text.lower()
        if len(intents_of[key]) > 1 or key in seen:
            continue  # ambiguous across intents, or a duplicate
        seen.add(key)
        slots = sorted(set(SLOT_RE.findall(text)))
        if any(s not in SLOT_VALUES for s in slots):
            continue  # a slot type we have no values for
        bare = SLOT_RE.sub("", text)
        literal = NUM_RE.findall(bare) + [m.lower() for m in LITERAL_RE.findall(bare)]
        by_intent[row["intent"]].append({"text": text, "slots": slots, "numbers": literal,
                                         "ref_words": len(row["response"].split())})
    return by_intent


def split(by_intent: dict[str, list[dict]], which: str, seed: int) -> dict[str, list[dict]]:
    out = {}
    for intent in sorted(by_intent):
        rows = list(by_intent[intent])
        random.Random(f"{seed}:{intent}").shuffle(rows)
        half = len(rows) // 2
        out[intent] = rows[:half] if which == "tune" else rows[half:]
    return out


def build(pool: dict[str, list[dict]], n: int, repeat: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    intents = sorted(pool)
    rng.shuffle(intents)
    weights = [1.0 / (rank + 1) for rank in range(len(intents))]  # Zipf(1.0)
    sent_by_label: dict[str, list[str]] = defaultdict(list)
    seen_text: set[str] = set()
    seen_label: set[str] = set()
    out = []
    for i in range(n):
        intent = rng.choices(intents, weights)[0]
        utter = rng.choice(pool[intent])
        values = {s: rng.choice(SLOT_VALUES[s]) for s in utter["slots"]}
        entities = sorted({v.lower() for v in values.values()} | set(utter["numbers"]))
        label = intent + "|" + ",".join(entities)
        if sent_by_label[label] and rng.random() < repeat:
            text = rng.choice(sent_by_label[label])
        else:
            text = utter["text"]
            for slot, value in values.items():
                text = text.replace(slot, value)
        if text in seen_text:
            category = "exact_repeat"
        elif label in seen_label:
            category = "paraphrase"
        else:
            category = "new_answer"
        out.append({"i": i, "text": text, "intent": intent, "slots": values, "label": label, "category": category,
                    "ref_words": utter["ref_words"]})
        seen_text.add(text)
        seen_label.add(label)
        sent_by_label[label].append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="the Bitext CSV (download from DATASET_URL at DATASET_REV)")
    ap.add_argument("--split", choices=["tune", "test"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--repeat", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = split(load(a.csv), a.split, a.seed)
    # Different seeds for the two streams' sampling as well as different text.
    rows = build(pool, a.n, a.repeat, a.seed * 1000 + (1 if a.split == "tune" else 2))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["category"]] += 1
    print(json.dumps({"split": a.split, "n": len(rows), "categories": dict(counts),
                      "labels": len({r["label"] for r in rows}), "dataset_rev": DATASET_REV}))


if __name__ == "__main__":
    main()
