# Cache replay: no cache vs exact vs semantic caching

**Question:** on realistic support traffic, how much does semantic caching save on top of exact caching, and how often
does it hand a user someone else's answer?

**Answer (v2, held-out stream of 1,000 requests):** exact caching alone avoids 32.8% of model calls with no wrong
answers. Semantic caching adds 1.4–4.0 more points at settings that made no wrong reuses (40 extra reuses at the
shipped neural default, 0.96). Pushing it to 6.2 points (lexical 0.70) costs 2 wrong answers, both "get my invoice"
served for "see my invoices". The extra savings are small next to exact caching's, and the errors come from intents
that differ by one verb, so the threshold has to stay high.

## Data

[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset),
revision `430d1a89bd93bd1fa23c16f29dd53e73f0087443`, CDLA-Sharing-1.0: 26,872 user utterances across 27 intents
(cancel_order, track_order, get_refund, track_refund, check_invoice, get_invoice, …), with typos and slots such as
`{{Order Number}}`. It is downloaded at run time and not committed.

## How the traffic is generated ([`build_workload.py`](build_workload.py), seeded)

1. **Split:** utterances that appear under two intents are dropped and duplicates merged. Each intent's utterances
   are split in half: one half builds the *tune* stream, the other the held-out *test* stream (no shared text).
2. **Correct answer = label:** intent + every entity in the request, which covers:
   - filled slots (orders 4417/9032/…, account types, names, cities);
   - numbers written in the text;
   - slot values written literally ("a gold account").

   "Cancel order 4417" and "cancel order 9032" need different answers, and so do "track my refund" and "what is the
   refund policy". Labelling is conservative: a reuse across different entities always counts as wrong.
3. **Traffic:** intents follow Zipf(1.0) popularity. With probability 0.3 a request re-sends an earlier request's exact
   text with the same label (retries, FAQ buttons); otherwise it draws a fresh utterance.
4. **Each request is classed** against what came before:
   - **exact_repeat:** reuse is correct;
   - **paraphrase:** same label, new wording; reuse is correct;
   - **new_answer:** reuse is wrong.

## How it is replayed ([`run_replay.py`](run_replay.py))

- **Setup:** one client sends requests sequentially in stream order. The gateway runs 2 workers and is started fresh
  (empty cache) for every run. The upstream is the mock LLM answering in 200–600 ms, so **latency is simulated**.
- **Judging a reuse:** the mock's response id is a hash of the prompt, so each served answer maps back to the request
  that filled the cache. A reuse is correct when the labels match.
- **Tune:** a 3,000-request tune stream is swept over a threshold grid for the lexical encoder and the neural one
  (bge-small through `benchmarks/embedder_sidecar.py`, CPU). The threshold picked is the one with the most correct
  reuses among those with **zero** wrong reuses on the tune stream.
- **Test:** the held-out stream is replayed once per mode:
  - `off`: every request sent `Cache-Control: no-cache, no-store`;
  - `exact`: exact-match cache only;
  - `lexical`: exact plus lexical semantic matching;
  - `neural`: exact plus bge-small semantic matching.

  Each semantic mode runs at its tuned threshold and at its shipped default.

## Results (v2: [`replay_bitext_v2.json`](../reports/replay_bitext_v2.json))

Held-out stream: 1,000 requests, of which 328 are exact repeats, 583 paraphrases and 89 need a new answer.

| Mode | Model calls | Avoided vs off | Correct reuses | Wrong reuses | p50 / p95 latency (simulated) |
|---|---|---|---|---|---|
| off | 1,000 | — | 0 | 0 | 413 / 586 ms |
| exact | 672 | 32.8% | 328 | 0 | 311 / 563 ms |
| lexical @ 0.94 (default) | 654 | 34.6% | 346 | 0 | 295 / 571 ms |
| neural @ 0.98 (tuned) | 658 | 34.2% | 342 | 0 | 306 / 581 ms |
| neural @ 0.96 (default) | 632 | **36.8%** | 368 | 0 | 298 / 578 ms |
| lexical @ 0.70 (tuned) | 610 | 39.0% | 388 | **2** | 278 / 577 ms |

**Tuning sweep (3,000-request tune stream), wrong reuses by threshold:**
- **Lexical:** 8 at 0.60, 7 at 0.65, then 0 from 0.70 up.
- **Neural:** 28 at 0.70, 12 at 0.88, 6 at 0.90, 3 at 0.92, 2 at 0.94–0.96, 0 at 0.98.

So the zero-error rule picked 0.98 for neural. The shipped 0.96 made 2 wrong reuses in 3,000 tune requests and 0 on the
held-out stream.

**The two held-out errors** (lexical @ 0.70), at similarity 0.74 and 0.73:
- "I need to see the invoices from Maria Lopez" was served the answer to "I want to get my invoice from Maria Lopez" (`check_invoice` vs `get_invoice`).
- "help to get my bill from Sam Okafor" was served the answer to "help to find my bill from Sam Okafor".

**What it means:**
- **Most of the saving is exact caching.** Semantic caching reused only 14–60 of the 583 paraphrases: the salient-token
  guard and high thresholds reject most rewordings on purpose.
- **Errors come from intent pairs one verb apart,** never from wrong entities. The guard kept order numbers, names and
  amounts from ever being mixed up in either stream.
- **p95 barely moves,** because it is set by the remaining misses. p50 drops 413 → 311 ms with exact caching and to
  295–306 ms with semantic caching.
- **Reference-answer volume avoided:** 31,151 words with exact caching and 34,600 with neural @ 0.96, measured as words
  in the dataset's answers, not model tokens.

## v1 (superseded: [`replay_bitext_v1.json`](../reports/replay_bitext_v1.json))

The first run had two flaws, which it exposed itself:

1. **A labelling bug.** Labels ignored slot values that the dataset writes literally, so "can I create a gold
   account?" and "create a Gold account" got different labels. Correct reuses were scored as wrong, including the only
   "error" at the lexical default.
2. **The tune stream was too small.** Its 600 requests produced no errors at any threshold, so the rule picked the
   bottom of the grid (lexical 0.70, neural 0.80). On the test stream, neural 0.80 then made 14 wrong reuses (12 of them
   `delivery_options` vs `delivery_period`).

v2 fixes the labels, uses a 3,000-request tune stream and a wider grid, and flags a choice at the grid's edge. v1 was
built by the previous version of `build_workload.py`, so it can't be reproduced with the current one.

## Reproduce (Linux)

```bash
curl -L -o bitext.csv "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/430d1a89bd93bd1fa23c16f29dd53e73f0087443/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
python3 benchmarks/replay/build_workload.py --csv bitext.csv --split tune --n 3000 --out tune.jsonl
python3 benchmarks/replay/build_workload.py --csv bitext.csv --split test --n 1000 --out test.jsonl
pip install fastembed   # for --neural
python3 benchmarks/replay/run_replay.py --binary proxy-cpp/build/vanguard-proxy \
    --tune tune.jsonl --test test.jsonl --out replay.json --neural
```

## Limits

- **The traffic is generated,** from a public dataset that is itself partly synthetic; it is not real user logs.
  Repeat rate and popularity are assumptions (0.3 and Zipf 1.0) and move the exact-cache number directly.
- **The upstream is a mock,** so latency and "model calls" are simulated. Correctness doesn't depend on it.
- **One client, sequential:** this measures reuse decisions, not throughput.
- **Labels follow the dataset's intents.** Some intent pairs (`get_invoice` / `check_invoice`) may share an answer in a
  real product, which would make those two "errors" acceptable there.
