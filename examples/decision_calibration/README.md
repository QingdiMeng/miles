# Decision calibration pilot

This example trains a one-token, thinking-disabled decision policy. GPQA has
one-hot targets; coin, weighted-die and urn questions have analytically known
four-option target probabilities. Candidate probabilities are normalized over
the four letter tokens, rather than the complete vocabulary.

Install the lightweight dataset dependencies with `uv pip install
./examples/decision_calibration`. Run modules from the repository root so their
absolute imports resolve. Model-facing scripts also require the existing Miles
PyTorch/Transformers environment.

```bash
python -m examples.decision_calibration.prepare --output-dir /path/to/data
```

The builder fetches the original public GPQA archive, records its SHA256, shuffles
answer positions, and checks scenario IDs are disjoint across splits:

| Split | GPQA | Synthetic | Total |
| --- | ---: | ---: | ---: |
| Train | 256 | 768 | 1024 |
| Validation | 64 | 192 | 256 |
| Test | 128 | 384 | 512 |

Each synthetic split has equal numbers of coin, die and urn scenarios. Questions
are split before any variants. This prevents experiment-induced overlap, but
does not establish that GPQA was absent from the original model's pretraining.

## Objective and training hooks

For candidate distribution `p` and target `t`, minimize `sum((p-t)**2)`.
Sample action `a` from the rollout policy and recompute the detached reward
`r = 2*(t[a]-p_current[a])` on the trainer.
Then `E[r * grad(log(p[a]))]` is the negative Brier gradient at the rollout
policy. Async training uses the unclipped importance-ratio surrogate
`-r * p_current[a]/p_rollout[a]`. Its expected gradient under the recorded
rollout policy equals the current Brier gradient, including for stale actions.
Do not differentiate through the reward. The rollout reward is diagnostic;
the trainer refreshes it for optimization.

This is a REINFORCE-style calibration update, not an ordinary GRPO Brier reward.
Group standard-deviation normalization and PPO clipping would change this
gradient. The pilot deliberately uses neither. A direct differentiable Brier
baseline is useful if the stochastic estimator is too noisy.

Miles configuration hooks:

```text
--custom-generate-function-path examples.decision_calibration.rollout.generate
--custom-rm-path examples.decision_calibration.rollout.reward
--loss-type custom_loss
--custom-loss-function-path examples.decision_calibration.loss.policy_loss
--disable-compute-advantages-and-returns
--rollout-temperature 1
--rollout-top-p 1
--rollout-top-k -1
--rollout-max-response-len 1
```

Omit `--apply-chat-template`: the custom generator renders the original messages
with `enable_thinking=False`. It asks SGLang for all candidate log probabilities
at one answer position and samples locally from the normalized distribution.
The server's independently generated token is ignored. Training requires TP=1,
CP=1 and one-token responses; EP is supported. Use one optimizer update per fresh
rollout and no KL/entropy penalty. The importance-weighted reward refresh handles
the one-batch policy lag of async training.
Decision entropy and train/rollout probability differences are recorded by the
custom loss. The protocol must be checked against the actual SGLang version before
launching training.

## Two-node recipe

The recipe prints its configuration by default:

```bash
python scripts/run_qwen3_6_decision_calibration.py \
    --model-dir /path/to/models --data-dir /path/to/data \
    --output-dir /path/to/results --run-id YYMMDD-0123abcd \
    --wandb-project decision-calibration
```

After launch approval and source snapshots, use an externally joined Ray cluster
with `MILES_SCRIPT_EXTERNAL_RAY=1` and add `--launch`. The recipe uses eight
training GPUs with expert parallelism and eight single-GPU rollout engines,
200 async updates via `train_async.py` at learning rate 1e-6, 32 questions with eight actions
each per update, validation every 25 updates, and checkpoints every 50. Thinking
and MTP are disabled. Dashboard, traces, entropy observations, Prometheus and
cache-aware SGLang routing are enabled. Keep authentication in the environment
or netrc. Set `--dump-details` to a durable trace location and retain checkpoint
artifacts before releasing temporary hardware.

## Held-out comparison

```bash
python -m examples.decision_calibration.evaluate \
    --data /path/to/data/test.jsonl \
    --model /path/to/original-model \
    --endpoint http://localhost:30000 \
    --output /path/to/results/before.jsonl

python -m examples.decision_calibration.evaluate \
    --data /path/to/data/test.jsonl \
    --model /path/to/original-model \
    --endpoint http://localhost:30000 \
    --output /path/to/results/after.jsonl \
    --baseline /path/to/results/before.jsonl
```

Choose checkpoints using validation only; evaluate the chosen checkpoint on the
locked test set once. Reports include Brier excess, expected outcome Brier,
cross entropy, decision entropy, top-choice success probability, top-label ECE
and paired bootstrap intervals. GPQA and each synthetic family are reported
separately. For GPQA, excess and ordinary Brier coincide. For synthetic outcomes,
expected Brier adds the irreducible constant `1-sum(t**2)` to excess Brier.

Brier measures overall probability quality, including resolution, so report ECE
and GPQA accuracy alongside it. Treat 128 held-out GPQA questions as a pilot,
not evidence of broad generalization. An interval spanning zero is inconclusive;
do not claim training improved calibration just because the training loss fell.
# Larger GPQA/MMLU mixture

## Generated probability reports

Pass `--probability-report` to the launcher to use conventional sequence-level
GRPO instead of the sampled decision-token Brier-gradient estimator. The model
generates `{"A":0.1,"B":0.6,"C":0.2,"D":0.1}` with thinking disabled and a
128-token response cap. Each valid report receives negative full-distribution
Brier loss against the metadata target after normalization. Reports may omit
zero-probability options: `{"B":1}` is a valid one-hot report. Keys must be
unique and belong to the question's options. Finite nonnegative values are
normalized to sum to one, so `{"A":2,"B":6}` becomes probabilities 0.25 and
0.75. Negative values, nonnumeric values, unknown/duplicate keys, and empty or
zero-total reports receive -3, below the worst valid reward of -2. The prompt
explains normalization and replaces the strict instruction in previously
prepared data. Eight sampled reports per question receive GRPO group
advantages. This trains stated probabilities; answer-token logits require a
separate evaluation and are not assumed to become calibrated.


`python -m examples.decision_calibration.prepare_mixed --pilot-data-dir /path/to/pilot-data --mmlu-parquet /path/to/mmlu-test.parquet --output-dir /path/to/new-data`
prepares 8,192 training questions: 256 GPQA, 7,117 MMLU, and 819 synthetic.
It preserves the pilot GPQA holdouts and reserves subject-stratified MMLU
validation (1,024) and test (2,048) questions. Including GPQA and synthetic
holdouts, validation has 1,184 questions and test has 2,368.
Normalized duplicate question text and synthetic scenarios are excluded across
splits. Synthetic prompts explicitly ask the model to take a guess and retain
analytic soft probability targets. The output directory must not already exist.
Once MMLU test questions enter training, use the reserved subset for evaluation;
full-MMLU scores are no longer entirely held out. Prior benchmark exposure and
pretraining contamination are distinct from this fine-tuning split.

Evaluate JSON reports with `python -m examples.decision_calibration.evaluate_reports --data <test.jsonl> --model <model-path> --endpoint <url> --output <predictions.jsonl>`. The evaluator uses greedy decoding with thinking disabled and reports format validity, penalized reward, and calibration metrics on valid reports separately. Run both original and final models on the same held-out split.

The next calibration recipe uses an explicit Brier-scoring instruction, 32 samples per prompt and LR 3e-7. Pass --routing-replay for MoE rollout routing replay; generated reports request expert routes and use the standard Miles response transport. Save only the final checkpoint with --save-interval 512 when disk space is limited.

Future JSON-report runs log rollout/probability_collapse_pct (0-100): valid reports with any probability exactly 1, divided by all rollouts. Invalid JSON remains in the denominator but is not counted as collapsed. Also log report_valid_pct, probability_collapse_valid_pct, counts and source-specific metrics, including validation. These hooks preserve the default Miles logs.


### MMLU-Pro mixture

`examples.decision_calibration.prepare_mmlu_pro` replaces MMLU and preserves the GPQA and synthetic partitions. Supply `--previous-data-dir`, `--mmlu-pro-parquet`, `--source-revision`, and `--output-dir`.

Train: 8,192 questions (7,117 MMLU-Pro, 256 GPQA, 819 synthetic). Validation: 1,184 (1,024 MMLU-Pro, 64 GPQA, 96 synthetic). Test: 2,368 (2,048 MMLU-Pro, 128 GPQA, 192 synthetic). MMLU-Pro is subject-stratified across all 14 categories and deduplicated by normalized question text before splitting. Source revision and file hashes are saved in the manifest.

JSON reports support each question's actual option count, up to ten, with keys A through the final option. Use probability-report mode; the original four-option decision-token rollout is incompatible. Prepared prompts contain the scoring instruction, and prompt conversion is idempotent.

For ten-option MMLU-Pro reports, pass --report-max-response-len 512 to avoid truncating formatted JSON. This sets both training and validation response budgets.

### Automatic evaluation after termination

Run `python -m examples.decision_calibration.eval_after_training` in a persistent
tmux session on the trainer host. Supply `--run-id`, `--ray-job-id`,
`--checkpoint-dir`, `--model`, `--data`, `--result-dir`, and `--export-dir`.
The watcher starts evaluations after success, failure, or stop, with a driver
process fallback if Ray becomes unavailable. It checks distributed checkpoint
metadata and referenced file extents before exporting. Saves before 128 completed
updates are excluded; zero-based `iter_0000127` is labelled `step128`.

The original model and every complete saved checkpoint are evaluated sequentially
on the specified held-out split, with thinking disabled and 512 output tokens.
Each export/evaluation is retried once; a failed checkpoint does not prevent later
evaluations. JSON predictions, calibration summaries, logs, and `status.json`
are retained in the result directory. Exports go to the separately specified
scratch directory. A lock prevents duplicate watchers. `--no-wait --dry-run`
tests discovery without launching models. The running training source need not
be changed: use a separate checkout for the watcher.
# Public JevBench evaluation

`evaluate_jevbench.py` evaluates the 231 published easy, original, and hard
items from a pinned checkout of `fstandhartinger/jevbench`. Add that checkout
to `PYTHONPATH` to use its native scoring and metrics. The model generates
letter-keyed JSON probabilities; letters map to the original ordered labels.
Only the state, instructions, and option criteria enter the prompt. Gold labels
and provenance rationales never enter it. Reasoning is disabled. This is a
public-subset diagnostic, not an official score including sealed items.

The evaluator reports validity, full multiclass Brier, ten-bin top-label ECE,
accuracy, ordinal MAE, exact probability-one collapse, and request latency.
Requests are serial after an excluded warm-up. JSON parsing uses the training
sparse-report contract: omitted options become zero and finite nonnegative
weights with a positive total are normalized before scoring. Previously archived
evaluations used the strict parser; their results are unchanged. Ordinal
accuracy uses native argmax correctness while ordinal MAE uses expected level.

`run_jevbench.py` copies four HF exports from an archive, verifies every file's
SHA-256 against its manifest, and evaluates the baseline and four checkpoints
on five separate GPUs. Run it only after verifying these GPUs are idle. It
terminates its own servers after evaluation and preserves per-question results.

`compare_jevbench.py` checks identical question IDs across models and reports
paired bootstrap intervals for Brier changes. It also measures squared error
against exact `gold_probs` on the ten public questions that provide them.

To compare a subset of checkpoints, pass `--labels baseline step128 step256 step384` to `compare_jevbench.py`. The baseline must be included.

`prepare_supergpqa.py` expands the 8,192-question mixture to 16,384 training
questions by adding 7,373 medium/hard SuperGPQA items and 819 new analytical
random-outcome scenarios. Validation stays at 1,184 questions: 512 MMLU-Pro,
512 SuperGPQA, 64 GPQA, and 96 synthetic questions. All previous validation
and test questions remain excluded from the additional training pool.
The three pinned public JevBench files are used only for overlap exclusion;
the builder writes no test split. It validates answer mappings and target
distributions, checks cross-split duplicates, shuffles both output splits,
and records source revisions, checksums, counts, and exclusion checks.

```sh
python -m examples.decision_calibration.prepare_supergpqa \
    --previous-data-dir /path/to/previous-mixture \
    --supergpqa-jsonl /path/to/SuperGPQA-all.jsonl \
    --source-revision SUPERGPQA_COMMIT \
    --jevbench-dir /path/to/jevbench/datasets/public \
    --jevbench-revision JEVBENCH_COMMIT \
    --output-dir /path/to/new-mixture
```
