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
