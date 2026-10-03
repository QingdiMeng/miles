"""Schema target integrity, proper-score gradients, and released-head parity."""

import copy
import random
from dataclasses import replace

import pytest
import torch

from examples.clef.data import DecisionExample, LabeledRecord, augment_example, convert_row, validate_distribution
from examples.clef.joint_schema_model import EncodedQuestion, EncodedRecord, JointSchemaHead
from examples.clef.objective import decision_loss, prediction_rows, summarize


def _example() -> DecisionExample:
    return convert_row({
        "prompt": [{"role": "user", "content": "Take a guess at the next coin flip.\n\nA. Heads\nB. Tails\n\nOld JSON reporting instruction."}],
        "metadata": {"id": "coin-1", "source": "coin", "choices": ["Heads", "Tails"], "target": [0.6, 0.4]},
    })


def _label(target: tuple[float, ...] = (1.0, 0.0)) -> LabeledRecord:
    question = EncodedQuestion("answer", 1, (0, 2), ((2, 4), (4, 6)), ("A", "B"))
    encoded = EncodedRecord(tuple(range(8)), (question,), "example")
    return LabeledRecord(encoded, (target,), "gpqa")


@pytest.mark.parametrize("values", [[-0.1, 1.1], [float("nan"), 1.0], [0.2, 0.2], [True, False], [1.0], [float("inf"), 0.0]])
def test_invalid_targets_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_distribution(values)


def test_conversion_removes_only_old_reporting_instruction() -> None:
    example = _example()
    assert example.record["state"] == "Take a guess at the next coin flip."
    assert example.targets["answer"] == {"A": 0.6, "B": 0.4}


def test_option_shuffle_and_multi_field_targets_preserve_outcome() -> None:
    example = _example()
    unchanged = copy.deepcopy(example)
    orders = set()
    for seed in range(20):
        augmented = augment_example(example, random.Random(seed), multi_field_fraction=1)
        choices = augmented.record["questions"]["answer"]["criteria"]
        orders.add(tuple(choices.values()))
        for key, outcome in choices.items():
            assert augmented.targets["answer"][key] == (0.6 if outcome == "Heads" else 0.4)
        candidate = augmented.record["questions"]["candidate_is_answer"]["instructions"].split("option ")[1][0]
        assert augmented.targets["candidate_is_answer"]["true"] == augmented.targets["answer"][candidate]
        assert sum(augmented.targets["candidate_is_answer"].values()) == 1
    assert len(orders) == 2
    assert example == unchanged


def test_soft_target_brier_gradient_points_toward_true_distribution() -> None:
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = decision_loss([[logits]], [_label((0.7, 0.3))])
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]
    optimum = torch.tensor([0.7, 0.3]).log().requires_grad_()
    optimal_loss = decision_loss([[optimum]], [_label((0.7, 0.3))])
    optimal_loss.backward()
    assert optimal_loss.item() < 1e-12
    assert optimum.grad.abs().max().item() < 1e-6


def test_confident_wrong_prediction_has_maximal_penalty() -> None:
    label = _label()
    correct = decision_loss([[torch.tensor([100.0, -100.0])]], [label])
    wrong = decision_loss([[torch.tensor([-100.0, 100.0])]], [label])
    uncertain = decision_loss([[torch.zeros(2)]], [label])
    assert correct.item() == 0
    assert uncertain.item() == 0.5
    assert wrong.item() == 2


def test_metric_denominators_exclude_soft_targets_from_accuracy() -> None:
    rows = prediction_rows([[torch.tensor([100.0, -100.0])], [torch.zeros(2)]], [_label(), replace(_label((0.6, 0.4)), source="coin")])
    metrics = summarize(rows)
    assert metrics["single_answer_accuracy"] == 1
    assert metrics["collapse_percentage"] == 50
    assert metrics["ece"] == 0
    assert metrics["questions"] == 2


def test_head_option_permutation_and_normalization() -> None:
    torch.manual_seed(12)
    head = JointSchemaHead(hidden_size=16, width=16, routing_layers=1, layers=1, heads=4, feedforward=32).eval()
    label = _label()
    hidden = torch.randn(1, 8, 16)
    embedding = torch.randn(32, 16)
    ids, mask = torch.arange(8).unsqueeze(0), torch.ones(1, 8, dtype=torch.long)
    original = head(hidden, ids, mask, [label.encoded], embedding)[0][0]
    question = label.encoded.questions[0]
    reversed_question = replace(question, option_spans=tuple(reversed(question.option_spans)), option_ids=("B", "A"))
    reversed_record = replace(label.encoded, questions=(reversed_question,))
    permuted = head(hidden, ids, mask, [reversed_record], embedding)[0][0]
    torch.testing.assert_close(original.flip(0), permuted, atol=1e-6, rtol=1e-5)
    probabilities = original.softmax(-1)
    assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
    torch.testing.assert_close(probabilities.sum(), torch.ones(()))
    decision_loss([[original]], [replace(label, targets=((0.7, 0.3),))]).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
