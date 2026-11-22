import pytest

torch = pytest.importorskip("torch")
import json
import torch.nn as nn
from pathlib import Path

from stack.bulk_stage6 import (
    evaluate_bulk_adapter,
    load_adapter_checkpoint,
    preserve_stack_checkpoint,
    save_adapter_checkpoint,
)

RANDOM_SEED = 20261122


def _seed_test_rng():
    torch.manual_seed(RANDOM_SEED)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = nn.Parameter(torch.tensor(0.8))

    def forward_batch(self, batch):
        target = batch["composition"] * batch["scale_target"].reshape(-1, 1)
        return (
            target.unsqueeze(1).expand(-1, batch["cell_set"].shape[1], -1)
            / batch["cell_set"].shape[1]
            * self.multiplier
        )


class TinyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_genes = 3
        self.encoder = nn.Linear(3, 2)

    def _reduce_and_tokenize(self, features):
        return self.encoder(features)

    def _run_attention_layers(self, tokens):
        return tokens


def _eval_batch():
    real = torch.tensor(
        [
            [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0]],
            [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]],
        ]
    )
    bulk = real.sum(dim=1)
    total = bulk.sum(dim=-1, keepdim=True)
    return {
        "composition": bulk / total,
        "scale_target": torch.tensor([1.2, 0.8]),
        "cell_set": real,
        "t_ref": torch.tensor([10.0, 12.0]),
        "donor_id": ["heldout-a", "heldout-b"],
    }


def test_synthetic_validation_manifest_matches_fixture():
    manifest_path = Path(__file__).parents[1] / "configs" / "bulk_stage6_validation.json"
    manifest = json.loads(manifest_path.read_text())
    batch = _eval_batch()

    assert manifest["random_seed"] == RANDOM_SEED
    assert manifest["dataset"]["source"] == "tests/test_bulk_stage6.py::_eval_batch"
    assert manifest["dataset"]["donor_ids"] == batch["donor_id"]
    assert manifest["dataset"]["cell_counts"] == batch["cell_set"].tolist()
    assert manifest["dataset"]["t_ref"] == batch["t_ref"].tolist()
    assert manifest["dataset"]["scale_target"] == pytest.approx(
        batch["scale_target"].tolist()
    )
    assert manifest["evaluation"]["sliced_wasserstein_projections"] == 8


def test_adapter_checkpoint_round_trip_does_not_change_stack_checkpoint(tmp_path):
    _seed_test_rng()
    stack = TinyStack()
    stack_checkpoint = tmp_path / "stack.pt"
    torch.save(stack.state_dict(), stack_checkpoint)
    original_stack_file = stack_checkpoint.read_bytes()
    backup_checkpoint = tmp_path / "preserved" / "stack-original.pt"
    original_hash = preserve_stack_checkpoint(stack_checkpoint, backup_checkpoint)
    assert backup_checkpoint.read_bytes() == original_stack_file
    assert preserve_stack_checkpoint(stack_checkpoint, backup_checkpoint) == original_hash
    original_stack_state = {
        name: value.clone() for name, value in stack.state_dict().items()
    }

    trained_adapter = TinyAdapter()
    trained_adapter.multiplier.data.fill_(1.25)
    adapter_checkpoint = tmp_path / "adapter.pt"
    save_adapter_checkpoint(
        trained_adapter,
        adapter_checkpoint,
        config={"n_proj": 8, "adapter": "TinyAdapter"},
        dataset={"name": "synthetic_stage6_heldout_fixture", "donor_ids": ["heldout-a", "heldout-b"]},
        random_seed=RANDOM_SEED,
        metadata={"step": 23, "T_ref": 10.0},
        stack_checkpoint_path=stack_checkpoint,
    )

    restored_adapter = TinyAdapter()
    metadata = load_adapter_checkpoint(restored_adapter, adapter_checkpoint)

    assert metadata == {
        "step": 23,
        "T_ref": 10.0,
        "config": {"n_proj": 8, "adapter": "TinyAdapter"},
        "dataset": {"name": "synthetic_stage6_heldout_fixture", "donor_ids": ["heldout-a", "heldout-b"]},
        "random_seed": RANDOM_SEED,
    }
    assert restored_adapter.multiplier.item() == pytest.approx(1.25)
    with pytest.raises(ValueError, match="not a STACK bulk-adapter checkpoint"):
        load_adapter_checkpoint(TinyAdapter(), stack_checkpoint)
    assert stack_checkpoint.read_bytes() == original_stack_file
    assert all(
        torch.equal(original_stack_state[name], value)
        for name, value in stack.state_dict().items()
    )
    assert backup_checkpoint.read_bytes() == original_stack_file


def test_heldout_evaluation_reports_baselines_and_downstream_metrics():
    _seed_test_rng()
    adapter = TinyAdapter()
    stack = TinyStack()
    original_stack_state = {
        name: value.clone() for name, value in stack.state_dict().items()
    }

    report = evaluate_bulk_adapter(
        adapter,
        [_eval_batch()],
        stack,
        stack_baseline=lambda batch: (
            batch["composition"] * batch["scale_target"].reshape(-1, 1)
        ).unsqueeze(1).expand(-1, batch["cell_set"].shape[1], -1)
        / batch["cell_set"].shape[1],
        downstream_evaluator=lambda predicted, real: {
            "set_mae": (predicted - real).abs().mean().item(),
            "reference_scale_mean": real.mean().item(),
        },
        n_proj=8,
    )

    assert report["n_samples"] == 2
    assert report["heldout_donors"] == ["heldout-a", "heldout-b"]
    for baseline_name in ("adapter", "bulk_baseline", "unmodified_stack_baseline"):
        metrics = report[baseline_name]
        assert metrics is not None
        assert {
            "aggregate_mae",
            "aggregate_relative_l1",
            "profile_sliced_wasserstein",
            "embedding_sliced_wasserstein",
            "downstream/set_mae",
            "downstream/reference_scale_mean",
        } <= metrics.keys()
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert report["stack_baseline_status"] == "evaluated"
    expected_real = _eval_batch()["cell_set"] / torch.tensor([10.0, 12.0])[:, None, None]
    assert report["adapter"]["downstream/reference_scale_mean"] == pytest.approx(
        expected_real.mean().item()
    )
    assert all(
        torch.equal(original_stack_state[name], value)
        for name, value in stack.state_dict().items()
    )
    assert all(parameter.requires_grad for parameter in stack.parameters())
    assert stack.training


def test_evaluation_requires_heldout_batches():
    with pytest.raises(ValueError, match="No held-out"):
        evaluate_bulk_adapter(TinyAdapter(), [], TinyStack())
