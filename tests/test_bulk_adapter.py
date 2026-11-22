import pytest

torch = pytest.importorskip("torch")

from stack.bulk_adapter import BulkToSetAdapter, bulk_reconstruction_loss


def test_adapter_outputs_nonnegative_sparse_cell_sets_with_expected_shape():
    adapter = BulkToSetAdapter(n_genes=7, hidden_dim=12, max_cells=5, top_k_genes=2)
    composition = torch.rand(3, 7)
    composition = composition / composition.sum(dim=-1, keepdim=True)
    batch = {
        "composition": composition,
        "scale_target": torch.tensor([0.5, 1.0, 2.0]),
        "cell_set": torch.rand(3, 4, 7),
    }

    predicted = adapter.forward_batch(batch)

    assert predicted.shape == (3, 4, 7)
    assert torch.all(predicted >= 0)
    assert torch.all((predicted > 0).sum(dim=-1) <= 2)


def test_bulk_reconstruction_loss_backpropagates_through_adapter():
    adapter = BulkToSetAdapter(n_genes=4, hidden_dim=8, max_cells=3)
    composition = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    scale_target = torch.tensor([1.5, 0.75])
    real_cell_set = torch.rand(2, 3, 4)

    predicted = adapter(composition, scale_target, n_cells=real_cell_set.shape[1])
    loss = bulk_reconstruction_loss(predicted, composition, scale_target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in adapter.parameters()
    )


def test_reconstruction_loss_rejects_mismatched_target_shapes():
    with pytest.raises(ValueError, match="composition shape"):
        bulk_reconstruction_loss(torch.ones(2, 3, 4), torch.ones(2, 5), torch.ones(2))
