import pytest

torch = pytest.importorskip("torch")

from stack.bulk_distribution import profile_sliced_wasserstein_loss


def test_profile_sliced_wasserstein_is_finite_and_differentiable_for_equal_sets():
    predicted = torch.rand(2, 5, 7, requires_grad=True)
    real = torch.rand(2, 5, 7)

    loss = profile_sliced_wasserstein_loss(predicted, real, n_proj=16)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum() > 0


def test_profile_loss_normalizes_each_cell_independently():
    predicted = torch.tensor([[[1.0, 3.0], [4.0, 1.0]]])
    real = torch.tensor([[[10.0, 30.0], [8.0, 2.0]]])

    loss = profile_sliced_wasserstein_loss(predicted, real, n_proj=8)

    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_profile_loss_rejects_different_cell_counts():
    with pytest.raises(ValueError, match="identical shapes"):
        profile_sliced_wasserstein_loss(torch.ones(2, 4, 3), torch.ones(2, 5, 3))


def test_zero_library_profiles_produce_finite_loss_and_gradients():
    predicted = torch.zeros(1, 2, 3, requires_grad=True)
    real = torch.zeros(1, 2, 3)

    loss = profile_sliced_wasserstein_loss(predicted, real)
    loss.backward()

    assert loss.item() == 0
    assert torch.isfinite(predicted.grad).all()
