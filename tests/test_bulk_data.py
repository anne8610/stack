import anndata
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from stack.data import create_bulk_train_val_test_datasets, prepare_bulk_inference_batch
from stack.bulk_adapter import BulkToSetAdapter, bulk_reconstruction_loss


def _adata(values, donors, genes=("G2", "G1"), raw=True):
    adata = anndata.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame({"donor_id": donors}, index=[str(i) for i in range(len(donors))]),
        var=pd.DataFrame(index=list(genes)),
    )
    if raw:
        adata.raw = adata.copy()
    return adata


def test_pseudobulk_contains_raw_cell_set_and_normalized_composition():
    adata = _adata(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        ["d1", "d1", "d1", "d2", "d2", "d2"],
    )
    adata.X = sparse.csr_matrix(np.full((6, 2), 0.5, dtype=np.float32))
    train, _, _ = create_bulk_train_val_test_datasets(
        [adata], ["G1", "G2", "G3"], "donor_id", sample_size=2,
        samples_per_donor=2, test_ratio=0, val_ratio=0, random_state=3,
    )

    sample = train[0]
    assert sample["cell_set"].shape == (2, 3)
    assert sample["bulk_counts"].shape == (3,)
    assert sample["composition"].shape == (3,)
    assert sample["bulk_counts"].tolist() == sample["cell_set"].sum(dim=0).tolist()
    assert (sample["cell_set"][:, 0] > sample["cell_set"][:, 1]).all()
    assert sample["composition"].sum().item() == pytest.approx(1.0)
    assert sample["composition"][2].item() == 0
    assert sample["scale_target"].item() == pytest.approx(
        sample["bulk_counts"].sum().item() / train.t_ref
    )
    source_rows = np.flatnonzero(adata.obs["donor_id"].to_numpy() == sample["donor_id"])
    expected_cells = adata.raw.X[source_rows][sample["cell_indices"].numpy()].toarray()
    expected_aligned = np.column_stack((expected_cells[:, 1], expected_cells[:, 0], np.zeros(2)))
    np.testing.assert_array_equal(sample["cell_set"].numpy(), expected_aligned)
    np.testing.assert_array_equal(sample["bulk_counts"].numpy(), expected_aligned.sum(axis=0))
    assert len(train) == 4


def test_pseudobulk_pipeline_accepts_user_supplied_h5ad_path(tmp_path):
    source = _adata(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        ["donor-a", "donor-a", "donor-b", "donor-b"],
    )
    source_path = tmp_path / "input.h5ad"
    source.write_h5ad(source_path)

    train, _, _ = create_bulk_train_val_test_datasets(
        [source_path],
        ["G1", "G2", "G3"],
        donor_col="donor_id",
        sample_size=1,
        samples_per_donor=1,
        test_ratio=0,
        val_ratio=0,
        random_state=5,
    )

    assert set(train.donor_ids) == {"donor-a", "donor-b"}
    assert all(sample["cell_set"].shape == (1, 3) for sample in train)


def test_real_bulk_preprocessing_matches_pseudobulk_normalization_contract():
    counts = torch.tensor([[2.0, 3.0, 0.0], [1.0, 1.0, 2.0]])
    t_ref = torch.tensor([10.0, 20.0])

    inference = prepare_bulk_inference_batch(counts, t_ref)

    totals = counts.sum(dim=-1)
    torch.testing.assert_close(
        inference["composition"],
        counts / (totals.unsqueeze(-1) + 1e-8),
    )
    torch.testing.assert_close(inference["scale_target"], totals / t_ref)
    torch.testing.assert_close(
        inference["composition"] * inference["scale_target"].unsqueeze(-1),
        counts / t_ref.unsqueeze(-1),
    )
    with pytest.raises(ValueError, match="positive library"):
        prepare_bulk_inference_batch(torch.zeros(1, 3), 10.0)


def test_stage1_dataloader_batch_feeds_stage2_adapter():
    adata = _adata(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        ["d1", "d1", "d1", "d2", "d2", "d2"],
    )
    train, _, _ = create_bulk_train_val_test_datasets(
        [adata],
        ["G1", "G2", "G3"],
        "donor_id",
        sample_size=2,
        samples_per_donor=2,
        test_ratio=0,
        val_ratio=0,
        random_state=4,
    )
    batch = next(iter(DataLoader(train, batch_size=2)))
    adapter = BulkToSetAdapter(n_genes=3, hidden_dim=8, max_cells=2, top_k_genes=2)

    generated = adapter.forward_batch(batch)
    loss = bulk_reconstruction_loss(
        generated, batch["composition"], batch["scale_target"]
    )

    assert generated.shape == batch["cell_set"].shape == (2, 2, 3)
    assert torch.isfinite(loss)
    assert torch.all(generated >= 0)


def test_donors_are_split_before_sampling_and_t_ref_uses_training_only():
    donors = np.repeat(["d1", "d2", "d3", "d4", "d5", "d6"], 4)
    values = np.ones((len(donors), 2), dtype=np.int64)
    split_args = dict(
        target_genes=["G1", "G2"],
        donor_col="donor_id",
        sample_size=2,
        samples_per_donor=2,
        test_ratio=1 / 6,
        val_ratio=1 / 6,
        random_state=17,
    )
    baseline = create_bulk_train_val_test_datasets(
        [_adata(values, donors)], **split_args
    )
    heldout_donor = baseline[2].donor_ids[0]
    values[donors == heldout_donor] *= 10000
    train, val, test = create_bulk_train_val_test_datasets(
        [_adata(values, donors)], **split_args
    )

    assert set(train.donor_ids).isdisjoint(val.donor_ids)
    assert set(train.donor_ids).isdisjoint(test.donor_ids)
    assert set(val.donor_ids).isdisjoint(test.donor_ids)
    assert {item["donor_id"] for item in train} == set(train.donor_ids)
    assert all(item["t_ref"] == train.t_ref for item in val)
    train_totals = [item["bulk_counts"].sum().item() for item in train]
    test_totals = [item["bulk_counts"].sum().item() for item in test]
    assert train.t_ref == pytest.approx(np.median(train_totals))
    assert train.t_ref == baseline[0].t_ref
    assert min(test_totals) > train.t_ref
    assert val.t_ref == test.t_ref == train.t_ref


def test_bad_count_matrix_is_not_silently_accepted():
    adata = _adata([[1.5, 2], [3, 4], [5, 6]], ["d1", "d1", "d2"])
    with pytest.raises(ValueError, match="non-integer"):
        create_bulk_train_val_test_datasets(
            [adata], ["G1", "G2"], "donor_id", sample_size=1,
            samples_per_donor=1, test_ratio=0, val_ratio=0,
        )


def test_donor_with_too_few_cells_fails_instead_of_sampling_with_replacement():
    adata = _adata([[1, 2], [3, 4], [5, 6]], ["d1", "d1", "d2"])
    with pytest.raises(ValueError, match="fewer than sample_size"):
        create_bulk_train_val_test_datasets(
            [adata], ["G1", "G2"], "donor_id", sample_size=2,
            samples_per_donor=1, test_ratio=0, val_ratio=0,
        )
