from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def api():
    from workflow_distillation.lora_fp16 import span_cache, span_scoring

    return span_scoring, span_cache


def reference():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(130, 3)

        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids).cumsum(dim=1))

    torch.manual_seed(9)
    decoder, head = Decoder(), torch.nn.Linear(3, 130, bias=False)
    return SimpleNamespace(
        decoder=decoder, head=head, model=torch.nn.ModuleDict({"decoder": decoder, "head": head})
    )


class Characters:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_cached_logps_and_span_ablation_match_dense_teacher_forcing():
    s, _ = api()
    state = reference()
    ids = [ord(c) for c in "hAlpha.Beta.TT"]
    groups = [0] + [1] * 11 + [3] * 2
    q = s.cache_reference(state, ids, groups, chunk_size=2)
    h = state.decoder(input_ids=torch.tensor([ids])).last_hidden_state
    ce = F.cross_entropy(
        state.head(h[:, :-1]).transpose(1, 2), torch.tensor([ids[1:]]), reduction="none"
    )[0]
    np.testing.assert_allclose(q, -ce.detach().numpy(), rtol=1e-6)
    turn = dict(
        text="hAlpha.Beta.TT",
        spans=[[1, 7], [7, 12]],
        action_chars=[12, 14],
        action_ids=[ord("T")] * 2,
    )
    scored = s.score_turn(state, Characters(), turn)
    expected = []
    for text in ["hAlpha.Beta.TT", "hBeta.TT", "hAlpha.TT"]:
        x = torch.tensor([Characters().encode(text)])
        logits = state.head(state.decoder(input_ids=x).last_hidden_state)[0, -3:-1]
        expected.append(float(logits.log_softmax(-1)[:, ord("T")].sum().detach()))
    assert scored["full_logp"] == pytest.approx(expected[0], abs=1e-5)
    assert scored["deltas"] == pytest.approx([expected[0] - v for v in expected[1:]], abs=1e-5)
    assert not state.model.training


def test_sidecar_atomic_resume_integrity_and_complete_alignment(tmp_path):
    _, c = api()
    identity = {"model": "test", "revision": "pinned"}
    store = c.ScoreStore(tmp_path / "cache", "corpus", identity)
    ids, groups = [3, 4, 5, 6], [0, 1, 3, 4]
    assert store.read_record(0, ids, groups) is None
    store.write_reference(0, ids, groups, [-1, -2, -3])
    np.testing.assert_array_equal(store.read_reference(0, ids, groups), [-1, -2, -3])
    store.write_turn(0, 2, {"deltas": [1], "weights": [1]})
    assert store.read_turn(0, 2)["weights"] == [1]
    store.write_record(0, ids, groups, [-1, -2, -3], [0.5, 1, 1], [1, 1e-6, 1])
    assert store.read_record(0, ids, groups) is not None
    with pytest.raises(ValueError, match="complete"):
        store.finalize(
            [
                dict(trajectory_id="a", input_ids=ids, groups=groups),
                dict(trajectory_id="b", input_ids=ids, groups=groups),
            ]
        )
    store.finalize([dict(trajectory_id="a", input_ids=ids, groups=groups)])
    loaded = c.load_sidecar(tmp_path / "cache", "corpus", identity)
    np.testing.assert_array_equal(loaded.record(0)["ref_logp"], [-1, -2, -3])
    with pytest.raises(ValueError):
        c.load_sidecar(tmp_path / "cache", "wrong", identity)
    with pytest.raises(ValueError):
        c.ScoreStore(tmp_path / "cache", "corpus", {"revision": "different"})
    with pytest.raises(ValueError):
        store.read_record(0, [3, 4, 5, 7], groups)
    (tmp_path / "cache/ref_logp.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum|size"):
        c.load_sidecar(tmp_path / "cache", "corpus", identity)


@pytest.mark.parametrize(
    "q,w,gamma",
    [([float("nan")], [1], [1]), ([-1], [0], [1]), ([-1], [1], [0]), ([-1], [0.5], [1e-6])],
)
def test_cache_rejects_nonfinite_bad_mass_or_nonreasoning_weight(tmp_path, q, w, gamma):
    _, c = api()
    store = c.ScoreStore(tmp_path / "cache", "hash", {})
    with pytest.raises(ValueError):
        store.write_record(0, [3, 4], [0, 3], q, w, gamma)
