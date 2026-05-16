"""Tests for the LLM Word Composer (U5).

Covers:
- ``core.spine.words.compose_with_llm`` instance-picking + repeat policies.
- ``core.remix.word_llm_composer.generate_llm_word_sequence`` end-to-end
  with a stubbed LLM.
- Round-robin distribution across multiple corpus instances (AE3).
- OOV emission failure (AE4 negative).
- Seeded random reproducibility.
- ``LLMEmptyResponseError`` propagation.
- Real-Ollama integration test (gated by ``SCENE_RIPPER_OLLAMA_INTEGRATION``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import pytest

from core.remix.word_llm_composer import (
    MissingWordDataError,
    generate_llm_word_sequence,
)
from core.spine.words import (
    WordInstance,
    build_inventory,
    compose_with_llm,
)
from core.transcription import TranscriptSegment, WordTimestamp


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


@dataclass
class MockSource:
    id: str = "src1"
    fps: float = 24.0


@dataclass
class MockClip:
    id: str = "clip1"
    source_id: str = "src1"
    start_frame: int = 0
    end_frame: int = 480  # 20 s at 24 fps
    transcript: Optional[list[TranscriptSegment]] = None


def _seg(words: list[tuple[str, float, float]], language: str = "en") -> TranscriptSegment:
    word_list = [WordTimestamp(start=s, end=e, text=t) for (t, s, e) in words]
    return TranscriptSegment(
        start_time=words[0][1] if words else 0.0,
        end_time=words[-1][2] if words else 0.0,
        text=" ".join(t for t, _, _ in words),
        confidence=1.0,
        words=word_list,
        language=language,
    )


def _fake_completion(words: list[str]):
    """Build a stub for ``compose_with_llm``'s ``_completion_fn`` hook."""

    def _fn(**kwargs):  # noqa: ANN003
        return list(words)

    return _fn


# ---------------------------------------------------------------------------
# compose_with_llm — spine layer
# ---------------------------------------------------------------------------


class TestComposeWithLlmHappyPath:
    def test_basic_five_word_response(self):
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("i", 0.0, 0.2),
                        ("have", 0.3, 0.5),
                        ("always", 0.6, 1.0),
                        ("loved", 1.1, 1.5),
                        ("silence", 1.6, 2.2),
                    ]
                )
            ]
        )
        inv = build_inventory([(clip, MockSource())])

        result = compose_with_llm(
            inv,
            prompt="compose",
            target_length=5,
            _completion_fn=_fake_completion(
                ["i", "have", "always", "loved", "silence"]
            ),
        )
        assert len(result) == 5
        assert [w.text for w in result] == [
            "i", "have", "always", "loved", "silence"
        ]

    def test_single_word_response(self):
        clip = MockClip(transcript=[_seg([("hello", 0.0, 0.3)])])
        inv = build_inventory([(clip, MockSource())])
        result = compose_with_llm(
            inv,
            prompt="x",
            target_length=1,
            _completion_fn=_fake_completion(["hello"]),
        )
        assert len(result) == 1


class TestRoundRobinPolicyAE3:
    def test_round_robin_cycles_through_instances(self):
        """With 12 instances of 'the' and 4 emissions under round-robin,
        the 4 emitted SequenceClips reference 4 different source instances
        in inventory order. Covers AE3.
        """
        # Build a clip with 12 instances of "the".
        words = [("the", i * 0.5, i * 0.5 + 0.3) for i in range(12)]
        clip = MockClip(
            end_frame=480,  # 20s @ 24fps so all "the" instances fit
            transcript=[_seg(words)],
        )
        inv = build_inventory([(clip, MockSource())])
        # Sanity: 12 instances in inventory.
        assert len(inv.by_word["the"]) == 12

        result = compose_with_llm(
            inv,
            prompt="x",
            target_length=4,
            repeat_policy="round-robin",
            _completion_fn=_fake_completion(["the", "the", "the", "the"]),
        )

        # 4 *different* instances, in inventory order (instances 0, 1, 2, 3).
        assert len(result) == 4
        # All from same word but different word_index — i.e., different
        # corpus instances.
        word_indices = [r.word_index for r in result]
        assert word_indices == [0, 1, 2, 3]
        assert len(set(word_indices)) == 4

    def test_round_robin_wraps_when_more_emissions_than_instances(self):
        """Documented: with a single corpus instance and 3 emissions, the
        same instance is reused 3 times (round-robin wraps).
        """
        clip = MockClip(transcript=[_seg([("solo", 0.0, 0.5)])])
        inv = build_inventory([(clip, MockSource())])
        result = compose_with_llm(
            inv,
            prompt="x",
            target_length=3,
            repeat_policy="round-robin",
            _completion_fn=_fake_completion(["solo", "solo", "solo"]),
        )
        assert len(result) == 3
        # All reuse the single instance.
        assert all(r.word_index == 0 for r in result)


class TestRepeatPolicies:
    def _three_the_clip(self) -> MockClip:
        return MockClip(
            transcript=[
                _seg(
                    [
                        ("the", 0.0, 0.2),  # short
                        ("the", 0.3, 1.0),  # long (0.7)
                        ("the", 1.1, 1.5),  # medium (0.4)
                    ]
                )
            ]
        )

    def test_first_always_returns_index_0(self):
        clip = self._three_the_clip()
        inv = build_inventory([(clip, MockSource())])
        result = compose_with_llm(
            inv, prompt="x", target_length=3,
            repeat_policy="first",
            _completion_fn=_fake_completion(["the", "the", "the"]),
        )
        assert all(r.word_index == 0 for r in result)

    def test_longest_picks_the_longest(self):
        clip = self._three_the_clip()
        inv = build_inventory([(clip, MockSource())])
        result = compose_with_llm(
            inv, prompt="x", target_length=1,
            repeat_policy="longest",
            _completion_fn=_fake_completion(["the"]),
        )
        # word_index=1 has end-start = 0.7, the longest.
        assert result[0].word_index == 1

    def test_shortest_picks_the_shortest(self):
        clip = self._three_the_clip()
        inv = build_inventory([(clip, MockSource())])
        result = compose_with_llm(
            inv, prompt="x", target_length=1,
            repeat_policy="shortest",
            _completion_fn=_fake_completion(["the"]),
        )
        # word_index=0 has end-start = 0.2, the shortest.
        assert result[0].word_index == 0

    def test_random_seeded_reproducible(self):
        """Same seed → identical output across runs."""
        # Build inventory with 5 instances of "x" so the policy actually
        # has choices to make.
        clip = MockClip(
            transcript=[
                _seg([("x", i * 0.5, i * 0.5 + 0.3) for i in range(5)])
            ]
        )
        inv = build_inventory([(clip, MockSource())])

        result_a = compose_with_llm(
            inv, prompt="p", target_length=10,
            repeat_policy="random", seed=42,
            _completion_fn=_fake_completion(["x"] * 10),
        )
        result_b = compose_with_llm(
            inv, prompt="p", target_length=10,
            repeat_policy="random", seed=42,
            _completion_fn=_fake_completion(["x"] * 10),
        )
        assert [r.word_index for r in result_a] == [r.word_index for r in result_b]

    def test_unknown_policy_raises(self):
        clip = MockClip(transcript=[_seg([("a", 0.0, 0.5)])])
        inv = build_inventory([(clip, MockSource())])
        with pytest.raises(ValueError, match="repeat_policy"):
            compose_with_llm(
                inv, prompt="p", target_length=1,
                repeat_policy="bogus",  # type: ignore[arg-type]
                _completion_fn=_fake_completion(["a"]),
            )


class TestOOVAndEmptyHandling:
    def test_oov_emission_raises_loudly_ae4(self):
        """LLM emits a word that's not in the corpus → raise loudly.

        The format+enum constraint should make this impossible. If we see
        an OOV word, the bug must surface — silently dropping the slot
        would hide a serious constraint-decoding failure. Covers AE4.
        """
        clip = MockClip(transcript=[_seg([("hello", 0.0, 0.3)])])
        inv = build_inventory([(clip, MockSource())])
        with pytest.raises(ValueError, match="out-of-vocabulary"):
            compose_with_llm(
                inv, prompt="x", target_length=2,
                _completion_fn=_fake_completion(["hello", "world"]),
            )

    def test_empty_inventory_raises(self):
        # No clips → empty inventory.
        from core.spine.words import WordInventory
        inv = WordInventory(by_word={}, by_clip={})
        with pytest.raises(ValueError, match="empty"):
            compose_with_llm(
                inv, prompt="x", target_length=2,
                _completion_fn=_fake_completion(["x"]),
            )

    def test_zero_target_length_raises(self):
        clip = MockClip(transcript=[_seg([("a", 0.0, 0.5)])])
        inv = build_inventory([(clip, MockSource())])
        with pytest.raises(ValueError, match="target_length"):
            compose_with_llm(
                inv, prompt="x", target_length=0,
                _completion_fn=_fake_completion(["a"]),
            )


class TestLLMEmptyResponsePropagates:
    def test_empty_response_error_bubbles_up(self):
        from core.llm_client import LLMEmptyResponseError

        clip = MockClip(transcript=[_seg([("a", 0.0, 0.5)])])
        inv = build_inventory([(clip, MockSource())])

        def _fail(**kwargs):
            raise LLMEmptyResponseError(prompt="p", hint="empty")

        with pytest.raises(LLMEmptyResponseError):
            compose_with_llm(
                inv, prompt="p", target_length=1,
                _completion_fn=_fail,
            )

    def test_ollama_unreachable_bubbles_up(self):
        from core.llm_client import OllamaUnreachableError

        clip = MockClip(transcript=[_seg([("a", 0.0, 0.5)])])
        inv = build_inventory([(clip, MockSource())])

        def _fail(**kwargs):
            raise OllamaUnreachableError("nope")

        with pytest.raises(OllamaUnreachableError):
            compose_with_llm(
                inv, prompt="p", target_length=1,
                _completion_fn=_fail,
            )


# ---------------------------------------------------------------------------
# generate_llm_word_sequence — remix wrapper
# ---------------------------------------------------------------------------


def _fake_spine_composer(words: list[str]):
    """Fake the spine ``compose_with_llm`` for the wrapper-level tests.

    Returns a callable matching ``compose_with_llm`` signature; ignores the
    LLM entirely and walks the inventory directly.
    """

    def _fn(inv, *, prompt, target_length, repeat_policy="round-robin",
            seed=None, model=None, api_base=None, temperature=0.7,
            timeout=120.0, system_prompt=None, think=False):  # noqa: ANN001
        # Translate the requested words into round-robin instance picks
        # so the test exercises the wrapper's plumbing without involving
        # any LLM.
        counters: dict[str, int] = {}
        out: list[WordInstance] = []
        for raw in words:
            key = raw.lower()
            candidates = inv.by_word.get(key)
            if not candidates:
                raise ValueError(f"OOV {raw!r}")
            c = counters.get(key, 0)
            out.append(candidates[c % len(candidates)])
            counters[key] = c + 1
        return out

    return _fn


class TestGenerateLlmWordSequence:
    def test_end_to_end_stubbed_llm(self):
        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("i", 0.0, 0.2),
                        ("have", 0.3, 0.5),
                        ("always", 0.6, 1.0),
                        ("loved", 1.1, 1.5),
                        ("silence", 1.6, 2.2),
                    ]
                )
            ]
        )
        result = generate_llm_word_sequence(
            clips=[(clip, MockSource())],
            prompt="compose",
            target_length=5,
            _compose_fn=_fake_spine_composer(
                ["i", "have", "always", "loved", "silence"]
            ),
        )
        assert len(result) == 5
        # All entries reference the same source clip and have well-formed
        # frame coords.
        for sc in result:
            assert sc.source_clip_id == clip.id
            assert sc.source_id == clip.source_id
            assert sc.in_point >= 0
            assert sc.out_point > sc.in_point

    def test_frame_math_matches_word_sequencer(self):
        """The U5 wrapper must use the same floor/ceil/handle frame math
        as U4's preset-modes path — they share the
        ``instances_to_sequence_clips`` helper, so identical inputs must
        produce byte-identical outputs.
        """
        from core.remix.word_sequencer import generate_word_sequence

        clip = MockClip(
            transcript=[
                _seg(
                    [
                        ("apple", 0.5, 0.9),  # in=floor(0.5*24)=12, out=ceil(0.9*24)=22
                    ]
                )
            ]
        )

        # U4: alphabetical mode on a single-word corpus yields the single
        # instance.
        preset_result = generate_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            mode="alphabetical",
            mode_params={},
            handle_frames=0,
        )

        # U5: LLM emits the same single word.
        llm_result = generate_llm_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            prompt="x",
            target_length=1,
            handle_frames=0,
            _compose_fn=_fake_spine_composer(["apple"]),
        )

        assert len(preset_result) == len(llm_result) == 1
        assert preset_result[0].in_point == llm_result[0].in_point
        assert preset_result[0].out_point == llm_result[0].out_point

    def test_empty_clips_returns_empty(self):
        result = generate_llm_word_sequence(
            clips=[], prompt="x", target_length=1,
            _compose_fn=_fake_spine_composer([]),
        )
        assert result == []

    def test_missing_word_data_raises(self):
        """Same MissingWordDataError contract as the preset-modes path —
        the dialog catches this and triggers alignment.
        """
        clip = MockClip(
            transcript=[
                TranscriptSegment(
                    start_time=0.0, end_time=1.0, text="x",
                    confidence=1.0, words=None, language="en",
                )
            ]
        )
        with pytest.raises(MissingWordDataError) as exc:
            generate_llm_word_sequence(
                clips=[(clip, MockSource())],
                prompt="x", target_length=1,
                _compose_fn=_fake_spine_composer([]),
            )
        assert clip.id in exc.value.clip_ids

    def test_handle_frames_passed_through(self):
        clip = MockClip(
            transcript=[_seg([("mid", 1.0, 2.0)])]
        )
        result = generate_llm_word_sequence(
            clips=[(clip, MockSource(fps=24.0))],
            prompt="x", target_length=1,
            handle_frames=2,
            _compose_fn=_fake_spine_composer(["mid"]),
        )
        # Same math as U4 test_handle_widens_window:
        # in = floor((1.0 - 2/24)*24) = 22
        # out = ceil((2.0 + 2/24)*24) = 50
        assert result[0].in_point == 22
        assert result[0].out_point == 50


# ---------------------------------------------------------------------------
# Real-Ollama integration test (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("SCENE_RIPPER_OLLAMA_INTEGRATION") != "1",
    reason="Set SCENE_RIPPER_OLLAMA_INTEGRATION=1 to run real-Ollama checks",
)
def test_ollama_integration_constrained_decode_latency(capsys):
    """Real-Ollama smoke against qwen3:8b.

    The plan's go/no-go threshold for v1 ship: a 1000-word corpus + 20-word
    target should complete in under ~30s. This test records latency as a
    SOFT signal (printed warning) rather than a hard CI gate — the plan
    explicitly documents that latency at corpus scale is unknown and
    deferred to implementation-time measurement.

    Initial measurement on the implementing engineer's M-series Mac
    (qwen3:8b, vocab=1000, target=20): ~169s. That is well above the plan's
    30s threshold, confirming the plan's pre-existing concern that Tier 1
    (Ollama JSON-schema enum) does not parallelize the enum mask. The plan
    explicitly anticipates this: ``llama-cpp-python`` GBNF (Tier 2) is the
    documented follow-up if Tier 1 latency is unacceptable for artistic
    corpora. See docs/plans/2026-05-11-001-feat-word-sequencer-plan.md
    "Deferred to Implementation".

    To avoid noisy CI failures during the deferred-Tier-2 window, this
    test asserts only the *correctness* contract (non-empty result, output
    has a usable shape) and emits latency as a warning when it exceeds the
    plan threshold. Hard-fail conditions are reserved for catastrophic
    timeouts (>5min, which would mask a stuck process).
    """
    # Build a 1000-unique-word corpus by synthesizing per-word
    # TranscriptSegment entries. Use throwaway words ``w0001`` … ``w1000``.
    n_unique = 1000
    words = [(f"w{idx:04d}", idx * 0.01, idx * 0.01 + 0.005) for idx in range(n_unique)]
    clip = MockClip(end_frame=int(n_unique * 0.01 * 24) + 100, transcript=[_seg(words)])

    t0 = time.perf_counter()
    result = generate_llm_word_sequence(
        clips=[(clip, MockSource())],
        prompt="produce twenty distinct words from the vocabulary",
        target_length=20,
        timeout=300.0,
    )
    elapsed = time.perf_counter() - t0

    plan_threshold = 30.0  # seconds (from plan "Deferred to Implementation")
    over_budget = elapsed > plan_threshold

    print(
        f"\n[ollama-integration] vocab={n_unique} target=20 elapsed={elapsed:.2f}s "
        f"got={len(result)} clips threshold={plan_threshold}s "
        f"({'OVER' if over_budget else 'UNDER'} budget)"
    )
    if over_budget:
        # Soft warning surfaces the Tier-2 trigger condition.
        import warnings
        warnings.warn(
            f"Ollama constrained-decode latency {elapsed:.1f}s exceeded "
            f"plan threshold {plan_threshold}s for vocab=1000 target=20 — "
            "Tier 2 (llama-cpp-python GBNF) is the documented follow-up.",
            stacklevel=2,
        )

    # Hard-fail only on catastrophic latency that would mask a stuck process.
    assert elapsed < 300.0, (
        f"Catastrophic constrained-decode latency: {elapsed:.1f}s > 300s. "
        "Likely a stuck Ollama process; check `ollama ps` and the server log."
    )
    assert len(result) > 0, "Real-Ollama integration returned zero clips"
