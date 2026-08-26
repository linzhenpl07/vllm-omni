# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Spectral Self-Anchoring.

The claim under test is narrow and checkable without a checkpoint: the blend
takes the anchor's *low* frequencies and leaves the local branch's high ones
alone. Everything here is CPU and deterministic.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world.spectral_anchoring import (
    AnchorFramePolicy,
    SpectralAnchorConfig,
    gaussian_low_pass,
    spectral_anchor_fuse,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

FRAMES, HEIGHT, WIDTH = 3, 8, 8
HEADS, HEAD_DIM = 2, 4
TOKENS = FRAMES * HEIGHT * WIDTH


def _branch(seed: int) -> torch.Tensor:
    return torch.randn(1, TOKENS, HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(seed))


def _plane(values: torch.Tensor) -> torch.Tensor:
    """Broadcast one value per token across every head and channel."""
    return values.reshape(1, TOKENS, 1, 1).expand(1, TOKENS, HEADS, HEAD_DIM).contiguous()


class TestGaussianLowPass:
    def test_dc_passes_and_high_frequencies_are_suppressed(self):
        envelope = gaussian_low_pass(FRAMES, HEIGHT, WIDTH, sigma_t=0.125, sigma_xy=0.125)
        assert envelope.shape == (FRAMES, HEIGHT, WIDTH)
        assert envelope[0, 0, 0] == pytest.approx(1.0)  # DC untouched
        assert envelope.max() == pytest.approx(1.0)
        # Nyquist on the spatial axes, which is where the noise lives.
        assert envelope[0, HEIGHT // 2, WIDTH // 2] < 1e-6

    def test_a_wider_sigma_keeps_more(self):
        narrow = gaussian_low_pass(FRAMES, HEIGHT, WIDTH, sigma_t=0.125, sigma_xy=0.125)
        wide = gaussian_low_pass(FRAMES, HEIGHT, WIDTH, sigma_t=0.5, sigma_xy=0.5)
        assert wide.sum() > narrow.sum()

    def test_the_envelope_is_symmetric_about_dc(self):
        envelope = gaussian_low_pass(1, HEIGHT, WIDTH, sigma_t=0.125, sigma_xy=0.125)
        assert torch.allclose(envelope[0, 1], envelope[0, -1])
        assert torch.allclose(envelope[0, :, 1], envelope[0, :, -1])


class TestSpectralAnchorFuse:
    def test_zero_weight_is_the_identity(self):
        local, anchor = _branch(0), _branch(1)
        fused = spectral_anchor_fuse(
            local, anchor, frames=FRAMES, height=HEIGHT, width=WIDTH, config=SpectralAnchorConfig(weight=0.0)
        )
        assert torch.equal(fused, local)

    def test_identical_branches_leave_the_output_alone(self):
        local = _branch(0)
        fused = spectral_anchor_fuse(local, local.clone(), frames=FRAMES, height=HEIGHT, width=WIDTH)
        assert torch.allclose(fused, local, atol=1e-5)

    def test_a_dc_offset_is_taken_at_full_blending_weight(self):
        # A constant difference is pure DC, and the filter passes DC untouched,
        # so the blend must move by exactly the weight times the offset.
        local = torch.zeros(1, TOKENS, HEADS, HEAD_DIM)
        anchor = torch.full_like(local, 4.0)
        fused = spectral_anchor_fuse(
            local, anchor, frames=FRAMES, height=HEIGHT, width=WIDTH, config=SpectralAnchorConfig(weight=0.6)
        )
        assert torch.allclose(fused, torch.full_like(local, 2.4), atol=1e-5)

    def test_high_frequency_content_in_the_anchor_is_rejected(self):
        # The whole point of filtering: a frozen anchor cannot know the current
        # detail, so its high frequencies must not reach the output.
        rows = torch.arange(HEIGHT).view(1, -1, 1)
        checker = ((rows + torch.arange(WIDTH).view(1, 1, -1)) % 2).float() * 2.0 - 1.0
        checker = checker.expand(FRAMES, HEIGHT, WIDTH).reshape(-1)

        local = torch.zeros(1, TOKENS, HEADS, HEAD_DIM)
        anchor = _plane(checker)
        fused = spectral_anchor_fuse(local, anchor, frames=FRAMES, height=HEIGHT, width=WIDTH)
        assert fused.abs().max() < 1e-4

    def test_low_frequency_drift_is_corrected_and_local_detail_survives(self):
        # The realistic case, and the one the method is named for: the local
        # branch carries the truth at high frequency and a slow error at low
        # frequency; the anchor carries the correct slow structure and nothing
        # useful at high frequency.
        y = torch.arange(HEIGHT, dtype=torch.float32).view(1, -1, 1)
        ramp = (y / HEIGHT).expand(FRAMES, HEIGHT, WIDTH).reshape(-1)
        detail = torch.randn(TOKENS, generator=torch.Generator().manual_seed(7))

        truth = _plane(ramp)
        drift = _plane(torch.full_like(ramp, 3.0))  # a DC error the anchor knows about
        local = truth + drift + _plane(detail)
        anchor = truth

        fused = spectral_anchor_fuse(
            local, anchor, frames=FRAMES, height=HEIGHT, width=WIDTH, config=SpectralAnchorConfig(weight=1.0)
        )
        # The drift is gone.
        assert (fused - truth - _plane(detail)).abs().max() < 0.5
        assert (local - truth - _plane(detail)).abs().max() == pytest.approx(3.0, abs=1e-4)

    def test_bfloat16_survives_the_round_trip(self):
        local, anchor = _branch(0).bfloat16(), _branch(1).bfloat16()
        fused = spectral_anchor_fuse(local, anchor, frames=FRAMES, height=HEIGHT, width=WIDTH)
        assert fused.dtype == torch.bfloat16
        assert torch.isfinite(fused.float()).all()

    def test_shapes_that_do_not_line_up_are_refused(self):
        local, anchor = _branch(0), _branch(1)
        with pytest.raises(ValueError):
            spectral_anchor_fuse(local, anchor, frames=FRAMES, height=HEIGHT, width=WIDTH + 1)
        with pytest.raises(ValueError):
            spectral_anchor_fuse(local, anchor[:, :-1], frames=FRAMES, height=HEIGHT, width=WIDTH)
        with pytest.raises(ValueError):
            spectral_anchor_fuse(local[0], anchor[0], frames=FRAMES, height=HEIGHT, width=WIDTH)


class TestSpectralAnchorConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sigma_t": 0.0},
            {"sigma_xy": -1.0},
            {"weight": 1.5},
            {"weight": -0.1},
            {"anchor_capacity": 0},
            {"frames_per_anchor": 0},
            {"denoising_steps": -1},
        ],
    )
    def test_nonsense_is_refused_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            SpectralAnchorConfig(**kwargs)


class TestAnchorFramePolicy:
    def test_it_takes_one_frame_in_three_until_it_is_full(self):
        policy = AnchorFramePolicy()
        taken = [index for index in range(30) if policy.offer(index)]
        assert taken == [0, 3, 6, 9, 12, 15]
        assert policy.frozen

    def test_it_freezes_rather_than_following_the_drift(self):
        # Refreshing the anchor from recent frames would anchor the session to
        # its own degradation, which corrects nothing.
        policy = AnchorFramePolicy()
        for index in range(200):
            policy.offer(index)
        assert policy.collected == [0, 3, 6, 9, 12, 15]
        assert policy.frames_seen == 200

    def test_frames_must_arrive_in_order(self):
        policy = AnchorFramePolicy()
        policy.offer(0)
        with pytest.raises(ValueError):
            policy.offer(5)

    def test_the_correction_waits_for_the_pretrained_horizon(self):
        policy = AnchorFramePolicy()
        pretrained = 21
        for index in range(pretrained):
            policy.offer(index)
            assert not policy.active(pretrained)
        # The cache filled long ago; what was still missing is having run past
        # the horizon the model was trained for.
        assert policy.frozen
        policy.offer(pretrained)
        assert policy.active(pretrained)
