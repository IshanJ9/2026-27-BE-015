"""
video_to_audio.py — Restore corrupted audio segments using video (lip) information.

Pipeline:
1. Convert audio to STFT spectrogram
2. Identify corrupted segments from the corruption mask
3. Use spectral interpolation + Griffin-Lim to reconstruct audio
4. Apply video-derived energy envelope from mouth motion for natural prosody

The mouth motion (optical flow of lip regions) provides an energy envelope
that modulates the reconstructed audio to match expected speech patterns.
"""

import cv2
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


class VideoToAudioRestorer:
    """
    Restores corrupted audio segments using spectral interpolation
    with Griffin-Lim reconstruction, modulated by video-derived
    mouth motion energy.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_iter_griffin_lim: int = 60,
        interpolation_context: int = 5,
    ):
        """
        Args:
            n_fft: FFT window size
            hop_length: STFT hop length
            win_length: STFT window length
            n_iter_griffin_lim: Number of Griffin-Lim iterations
            interpolation_context: Number of context frames for spectral interpolation
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_iter_griffin_lim = n_iter_griffin_lim
        self.interpolation_context = interpolation_context

    def _compute_mouth_motion_energy(
        self, frames: np.ndarray, mouth_bboxes: list, fps: float, sr: int
    ) -> np.ndarray:
        """
        Compute mouth motion energy from consecutive frames using optical flow.
        Returns an energy envelope aligned to audio sample rate.

        Args:
            frames: (N, H, W, 3) uint8 BGR
            mouth_bboxes: list of (x1, y1, x2, y2) or None
            fps: video frame rate
            sr: audio sample rate

        Returns:
            1-D energy envelope array of length (N_frames * sr / fps)
        """
        n_frames = len(frames)
        frame_energies = np.zeros(n_frames)

        for i in range(1, n_frames):
            bbox = mouth_bboxes[i] if mouth_bboxes[i] is not None else mouth_bboxes[i - 1]
            prev_bbox = mouth_bboxes[i - 1] if mouth_bboxes[i - 1] is not None else bbox

            if bbox is None or prev_bbox is None:
                frame_energies[i] = frame_energies[i - 1]
                continue

            # Extract mouth regions
            x1, y1, x2, y2 = bbox
            px1, py1, px2, py2 = prev_bbox

            curr_mouth = cv2.cvtColor(frames[i, y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            prev_mouth = cv2.cvtColor(frames[i - 1, py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)

            if curr_mouth.size == 0 or prev_mouth.size == 0:
                continue

            # Resize to common size
            target_size = (64, 32)
            curr_resized = cv2.resize(curr_mouth, target_size)
            prev_resized = cv2.resize(prev_mouth, target_size)

            # Compute absolute difference as motion energy
            diff = np.abs(curr_resized.astype(float) - prev_resized.astype(float))
            frame_energies[i] = np.mean(diff)

        # Normalize energy to [0, 1]
        max_energy = frame_energies.max()
        if max_energy > 0:
            frame_energies /= max_energy

        # Smooth the energy curve
        kernel_size = max(3, int(fps * 0.1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        frame_energies = np.convolve(frame_energies, kernel, mode="same")

        # Upsample to audio sample rate
        total_audio_samples = int(n_frames / fps * sr)
        audio_energy = np.interp(
            np.linspace(0, n_frames - 1, total_audio_samples),
            np.arange(n_frames),
            frame_energies,
        )

        logger.info(
            f"Mouth motion energy: mean={audio_energy.mean():.3f}, "
            f"max={audio_energy.max():.3f}"
        )

        return audio_energy

    def _spectral_interpolation(
        self, magnitude: np.ndarray, mask_frames: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate corrupted spectrogram frames using context from
        surrounding uncorrupted frames.

        Args:
            magnitude: (freq_bins, time_frames) spectrogram magnitude
            mask_frames: boolean array (time_frames,) — True = corrupted

        Returns:
            Interpolated magnitude spectrogram
        """
        result = magnitude.copy()
        n_bins, n_frames = magnitude.shape

        # Find contiguous corrupted segments
        segments = []
        in_segment = False
        seg_start = 0

        for i in range(n_frames):
            if mask_frames[i] and not in_segment:
                seg_start = i
                in_segment = True
            elif not mask_frames[i] and in_segment:
                segments.append((seg_start, i))
                in_segment = False
        if in_segment:
            segments.append((seg_start, n_frames))

        ctx = self.interpolation_context

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start

            # Get context frames before and after the gap
            before_start = max(0, seg_start - ctx)
            after_end = min(n_frames, seg_end + ctx)

            before_ctx = magnitude[:, before_start:seg_start]
            after_ctx = magnitude[:, seg_end:after_end]

            if before_ctx.shape[1] > 0 and after_ctx.shape[1] > 0:
                # Linear interpolation between mean of context
                before_mean = before_ctx.mean(axis=1, keepdims=True)
                after_mean = after_ctx.mean(axis=1, keepdims=True)

                weights = np.linspace(0, 1, seg_len).reshape(1, -1)
                interpolated = before_mean * (1 - weights) + after_mean * weights
                result[:, seg_start:seg_end] = interpolated

            elif before_ctx.shape[1] > 0:
                result[:, seg_start:seg_end] = before_ctx.mean(
                    axis=1, keepdims=True
                )
            elif after_ctx.shape[1] > 0:
                result[:, seg_start:seg_end] = after_ctx.mean(
                    axis=1, keepdims=True
                )

        logger.info(f"Interpolated {len(segments)} corrupted segments")
        return result

    def _griffin_lim(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Reconstruct audio from magnitude spectrogram using Griffin-Lim.

        Args:
            magnitude: (freq_bins, time_frames)

        Returns:
            1-D float32 audio waveform
        """
        audio = librosa.griffinlim(
            magnitude,
            n_iter=self.n_iter_griffin_lim,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
        )

        logger.info(f"Griffin-Lim: {len(audio)} samples reconstructed")
        return audio

    def restore(self, corrupted_data: dict) -> np.ndarray:
        """
        Main audio restoration pipeline.

        Args:
            corrupted_data: dict from StreamCorruptor.corrupt()

        Returns:
            restored_audio: 1-D float32 numpy array
        """
        corrupted_audio = corrupted_data["corrupted_audio"]
        audio_mask = corrupted_data["audio_corruption_mask"]
        original_audio = corrupted_data["original_audio"]
        frames = corrupted_data["original_frames"]
        mouth_bboxes = corrupted_data["mouth_bboxes"]
        fps = corrupted_data["fps"]
        sr = corrupted_data["sr"]

        if not audio_mask.any():
            logger.info("No audio corruption detected, returning original")
            return corrupted_audio.copy()

        logger.info("Starting audio restoration...")

        # Step 1: Compute STFT of corrupted audio
        stft = librosa.stft(
            corrupted_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Step 2: Determine which STFT frames are corrupted
        n_stft_frames = magnitude.shape[1]
        n_audio_samples = len(corrupted_audio)

        # Map audio sample mask to STFT frame mask
        stft_frame_mask = np.zeros(n_stft_frames, dtype=bool)
        for t in range(n_stft_frames):
            sample_start = t * self.hop_length
            sample_end = min(sample_start + self.win_length, n_audio_samples)
            if sample_start < n_audio_samples:
                # Mark frame as corrupted if >50% of its samples are corrupted
                segment_mask = audio_mask[sample_start:sample_end]
                if segment_mask.mean() > 0.5:
                    stft_frame_mask[t] = True

        corrupted_frames_count = stft_frame_mask.sum()
        logger.info(
            f"STFT: {n_stft_frames} frames, "
            f"{corrupted_frames_count} corrupted ({corrupted_frames_count/n_stft_frames:.1%})"
        )

        # Step 3: Spectral interpolation for magnitude
        restored_magnitude = self._spectral_interpolation(magnitude, stft_frame_mask)

        # Step 4: Reconstruct with Griffin-Lim (uses only magnitude)
        reconstructed = self._griffin_lim(restored_magnitude)

        # Step 5: Compute mouth motion energy for prosody modulation
        motion_energy = self._compute_mouth_motion_energy(
            frames, mouth_bboxes, fps, sr
        )

        # Trim/pad to match reconstructed length
        target_len = len(reconstructed)
        if len(motion_energy) > target_len:
            motion_energy = motion_energy[:target_len]
        elif len(motion_energy) < target_len:
            motion_energy = np.pad(
                motion_energy, (0, target_len - len(motion_energy))
            )

        # Step 6: Apply video-derived energy modulation to corrupted segments
        # Map audio mask to reconstructed signal length
        if len(audio_mask) > target_len:
            recon_mask = audio_mask[:target_len]
        else:
            recon_mask = np.pad(audio_mask, (0, max(0, target_len - len(audio_mask))))

        # Boost reconstructed signal energy using mouth motion
        energy_scale = motion_energy * 0.5 + 0.5  # Scale to [0.5, 1.0]
        reconstructed = reconstructed * energy_scale

        # Step 7: Blend reconstructed with original
        # Use original where not corrupted, reconstructed where corrupted
        restored = corrupted_audio.copy()
        if len(restored) > target_len:
            restored = restored[:target_len]
        elif len(restored) < target_len:
            restored = np.pad(restored, (0, target_len - len(restored)))

        # Crossfade at boundaries for smooth transitions
        crossfade_samples = int(0.005 * sr)  # 5ms crossfade

        # Find corruption boundaries
        mask_diff = np.diff(recon_mask.astype(int))
        starts = np.where(mask_diff == 1)[0] + 1
        ends = np.where(mask_diff == -1)[0] + 1

        # Handle edge cases
        if recon_mask[0]:
            starts = np.concatenate([[0], starts])
        if recon_mask[-1]:
            ends = np.concatenate([ends, [len(recon_mask)]])

        for s, e in zip(starts, ends):
            s = int(s)
            e = int(e)
            if e > len(reconstructed):
                e = len(reconstructed)

            # Fill corrupted region with reconstruction
            restored[s:e] = reconstructed[s:e]

            # Apply crossfade at start boundary
            cf_start = max(0, s - crossfade_samples)
            if cf_start < s:
                fade_len = s - cf_start
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = 1 - fade_in
                restored[cf_start:s] = (
                    restored[cf_start:s] * fade_out
                    + reconstructed[cf_start:s] * fade_in
                )

            # Apply crossfade at end boundary
            cf_end = min(len(restored), e + crossfade_samples)
            if cf_end > e:
                fade_len = cf_end - e
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = 1 - fade_out
                restored[e:cf_end] = (
                    restored[e:cf_end] * fade_in
                    + reconstructed[e:cf_end] * fade_out
                )

        # Normalize to prevent clipping
        max_val = np.abs(restored).max()
        if max_val > 1.0:
            restored = restored / max_val * 0.95

        logger.info(f"Audio restoration complete: {len(restored)} samples")
        return restored
