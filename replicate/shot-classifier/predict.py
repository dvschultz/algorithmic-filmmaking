"""VideoMAE Shot Type Classifier for Replicate.

Classifies video clips into 5 cinematographic shot types:
- LS: Long Shot (wide, establishing)
- FS: Full Shot (full body visible)
- MS: Medium Shot (waist up)
- CS: Close-up Shot (head and shoulders)
- ECS: Extreme Close-up Shot (face detail only)

Based on fine-tuned VideoMAE model.
"""

import tempfile
from pathlib import Path
from typing import Optional

import av
import numpy as np
import torch
from cog import BasePredictor, Input, Path as CogPath
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

# Shot type labels (must match model config.json id2label)
SHOT_TYPES = {
    0: "CS",   # Close-up Shot
    1: "ECS",  # Extreme Close-up Shot
    2: "FS",   # Full Shot
    3: "LS",   # Long Shot
    4: "MS",   # Medium Shot
}

SHOT_TYPE_DISPLAY = {
    "LS": "Long Shot",
    "FS": "Full Shot",
    "MS": "Medium Shot",
    "CS": "Close-up",
    "ECS": "Extreme Close-up",
}


class Predictor(BasePredictor):
    """VideoMAE shot type classifier."""

    def setup(self):
        """Load model into memory for fast inference."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "/src/model"  # Bundled with container via COPY . /src

        print(f"Loading model from {model_path} on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def read_video_pyav(self, container, indices: np.ndarray) -> np.ndarray:
        """Decode video frames at specified indices using PyAV.

        Args:
            container: PyAV container
            indices: Frame indices to extract

        Returns:
            numpy array of shape (num_frames, height, width, 3)
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]

        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)

        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frame_indices(
        self,
        clip_len: int,
        frame_sample_rate: int,
        seg_len: int,
    ) -> np.ndarray:
        """Sample frame indices for VideoMAE.

        VideoMAE expects 16 frames sampled from the video.

        Args:
            clip_len: Number of frames to sample (16 for VideoMAE)
            frame_sample_rate: Sampling rate
            seg_len: Total frames in video

        Returns:
            Array of frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)

        if seg_len < converted_len:
            # Video too short - sample what we have with interpolation
            indices = np.linspace(0, seg_len - 1, num=clip_len)
        else:
            # Random starting point for variety
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len
            indices = np.linspace(start_idx, end_idx, num=clip_len)

        return np.clip(indices, 0, seg_len - 1).astype(np.int64)

    def predict(
        self,
        video: CogPath = Input(description="Video file to classify"),
        return_all_scores: bool = Input(
            description="Return scores for all shot types",
            default=False,
        ),
    ) -> dict:
        """Classify shot type from video clip.

        Args:
            video: Path to video file
            return_all_scores: If True, include all class scores

        Returns:
            Dictionary with shot_type, confidence, and optionally all scores
        """
        # Open video
        container = av.open(str(video))
        stream = container.streams.video[0]

        # Get total frames
        total_frames = stream.frames
        if total_frames == 0:
            # Estimate from duration
            if stream.duration and stream.time_base:
                duration_sec = float(stream.duration * stream.time_base)
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                total_frames = int(duration_sec * fps)
            else:
                total_frames = 100  # Fallback

        # Sample 16 frames (VideoMAE requirement)
        indices = self.sample_frame_indices(
            clip_len=16,
            frame_sample_rate=1,
            seg_len=total_frames,
        )

        # Extract frames
        video_frames = self.read_video_pyav(container, indices)
        container.close()

        # Process for model
        inputs = self.processor(list(video_frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_idx = logits.argmax(-1).item()

        shot_type = SHOT_TYPES.get(predicted_idx, f"unknown_{predicted_idx}")
        confidence = probs[predicted_idx].item()

        result = {
            "shot_type": shot_type,
            "shot_type_display": SHOT_TYPE_DISPLAY.get(shot_type, shot_type),
            "confidence": round(confidence, 4),
        }

        if return_all_scores:
            result["all_scores"] = {
                SHOT_TYPES[i]: round(probs[i].item(), 4)
                for i in range(len(probs))
            }

        return result
