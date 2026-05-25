# Visual Match

Visual Match uses DINOv2 embeddings to compare clips by visual appearance.

## Model

Scene Ripper currently uses `facebook/dinov2-base`, stored on clips as the model tag `dinov2-vit-b-14`. The embedding vector is the normalized DINOv2 CLS token and has 768 dimensions.

## Sampling

For whole-clip visual similarity, Scene Ripper embeds each clip thumbnail. If embeddings are missing, the batch embedding path reads each clip's `thumbnail_path`, loads the image, and computes one vector per clip.

For Match Cut, Scene Ripper uses a different boundary-frame path. It extracts the first frame and the last frame of the clip from the source video with FFmpeg and computes separate DINOv2 vectors for those boundary frames.

## Limitations

DINOv2 visual similarity is image-based. It can capture composition, color, texture, objects, and overall visual structure, but it does not understand editing intent, narrative role, dialogue, or motion continuity. Matches can therefore differ from human perception, especially when clips share surface texture but not meaning, or when motion is the main reason two clips feel related.

Zero-vector or missing embeddings should be treated as unavailable for similarity ranking. Run the Embeddings analysis operation before using Visual Match on newly imported clips.
