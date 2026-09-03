"""
src/extract_embeddings.py
Precomputes and caches CLIP embeddings for every split, so Stage 3-4 training
never has to re-run the CLIP backbone. Run this once; re-run only if the
dataset or CLIP model choice changes.

Output: outputs/embeddings/{split}_embeddings.npz containing
    image_embeds: [N, 512] float32
    text_embeds:  [N, 512] float32
    labels:       [N]      int64
    ids:          [N]      the original meme ids, for traceability
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data import load_all, HatefulMemesDataset
from src.features import CLIPFeatureExtractor

BATCH_SIZE = 32
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "embeddings")


def extract_split(extractor, df, images_root, split_name):
    all_image_embeds = []
    all_text_embeds = []
    all_labels = []
    all_ids = []

    n = len(df)
    for start in tqdm(range(0, n, BATCH_SIZE), desc=f"Embedding {split_name}"):
        batch = df.iloc[start:start + BATCH_SIZE]
        images = [Image.open(os.path.join(images_root, p)).convert("RGB") for p in batch["img"]]
        texts = list(batch["text"])

        img_embeds = extractor.embed_images(images)
        txt_embeds = extractor.embed_texts(texts)

        all_image_embeds.append(img_embeds.numpy())
        all_text_embeds.append(txt_embeds.numpy())
        all_labels.extend(batch["label"].tolist())
        all_ids.extend(batch["id"].tolist() if "id" in batch.columns else batch.index.tolist())

    image_embeds = np.concatenate(all_image_embeds, axis=0)
    text_embeds = np.concatenate(all_text_embeds, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    ids = np.array(all_ids)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_embeddings.npz")
    np.savez(out_path, image_embeds=image_embeds, text_embeds=text_embeds, labels=labels, ids=ids)
    print(f"Saved {split_name}: {image_embeds.shape[0]} examples -> {out_path}")


def load_cached_embeddings(split_name):
    """Convenience loader for Stage 3-4: returns (image_embeds, text_embeds, labels, ids)."""
    path = os.path.join(OUTPUT_DIR, f"{split_name}_embeddings.npz")
    data = np.load(path)
    return data["image_embeds"], data["text_embeds"], data["labels"], data["ids"]


if __name__ == "__main__":
    clean_splits, images_root = load_all()
    extractor = CLIPFeatureExtractor()

    for split_name, df in clean_splits.items():
        extract_split(extractor, df, images_root, split_name)

    print("\nAll splits embedded and cached. Stage 3-4 can now train directly on these .npz files.")
