"""
src/data.py
Step 1 of the pipeline: load the Hateful Memes dataset (image + text + label)
via the Hugging Face mirror, and provide a PyTorch Dataset wrapper.

Known issue: the 'neuralcatcher/hateful_memes' mirror only ships 9,664 image
files total, short of what all splits combined require. This module filters
every split down to rows whose image file actually exists locally, and
reports the retention rate so it's always visible, not silently dropped.
"""

import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from PIL import Image

REPO_ID = "neuralcatcher/hateful_memes"


class HatefulMemesDataset(Dataset):
    """
    Wraps a Hugging Face 'neuralcatcher/hateful_memes' split (already filtered
    to only rows with an existing image file -- see load_all()).

    Each item returns:
        image: PIL.Image (RGB)
        text: str
        label: int (0 = not hateful, 1 = hateful)
    """

    def __init__(self, df, images_root):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row["img"])
        image = Image.open(img_path).convert("RGB")
        text = row["text"]
        label = int(row["label"])
        return image, text, label


def download_images():
    """
    Downloads (or resumes) the img/ folder from the dataset repo.
    Requires `hf auth login` to have been run once for a reasonable rate limit.
    """
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["img/*"],
    )


def filter_to_existing(df, existing_files):
    """Keeps only rows whose image file actually exists locally, and reports the retention rate."""
    mask = df["img"].apply(lambda p: os.path.basename(p) in existing_files)
    cleaned = df[mask].reset_index(drop=True)
    print(f"  {len(cleaned)} / {len(df)} usable ({len(cleaned) / len(df) * 100:.1f}%)")
    return cleaned


def load_all():
    """
    Loads every split (train/validation/test), downloads images, and filters
    each split down to rows with an actually-present image file.

    Returns a dict of clean pandas DataFrames: {"train": ..., "validation": ..., "test": ...}
    plus the images_root path (needed to build HatefulMemesDataset instances).
    """
    ds = load_dataset(REPO_ID)
    images_root = download_images()
    existing_files = set(os.listdir(os.path.join(images_root, "img")))

    clean = {}
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        print(f"Filtering split '{split_name}':")
        clean[split_name] = filter_to_existing(df, existing_files)

    return clean, images_root


if __name__ == "__main__":
    clean_splits, images_root = load_all()

    train_dataset = HatefulMemesDataset(clean_splits["train"], images_root)
    print(f"\nFinal usable train set size: {len(train_dataset)}")

    image, text, label = train_dataset[0]
    print("Sample text:", text)
    print("Sample label:", label, "(0=not hateful, 1=hateful)")
    print("Sample image size:", image.size)

    os.makedirs("outputs", exist_ok=True)
    image.save("outputs/sample_meme_check.png")
    print("Saved a sample image to outputs/sample_meme_check.png")
