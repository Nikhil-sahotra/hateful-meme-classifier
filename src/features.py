"""
src/features.py
Stage 2: frozen CLIP feature extraction.

Loads a frozen, pretrained CLIP model (image + text encoders) and provides
functions to embed images and text into CLIP's shared semantic space.
No gradients ever flow into CLIP -- it's used purely as a feature extractor.

Note: different transformers versions have varied in whether
get_image_features()/get_text_features() return a raw tensor or a structured
output object. _unwrap() below handles both so this code isn't tied to one
specific transformers version.
"""

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():
    """Prefers Apple Silicon's MPS backend, falls back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unwrap(output):
    """
    Normalizes the return value of get_image_features/get_text_features
    across transformers versions -- some return a plain Tensor, others
    return a structured output object with the tensor under an attribute.
    """
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None:
            if val.dim() == 3:  # e.g. last_hidden_state [batch, seq, dim] -> pool
                val = val.mean(dim=1)
            return val
    raise TypeError(
        f"Unexpected output type from CLIP feature method: {type(output)}. "
        f"Available attributes: {[a for a in dir(output) if not a.startswith('_')]}"
    )


class CLIPFeatureExtractor:
    """
    Wraps a frozen CLIP model + processor. Call .embed_images() and
    .embed_texts() to get L2-normalized embedding tensors.
    """

    def __init__(self, model_name=MODEL_NAME, device=None, attn_implementation=None):
        """
        attn_implementation: pass "eager" when you need output_attentions=True
        to actually work (e.g. for Attention Rollout in explain.py) -- newer
        transformers defaults to "sdpa", which is faster but cannot return
        attention weights at all. Leave as None for normal embedding extraction.
        """
        self.device = device or get_device()
        print(f"Loading {model_name} onto {self.device} ...")
        kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
        self.model = CLIPModel.from_pretrained(model_name, **kwargs).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # frozen -- eval mode, no dropout etc.
        for p in self.model.parameters():
            p.requires_grad = False  # belt-and-suspenders: no gradients into CLIP, ever

    @torch.no_grad()
    def embed_images(self, images):
        """images: list of PIL.Image (RGB). Returns L2-normalized tensor [N, 512]."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        raw = self.model.get_image_features(**inputs)
        embeds = _unwrap(raw)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.cpu()

    @torch.no_grad()
    def embed_texts(self, texts, max_length=40):
        """texts: list of str. Returns L2-normalized tensor [N, 512]."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.device)
        raw = self.model.get_text_features(**inputs)
        embeds = _unwrap(raw)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.cpu()
