"""
src/explain.py
Stage 5: explainability layer.

Image side: Attention Rollout (Abnar & Zuidema, 2020) over CLIP's vision
transformer -- traces how attention propagates from the [CLS] token down to
image patches across all layers, producing a heatmap. No gradients needed,
so this works cleanly on the frozen backbone as-is.

Text side: Gradient x Input over token embeddings -- runs a single forward
pass with gradients enabled just for this explanation (not training), then
backpropagates from the trained classifier's logit back to the token
embedding layer to score each token's contribution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


@torch.no_grad()
def attention_rollout_image(clip_model, processor, image, device):
    """
    Returns a [grid_size, grid_size] numpy array of patch importance scores,
    derived from the [CLS] token's rolled-out attention to all patches.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True)
    attentions = vision_outputs.attentions  # tuple of [1, heads, seq, seq], one per layer

    seq_len = attentions[0].size(-1)
    result = torch.eye(seq_len, device=device)

    for attn in attentions:
        attn_avg = attn.mean(dim=1)[0]                      # average over heads -> [seq, seq]
        attn_avg = attn_avg + torch.eye(seq_len, device=device)  # add residual connection (rollout requires this)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = attn_avg @ result

    cls_attention = result[0, 1:]  # CLS token's rolled-out attention to each patch (excludes CLS itself)
    num_patches = cls_attention.size(0)
    grid_size = int(num_patches ** 0.5)
    heatmap = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    return heatmap


def text_token_importance(clip_model, processor, classifier_head, image_embeds, text, device):
    """
    Returns (tokens: list[str], scores: list[float]) -- one importance score
    per text token, via gradient x input on the token embedding layer.
    Special tokens (BOS/EOS/padding) are excluded from the output.
    """
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=40).to(device)
    input_ids = inputs["input_ids"]

    embedding_layer = clip_model.text_model.embeddings.token_embedding
    captured = {}

    def hook_fn(module, inp, output):
        # CLIP's embedding weights are frozen (requires_grad=False), so the
        # embedding lookup output has no gradient-requiring input feeding it
        # and comes out with requires_grad=False by default. We explicitly
        # turn it on here, just for this one explanation pass -- this does
        # NOT unfreeze or modify the actual embedding weights themselves.
        output.requires_grad_(True)
        output.retain_grad()
        captured["embeds"] = output

    handle = embedding_layer.register_forward_hook(hook_fn)
    try:
        raw = clip_model.get_text_features(**inputs)
    finally:
        handle.remove()

    text_embeds = raw if isinstance(raw, torch.Tensor) else getattr(raw, "text_embeds", None)
    if text_embeds is None:
        text_embeds = getattr(raw, "pooler_output")
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logit = classifier_head(image_embeds, text_embeds)
    classifier_head.zero_grad()
    logit.backward()

    grad = captured["embeds"].grad          # [1, seq, dim]
    embeds = captured["embeds"].detach()    # [1, seq, dim]
    importance = (grad * embeds).sum(dim=-1).abs()[0]  # [seq]

    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
    special = set(processor.tokenizer.all_special_tokens)

    kept_tokens, kept_scores = [], []
    for tok, score in zip(tokens, importance.tolist()):
        if tok in special:
            continue
        kept_tokens.append(tok.replace("</w>", ""))  # CLIP's BPE tokenizer marks word-end with </w>
        kept_scores.append(score)

    return kept_tokens, kept_scores


def predict(classifier_head, image_embeds, text_embeds):
    """Returns predicted probability of 'hateful' for one example."""
    with torch.no_grad():
        logit = classifier_head(image_embeds, text_embeds)
        prob = torch.sigmoid(logit).item()
    return prob


def plot_explanation(image, heatmap, tokens, scores, prob, true_label, save_path=None):
    """
    Produces a 2-panel figure: image with attention-rollout heatmap overlay,
    and a horizontal bar chart of text token importance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: image + heatmap overlay
    heatmap_resized = np.array(
        Image.fromarray((heatmap / heatmap.max() * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    )
    axes[0].imshow(image)
    axes[0].imshow(heatmap_resized, cmap="jet", alpha=0.45)
    axes[0].axis("off")
    label_str = "Hateful" if true_label == 1 else "Not Hateful"
    axes[0].set_title(f"True label: {label_str} | Predicted P(hateful): {prob:.3f}", fontsize=10)

    # Panel 2: token importance bar chart
    order = np.argsort(scores)
    sorted_tokens = [tokens[i] for i in order]
    sorted_scores = [scores[i] for i in order]
    axes[1].barh(sorted_tokens, sorted_scores, color="#4C72B0")
    axes[1].set_xlabel("Importance (|gradient x input|)")
    axes[1].set_title("Text token contribution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
