"""
app.py -- Gradio app for Hugging Face Spaces.

Self-contained (doesn't import from src/) so the Space has minimal
dependencies -- only torch, transformers, gradio, pillow, matplotlib, numpy.
No datasets/huggingface_hub bulk downloading or sklearn needed for inference.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import spaces
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Model definition (same architecture as src/model.py) ----------
class FusionClassifierHead(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        fused_dim = embed_dim * 2 + embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def fuse(self, image_embeds, text_embeds):
        concat = torch.cat([image_embeds, text_embeds], dim=-1)
        product = image_embeds * text_embeds
        return torch.cat([concat, product], dim=-1)

    def forward(self, image_embeds, text_embeds):
        fused = self.fuse(image_embeds, text_embeds)
        return self.classifier(fused).squeeze(-1)


# ---------- Load models once at startup ----------
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

print("Loading trained classifier head...")
classifier = FusionClassifierHead().to(device)
classifier.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
classifier.eval()


# ---------- Feature extraction ----------
def _unwrap(output):
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None:
            if val.dim() == 3:
                val = val.mean(dim=1)
            return val
    raise TypeError(f"Unexpected CLIP output type: {type(output)}")


@torch.no_grad()
def embed_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    raw = clip_model.get_image_features(**inputs)
    embeds = _unwrap(raw)
    return embeds / embeds.norm(dim=-1, keepdim=True)


def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=40).to(device)
    raw = clip_model.get_text_features(**inputs)
    embeds = _unwrap(raw)
    return embeds / embeds.norm(dim=-1, keepdim=True)


# ---------- Explainability ----------
@torch.no_grad()
def attention_rollout_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True)
    attentions = vision_outputs.attentions

    seq_len = attentions[0].size(-1)
    result = torch.eye(seq_len, device=device)
    for attn in attentions:
        attn_avg = attn.mean(dim=1)[0]
        attn_avg = attn_avg + torch.eye(seq_len, device=device)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = attn_avg @ result

    cls_attention = result[0, 1:]
    grid_size = int(cls_attention.size(0) ** 0.5)
    return cls_attention.reshape(grid_size, grid_size).cpu().numpy()


def text_token_importance(image_embeds, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=40).to(device)
    input_ids = inputs["input_ids"]

    embedding_layer = clip_model.text_model.embeddings.token_embedding
    captured = {}

    def hook_fn(module, inp, output):
        output.requires_grad_(True)
        output.retain_grad()
        captured["embeds"] = output

    handle = embedding_layer.register_forward_hook(hook_fn)
    try:
        raw = clip_model.get_text_features(**inputs)
    finally:
        handle.remove()

    text_embeds = _unwrap(raw)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    logit = classifier(image_embeds, text_embeds)
    classifier.zero_grad()
    logit.backward()

    grad = captured["embeds"].grad
    embeds = captured["embeds"].detach()
    importance = (grad * embeds).sum(dim=-1).abs()[0]

    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
    special = set(processor.tokenizer.all_special_tokens)

    kept_tokens, kept_scores = [], []
    for tok, score in zip(tokens, importance.tolist()):
        if tok in special:
            continue
        kept_tokens.append(tok.replace("</w>", ""))
        kept_scores.append(score)
    return kept_tokens, kept_scores


def make_heatmap_overlay(image, heatmap):
    heatmap_img = Image.fromarray((heatmap / heatmap.max() * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.imshow(heatmap_arr, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    overlay = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return overlay


def make_token_chart(tokens, scores):
    order = np.argsort(scores)
    sorted_tokens = [tokens[i] for i in order]
    sorted_scores = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(5, max(2, len(tokens) * 0.4)))
    ax.barh(sorted_tokens, sorted_scores, color="#4C72B0")
    ax.set_xlabel("Importance (|gradient x input|)")
    fig.tight_layout()
    fig.canvas.draw()
    chart = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return chart


# ---------- Main prediction function ----------
@spaces.GPU  # ZeroGPU: allocates a shared GPU only for the duration of this call; effect-free on non-ZeroGPU hardware
def classify_and_explain(image, text):
    if image is None or not text or not text.strip():
        return "Please provide both an image and text.", None, None

    image = image.convert("RGB")
    image_embeds = embed_image(image)
    text_embeds_for_pred = embed_text(text)

    with torch.no_grad():
        logit = classifier(image_embeds, text_embeds_for_pred)
        prob = torch.sigmoid(logit).item()

    verdict = "HATEFUL" if prob >= 0.5 else "NOT HATEFUL"
    result_text = f"**Prediction: {verdict}**\n\nP(hateful) = {prob:.3f}"

    heatmap = attention_rollout_image(image)
    overlay = make_heatmap_overlay(image, heatmap)

    tokens, scores = text_token_importance(image_embeds, text)
    chart = make_token_chart(tokens, scores)

    return result_text, overlay, chart


# ---------- Gradio interface ----------
with gr.Blocks(title="Multimodal Hateful Content Classifier") as demo:
    gr.Markdown(
        "# Multimodal Hateful Content Classifier\n"
        "Upload a meme (image + text) to classify it, with explainability showing "
        "which image regions and text tokens drove the prediction.\n\n"
        "Built on frozen CLIP (ViT-B/32) with a trained fusion classifier head. "
        "Explanations use Attention Rollout (image) and Gradient x Input (text).\n\n"
        "*Note: trained on the Hateful Memes dataset for a portfolio/research project -- "
        "not a production content moderation system.*"
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Meme Image")
            text_input = gr.Textbox(label="Meme Text", placeholder="Enter the text overlay on the meme...")
            submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            result_output = gr.Markdown(label="Result")
            heatmap_output = gr.Image(label="Image Attention (Attention Rollout)")
            chart_output = gr.Image(label="Text Token Importance")

    submit_btn.click(
        fn=classify_and_explain,
        inputs=[image_input, text_input],
        outputs=[result_output, heatmap_output, chart_output],
    )

if __name__ == "__main__":
    demo.launch()
