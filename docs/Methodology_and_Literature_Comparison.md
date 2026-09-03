# Multimodal Hateful Content Classifier — Methodology & Literature Comparison

## 1. Chosen Methodology: "Frozen-CLIP Fusion with Post-Hoc Explainability" (FCF-XAI)

This is a **five-stage pipeline**, deliberately scoped to be buildable in 1-2 focused days while still producing a defensible, interview-ready artifact.

### Stage 1 — Data Pipeline
- Dataset: Facebook AI's Hateful Memes Challenge dataset (~10k memes: train / dev_seen / dev_unseen / test).
- Preprocessing: images resized/normalized using CLIP's native preprocessing transforms; meme text tokenized using CLIP's tokenizer (no OCR needed — the dataset already ships text as metadata).

### Stage 2 — Feature Extraction (Frozen Backbone)
- Use a **frozen, pretrained CLIP** (ViT-B/32) for both the image encoder and text encoder.
- Output: image embedding `e_i` and text embedding `e_t`, same dimensionality, already in a shared semantic space (this is CLIP's whole point — it was contrastively trained so image and text embeddings live in the same space).
- **Why frozen, not fine-tuned:** fine-tuning a ViT backbone on ~8.5k training examples in 1-2 days, on likely limited compute, risks overfitting and eats your whole time budget on training rather than on the explainability layer that actually differentiates this project.

### Stage 3 — Fusion
- Combine `e_i` and `e_t` two ways simultaneously: **concatenation** `[e_i ; e_t]` (preserves each modality's independent signal) and **element-wise product** `e_i ⊙ e_t` (captures cross-modal interaction — this is what catches the "benign alone, hateful together" cases).
- Concatenate both combined vectors into one fused representation.

### Stage 4 — Classification Head
- Small 2-layer MLP (fused vector → hidden layer with dropout → single logit).
- Binary cross-entropy loss, class-weighted to handle the dataset's label imbalance.
- Only this MLP head is trained — CLIP stays frozen throughout.

### Stage 5 — Explainability Layer
- **Visual side:** attention-rollout / Grad-CAM-style visualization over CLIP's vision transformer to produce a heatmap showing which image regions drove the prediction.
- **Textual side:** gradient × input or attention-weight scores over text tokens, highlighting which words contributed most.
- Both are overlaid on 3-5 example memes in the README as qualitative evidence — this is the piece that directly demonstrates the XAI angle for the T&S application.

### Evaluation
- Metrics: AUROC, Accuracy, Macro-F1 on dev_seen and dev_unseen — these are the standard metrics used across essentially every paper in this space, so your numbers are directly comparable to published baselines.

---

## 2. Comparison Against Surveyed Literature

| Work | Backbone / Approach | Fusion Strategy | Explainability | Compute Cost | Verdict for This Project |
|---|---|---|---|---|---|
| **Kiela et al. 2020** (Hateful Memes Challenge paper) | VisualBERT, ViLBERT baselines | Pretrained multimodal transformer fusion | None | Moderate-high | Foundational reference only — establishes the dataset and task definition, not a model to reproduce |
| **HateCLIPer** (Kumar & Nandakumar, 2022) | Frozen CLIP | Feature-level fusion (concat + elementwise) | None | **Low** (frozen backbone) | **Direct basis for our Stage 2-4** — strongest accuracy-per-compute-dollar among CLIP-based methods, and its fusion idea is simple enough to implement correctly in a day |
| **PromptHate** (Cao et al., 2023) | Caption generator (ClipCap) + RoBERTa | Converts multimodal → unimodal via captioning, then prompts LM | Indirect (via prompt) | Moderate | Not adopted — captioning pipeline adds a full extra model + prompt-engineering loop we don't have time for |
| **Pro-Cap** (Cao et al., 2023) | Frozen BLIP-2, VQA-style captioning | Caption-based, similar to PromptHate | Indirect | Moderate-high | Not adopted — same reason as PromptHate, plus BLIP-2 is heavier to run than CLIP |
| **Hee et al. 2022** ("On Explaining Multimodal Hateful Meme Detection Models") | Multimodal transformer | Standard fusion | **Grad-CAM based** — direct inspiration for our Stage 5 | Moderate | **Basis for our explainability design** — proves Grad-CAM-style visual explanation is an established, defensible technique in this exact literature |
| **ExplainHM** (Lin et al., 2024) | LLM-based | N/A (text-converted) | LLM-generated contradictory rationales, judged by a tunable model | **High** (multiple LLM calls, orchestration) | Not adopted — powerful but requires an LLM-debate pipeline that's out of scope for a 1-2 day build |
| **Tzelepi et al. 2025** (CVPR Workshop, MULA) | LMM + CLIP embeddings | Concatenates CLIP + LMM-generated semantic/emotion embeddings, lightweight head | None (efficiency-focused) | Low-moderate | Noted as a possible future extension (adding an LMM-generated caption embedding as a 3rd fusion input) if time permits after core pipeline works |

---

## 3. Why This Methodology, In One Paragraph

We anchor on **HateCLIPer's fusion strategy** because it is the best accuracy-for-compute tradeoff in the entire surveyed literature — a frozen backbone means the whole project fits in a 1-2 day budget without needing a GPU cluster. We then graft on the **explainability approach from Hee et al. 2022** rather than the newer but far heavier LLM-rationale approach (ExplainHM), because Grad-CAM-style visual explanation is something we can implement and verify ourselves in an afternoon, while an LLM-debate pipeline would consume the entire time budget just on orchestration. The caption-based methods (PromptHate, Pro-Cap) were consciously excluded because they require a second full model (ClipCap or BLIP-2) just to produce an intermediate representation — extra engineering surface with no clear payoff for a project whose main goal is demonstrating fusion + explainability, not chasing state-of-the-art accuracy.
