# 🏥 Medical Image Visual Question Answering with Multimodal RAG

A retrieval-augmented generation (RAG) pipeline for medical image Visual Question Answering (VQA), combining **BiomedCLIP** embeddings, a **FAISS** vector index, and **LLaVA-1.5-7B** as the generative model — all grounded on the **SLAKE** medical VQA dataset.

---

## 📌 Overview

This project implements an end-to-end multimodal RAG system designed to answer clinical questions about medical images (X-rays, MRIs, CT scans). Rather than relying on a generative model's parametric knowledge alone, the system retrieves relevant question-answer pairs and structured medical knowledge from the SLAKE dataset at inference time, significantly reducing hallucination.

**Pipeline summary:**

```
Query (text + image)
        │
        ▼
BiomedCLIP Encoder (fused image + text embedding)
        │
        ▼
FAISS Index Search (top-K nearest neighbors from SLAKE)
        │
        ▼
Grounded Prompt Builder (anti-hallucination context injection)
        │
        ▼
LLaVA-1.5-7B Generator (4-bit quantized, multimodal)
        │
        ▼
Answer
```

---

## 🗂️ Dataset

**SLAKE** (Structured Language and Knowledge Enabled VQA for Medical Images) — available on Hugging Face at [`BoKelvin/SLAKE`](https://huggingface.co/datasets/BoKelvin/SLAKE).

- Covers radiology images across multiple modalities (X-ray, MRI, CT)
- Includes structured knowledge base (`base`) per sample with medical facts
- Question types: **CLOSED** (Yes/No) and **OPEN** (factual, descriptive)
- Annotated with organ, body part, content type (Abnormality, Organ, etc.)

---

## 🧠 Models

| Component | Model |
|---|---|
| Image + Text Encoder | [`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| Generative VQA Model | [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| Vector Index | FAISS `IndexFlatIP` (cosine similarity) |
| Evaluation Metric | Exact-match accuracy (closed) + BERTScore F1 (open) |

**BiomedCLIP** is pretrained on 15M biomedical image-caption pairs from PubMed Central, making it far more suited for medical retrieval than generic CLIP. **LLaVA-1.5-7B** is loaded in 4-bit NF4 quantization via `bitsandbytes` to fit on a 16GB GPU (e.g., NVIDIA T4).

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.9+
- CUDA-enabled GPU (recommended: 16GB+ VRAM, e.g., T4, A100)
- Google Colab (recommended environment) or equivalent

### Install Dependencies

```bash
pip install transformers accelerate datasets faiss-cpu \
    langchain langchain-community sentence-transformers \
    Pillow requests tqdm torch torchvision
pip install bitsandbytes
pip install bert-score
pip install huggingface_hub
pip install open_clip_torch
pip install ipywidgets
```

### Hugging Face Authentication (optional but recommended)

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

Setting a token avoids rate limits when downloading models and datasets.

---

## 🚀 Usage

Open and run `capstone-b-ipynb.ipynb` cell by cell. The notebook is organized into the following sections:

1. **Dependencies installation** — installs all required packages
2. **Imports and configuration** — sets device, model names, index paths
3. **Dataset** — loads SLAKE from Hugging Face and downloads images
4. **BiomedCLIP encoder** — loads the biomedical CLIP model and defines `embed_image` / `embed_text` helpers
5. **Build FAISS index** — indexes the SLAKE training set with fused image+text vectors (built once, cached to disk)
6. **Retriever** — retrieves top-K nearest contexts given a query image and/or text
7. **Context builder** — constructs a grounded, anti-hallucination prompt from retrieved results
8. **Load LLaVA-1.5-7B** — loads the generative model in 4-bit quantization
9. **Full RAG inference pipeline** — `rag_answer()` ties all steps together
10. **LangChain wrapper** — wraps the pipeline as a `RunnableLambda` chain
11. **Demo queries** — runs closed-ended, open-ended, and descriptive test questions
12. **Evaluation** — computes closed-QA accuracy and open-QA BERTScore on the SLAKE eval set
13. **Interactive dashboard** — ipywidgets UI for exploring queries and images

### Example Inference

```python
from PIL import Image

image = Image.open("path/to/chest_xray.jpg")
answer = rag_answer(
    query_text="Is there any abnormality in this chest X-ray?",
    query_image=image,
    top_k=5,
    verbose=True
)
print(answer)
```

### Using the LangChain Chain

```python
result = rag_chain.invoke({
    "text": "What organ is shown in this medical image?",
    "image": image   # PIL Image or None
})
print(result)
```

---

## 📊 Evaluation

The `evaluate_slake()` function benchmarks the pipeline on a held-out portion of the SLAKE validation set:

| Question Type | Metric | Description |
|---|---|---|
| CLOSED (Yes/No) | Exact-match accuracy | Flexible token overlap check |
| OPEN (Factual) | BERTScore F1 | Semantic similarity to ground-truth answer |

```python
closed_acc, open_f1 = evaluate_slake(n_closed=30, n_open=20)
```

---

## 🗺️ Project Structure

```
.
├── capstone-b-ipynb.ipynb   # Main notebook (full pipeline)
├── slake_index.faiss        # FAISS index (generated at runtime)
├── slake_meta.json          # Index metadata (generated at runtime)
└── README.md
```

---

## 🔧 Configuration

Key constants defined in Cell 2 of the notebook:

| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL` | `microsoft/BiomedCLIP-...` | Encoder model ID |
| `GEN_MODEL` | `llava-hf/llava-1.5-7b-hf` | Generator model ID |
| `TOP_K` | `5` | Number of retrieved contexts |
| `INDEX_PATH` | `/content/slake_index.faiss` | FAISS index save path |
| `META_PATH` | `/content/slake_meta.json` | Metadata save path |

The retrieval fusion weight (`alpha`) can also be tuned in both `build_slake_index` and `retrieve`. A value of `alpha=0.6` weights image embeddings slightly more heavily, which suits radiology tasks.

---

## 📋 Notes & Limitations

- The FAISS index is built once from the SLAKE training split and cached to disk. Subsequent runs load the cached index automatically.
- LLaVA-1.5-7B requires an image input at all times; text-only queries use a blank white image as a placeholder.
- SLAKE images must be downloaded separately from the HF dataset repo (`imgs.zip`) — the dataset loader handles this automatically.
- Dependency version conflicts with `google-colab` base packages are non-critical and do not affect functionality.

---

## 📄 License

This project is intended for research and educational purposes. Please refer to the individual licenses of the SLAKE dataset, BiomedCLIP, and LLaVA-1.5 before any commercial use.