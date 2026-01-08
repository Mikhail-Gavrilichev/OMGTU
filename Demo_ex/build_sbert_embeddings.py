import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

# =====================
# НАСТРОЙКИ
# =====================
DATASET_PATH = "dataset.json"
OUTPUT_PATH = "embeddings_multi.pt"
multi_PATH = "../models/multi-qa-mpnet-base-cos-v1"

MAX_LENGTH = 512   # SBERT больше не нужно
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# ЗАГРУЗКА МОДЕЛИ
# =====================
print("Loading SBERT...")
tokenizer = AutoTokenizer.from_pretrained(multi_PATH)
model = AutoModel.from_pretrained(multi_PATH).to(DEVICE).eval()

# =====================
# ЗАГРУЗКА ДАТАСЕТА
# =====================
print("Loading dataset...")
df = pd.read_json(DATASET_PATH)

def prepare_text(row):
    parts = []

    if row.get("title"):
        parts.append(f"[TITLE] {row['title']}")
    if row.get("summary"):
        parts.append(f"[SUMMARY] {row['summary']}")
    if row.get("fandoms"):
        parts.append(f"[FANDOMS] {', '.join(row['fandoms']) if isinstance(row['fandoms'], list) else row['fandoms']}")
    if row.get("characters"):
        parts.append(f"[CHARACTERS] {', '.join(row['characters']) if isinstance(row['characters'], list) else row['characters']}")
    if row.get("text_sample"):
        parts.append(f"[STORY] {row['text_sample']}")

    return "\n".join(parts)

texts = df.apply(prepare_text, axis=1).tolist()

# =====================
# SBERT MEAN POOLING
# =====================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    mask = attention_mask.unsqueeze(-1)
    return (token_embeddings * mask).sum(1) / mask.sum(1)

# =====================
# ПОДСЧЁТ ЭМБЕДДИНГОВ
# =====================
all_embeddings = []

print("Computing SBERT embeddings...")
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = model(**enc)
        emb = mean_pooling(outputs.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0)

# =====================
# СОХРАНЕНИЕ
# =====================
torch.save(
    {
        "model": "ai-forever_sbert_large_nlu_ru",
        "embeddings": embeddings,
        "dim": embeddings.shape[1],
        "count": embeddings.shape[0]
    },
    OUTPUT_PATH
)

print("===================================")
print(f"Saved SBERT embeddings to {OUTPUT_PATH}")
print(f"Shape: {embeddings.shape}")
print("===================================")
