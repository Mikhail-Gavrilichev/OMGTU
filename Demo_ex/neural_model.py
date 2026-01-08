import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BigBirdTokenizer, BigBirdModel
from data.blocked_tags import BLOCKED_TAGS
from models.embedding_encoder import EmbeddingEncoder, EmbeddingModelType
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


class BigBirdForTags(nn.Module):
    def __init__(self, num_classes, model_name):
        super().__init__()
        self.body = BigBirdModel.from_pretrained(model_name)
        hidden = self.body.config.hidden_size
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def encode(self, input_ids, attention_mask):
        out = self.body(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.norm(out.last_hidden_state[:, 0])
        return cls

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        cls = self.encode(input_ids, attention_mask)
        if return_embeddings:
            return cls
        return self.classifier(cls)


class NeuralSearchModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")

        if config.EMBEDDING_MODEL == EmbeddingModelType.bigbird:
            max_length = getattr(config, "BIGBIRD_MAX_LENGTH", 4096)
        else:
            max_length = getattr(config, "MPNET_MAX_LENGTH", 512)

        self.tokenizer = BigBirdTokenizer.from_pretrained(config.TOKENIZER_PATH)

        with open(config.MLB_PATH, "rb") as f:
            self.mlb = pickle.load(f)

        self.model = BigBirdForTags(len(self.mlb.classes_), config.MODEL_PATH)
        self.model.load_state_dict(
            torch.load(config.MODEL_CHECKPOINT, map_location=self.device)
        )
        self.model.to(self.device).eval()

        self.embedding_model_type = config.EMBEDDING_MODEL

        if self.embedding_model_type == EmbeddingModelType.bigbird:
            print("Загрузка BigBird эмбеддингов...")
            try:
                data = torch.load(config.EMBEDDINGS_BIGBIRD_PATH, map_location="cpu")
                if "train_embeddings" in data:
                    emb = data["train_embeddings"]
                elif "embeddings" in data:
                    emb = data["embeddings"]
                else:
                    raise KeyError("Не найден ключ с эмбеддингами")
            except Exception as e:
                print(f"Ошибка загрузки BigBird эмбеддингов: {e}")
                raise

        elif self.embedding_model_type == EmbeddingModelType.multi:
            print("Загрузка MPNet эмбеддингов...")
            try:
                data = torch.load(config.EMBEDDINGS_MULTI_PATH, map_location="cpu")
                if "embeddings" in data:
                    emb = data["embeddings"]
                elif "train_embeddings" in data:
                    emb = data["train_embeddings"]
                else:
                    print(f"Создаём пустые эмбеддинги для MPNet")
                    emb = torch.zeros((0, 768))
            except Exception as e:
                print(f"Ошибка загрузки MPNet эмбеддингов: {e}")
                print(f"Создаём пустые эмбеддинги")
                emb = torch.zeros((0, 768))

        self.embeddings = emb.numpy() if isinstance(emb, torch.Tensor) else emb
        print(f"Загружено эмбеддингов: {self.embeddings.shape}")

        self.dataset = pd.read_json(config.DATASET_PATH)
        self.dataset["tags"] = self.dataset["tags"].apply(self._normalize_tags)
        print(f"Загружено произведений: {len(self.dataset)}")

        self.encoder = EmbeddingEncoder(
            model_type=self.embedding_model_type,
            device=self.device,
            bigbird_model=self.model,
            bigbird_tokenizer=self.tokenizer,
            multi_model_name=getattr(config, "MPNET_MODEL_NAME",
                                     "sentence-transformers/multi-qa-mpnet-base-cos-v1"),
            bigbird_max_length=getattr(config, "BIGBIRD_MAX_LENGTH", 4096),
            multi_max_length=getattr(config, "MPNET_MAX_LENGTH", 512)
        )

        self._reranker_initialized = False

    def _normalize_tags(self, tags):
        if tags is None or (isinstance(tags, float) and np.isnan(tags)):
            return []
        if isinstance(tags, str):
            tags = tags.split(",")
        return [str(t).lower().strip() for t in tags if str(t).strip()]

    def prepare_query_text(self, title, summary, fandoms, characters, story):
        parts = []
        if title:
            parts.append(f"[TITLE] {title}")
        if summary:
            parts.append(f"[SUMMARY] {summary}")
        if fandoms:
            parts.append(f"[FANDOMS] {fandoms}")
        if characters:
            parts.append(f"[CHARACTERS] {characters}")
        if story:
            parts.append(f"[STORY] {story}")

        if not parts:
            return ""

        return "\n".join(parts)

    def get_query_embedding(self, text: str) -> np.ndarray:
        try:
            if not text or text.strip() == "":
                if self.embeddings.shape[0] > 0:
                    return np.zeros((1, self.embeddings.shape[1]))
                else:
                    if self.embedding_model_type == EmbeddingModelType.bigbird:
                        return np.zeros((1, 768))
                    else:
                        return np.zeros((1, 768))

            emb = self.encoder.encode(text)

            if len(emb.shape) == 1:
                emb = emb.reshape(1, -1)

            return emb
        except Exception as e:
            print(f"Ошибка при получении эмбеддинга: {e}")
            return np.zeros((1, self.embeddings.shape[1] if self.embeddings.shape[0] > 0 else 768))

    def predict_tags(self, text):
        if not text or text.strip() == "":
            return []

        try:
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=getattr(self.config, "BIGBIRD_MAX_LENGTH", 4096),
                return_tensors="pt"
            )

            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device)
                )
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            idx = np.where(probs > self.config.TAG_THRESHOLD)[0]
            return [self.mlb.classes_[i].lower().strip() for i in idx]
        except Exception as e:
            print(f"Ошибка при предсказании тегов: {e}")
            return []

    def search(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = getattr(self.config, "TOP_K", 5)

        self._init_reranker_once()

        query_emb = self.get_query_embedding(query_text)

        if self.embeddings.shape[0] == 0:
            print("В базе нет эмбеддингов")
            return []

        if query_emb.shape[1] != self.embeddings.shape[1]:
            print(f"Несоответствие размерностей: запрос {query_emb.shape[1]}, база {self.embeddings.shape[1]}")
            return []

        similarities = cosine_similarity(query_emb, self.embeddings)[0]

        query_tags = set(self.predict_tags(query_text))
        max_tag_matches = max(len(query_tags), 1)

        results = []
        for idx, sim in enumerate(similarities):
            if idx >= len(self.dataset):
                continue

            row = self.dataset.iloc[idx]
            tags = row.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            work_tags = set(tags)

            rating = str(row.get("rating", "")).lower().strip()
            categories = row.get("categories", [])
            categories = [str(c).lower().strip() for c in categories] if isinstance(categories, list) else []

            is_blocked = (
                    rating in {"explicit", "mature"} or
                    any(cat in {"m/m", "f/f"} for cat in categories) or
                    any(tag in BLOCKED_TAGS for tag in tags)
            )

            if is_blocked:
                continue

            tag_matches = len(query_tags.intersection(work_tags))
            normalized_cosine = (sim + 1) / 2
            normalized_tags = tag_matches / max_tag_matches
            score = 0.9 * normalized_cosine + 0.1 * normalized_tags

            results.append({
                'index': int(idx),
                'score': float(score),
                'cosine_sim': float(sim),
                'tag_matches': int(tag_matches),
                'row': row
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        rerank_candidates = results[:min(
            getattr(self.config, "RERANKER_CANDIDATES", 40),
            len(results)
        )]

        if rerank_candidates and hasattr(self, 'reranker_model'):
            rerank_candidates = self.apply_reranker(query_text, rerank_candidates)

        search_results = []
        for res in rerank_candidates[:top_k]:
            row = res['row']

            search_results.append({
                'title': str(row.get('title', 'Без названия')),
                'summary': str(row.get('summary', '')),
                'tags': list(row.get('tags', [])),
                'predicted_tags': list(query_tags),
                'fandoms': list(row.get('fandoms', [])),
                'characters': list(row.get('characters', [])),
                'categories': list(row.get('categories', [])),
                'relationships': list(row.get('relationships', [])),
                'work_id': str(row.get('work_id', '')),
                'similarity_score': float(res['score']),
                'cosine_similarity': float(res['cosine_sim']),
                'tag_overlap': int(res['tag_matches'])
            })

        return to_python(search_results)

    def _init_reranker_once(self):
        if not self._reranker_initialized:
            try:
                print("Загрузка реранкера...")
                self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                    "models/bge-reranker-large"
                )
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    "models/bge-reranker-large"
                ).to(self.device).eval()
                self._reranker_initialized = True
                print("Реранкер успешно загружен")
            except Exception as e:
                print(f"Не удалось загрузить реранкер: {e}")
                self._reranker_initialized = False

    def apply_reranker(self, query_text, candidates):
        try:
            pairs = []
            indices = []

            for cand in candidates:
                text = cand['row'].get('text_sample', '')
                if not text:
                    text = cand['row'].get('summary', '')
                if text:
                    pairs.append((query_text, text))
                    indices.append(cand['index'])

            if not pairs:
                return candidates

            enc = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.reranker_model(**enc)
                scores = torch.sigmoid(outputs.logits).cpu().numpy()

            if scores.ndim == 2:
                if scores.shape[1] == 1:
                    scores = scores[:, 0]
                else:
                    scores = scores[:, 1]

            scores = np.squeeze(scores)

            score_map = {idx: score for idx, score in zip(indices, scores)}
            for cand in candidates:
                idx = cand['index']
                if idx in score_map:
                    rerank_score = float(score_map[idx])
                    cand['score'] = cand['score'] * 0.8 + rerank_score * 0.2

            candidates.sort(key=lambda x: x['score'], reverse=True)

        except Exception as e:
            print(f"Ошибка при реранкинге: {e}")

        return candidates

    def add_new_work(self, work_data: Dict[str, Any]) -> Tuple[bool, str]:
        if not work_data.get("work_id"):
            return False, "work_id обязателен"

        work_data["full_text_input"] = self.prepare_query_text(
            work_data.get("title", ""),
            work_data.get("summary", ""),
            ",".join(work_data.get("fandoms", [])) if work_data.get("fandoms") else "",
            ",".join(work_data.get("characters", [])) if work_data.get("characters") else "",
            work_data.get("text_sample", "")
        )

        work_data["tags"] = self._normalize_tags(work_data.get("tags", []))

        try:
            new_row = pd.DataFrame([work_data])
            self.dataset = pd.concat([self.dataset, new_row], ignore_index=True)

            self.dataset.to_json(
                self.config.DATASET_PATH,
                orient="records",
                force_ascii=False,
                indent=2
            )

            emb = self.get_query_embedding(work_data["full_text_input"])

            if self.embeddings.shape[0] == 0:
                self.embeddings = emb
            else:
                self.embeddings = np.vstack([self.embeddings, emb])

            if self.embedding_model_type == EmbeddingModelType.bigbird:
                save_path = self.config.EMBEDDINGS_BIGBIRD_PATH
            else:
                save_path = self.config.EMBEDDINGS_MULTI_PATH

            torch.save({
                "embeddings": torch.tensor(self.embeddings),
                "model_type": str(self.embedding_model_type)
            }, save_path)

            return True, work_data["work_id"]

        except Exception as e:
            print(f"Ошибка при добавлении работы: {e}")
            return False, str(e)