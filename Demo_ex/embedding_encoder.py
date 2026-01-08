import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Optional
from enum import Enum


class EmbeddingModelType(str, Enum):
    bigbird = "bigbird"
    multi = "multi"


class EmbeddingEncoder:
    def __init__(
            self,
            model_type: EmbeddingModelType,
            device: torch.device,
            bigbird_model: Optional = None,
            bigbird_tokenizer: Optional = None,
            multi_model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1",
            bigbird_max_length: int = 4096,
            multi_max_length: int = 512
    ):
        self.model_type = model_type
        self.device = device

        if model_type == EmbeddingModelType.bigbird:
            if bigbird_model is None or bigbird_tokenizer is None:
                raise ValueError("Для BigBird модели необходимо передать model и tokenizer")
            self.model = bigbird_model
            self.tokenizer = bigbird_tokenizer
            self.max_length = bigbird_max_length

        elif model_type == EmbeddingModelType.multi:
            try:
                print(f"Загрузка MPNet модели: {multi_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(multi_model_name)
                self.model = AutoModel.from_pretrained(multi_model_name)
                self.model.to(device).eval()
                self.max_length = multi_max_length
                print(f"MPNet модель успешно загружена на устройство: {device}")
            except Exception as e:
                print(f"Ошибка загрузки MPNet модели: {e}")
                raise
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        try:
            if not text or not isinstance(text, str):
                text = ""

            if self.model_type == EmbeddingModelType.multi:
                words = text.split()
                if len(words) > 200:
                    text = " ".join(words[:200])

            enc = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            if self.model_type == EmbeddingModelType.bigbird:
                emb = self.model(
                    enc["input_ids"],
                    enc["attention_mask"],
                    return_embeddings=True
                )
            else:
                model_output = self.model(**enc)

                token_embeddings = model_output.last_hidden_state
                attention_mask = enc["attention_mask"]

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()

                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                emb = sum_embeddings / sum_mask

            if emb.dim() > 1:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            else:
                emb = torch.nn.functional.normalize(emb.unsqueeze(0), p=2, dim=1)

            emb_np = emb.cpu().numpy()

            if emb_np.ndim == 2:
                emb_np = emb_np[0]

            return emb_np

        except Exception as e:
            print(f"Ошибка в encode() для модели {self.model_type}: {e}")
            print(f"Текст: {text[:100]}...")
            raise

    def batch_encode(self, texts: list) -> np.ndarray:
        embeddings = []
        for text in texts:
            emb = self.encode(text)
            embeddings.append(emb)
        return np.array(embeddings)