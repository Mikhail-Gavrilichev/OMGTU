from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from models.ollama_client import OllamaExplainer
from models.embedding_encoder import EmbeddingModelType
import torch
import numpy as np

from config import settings
from models.neural_model import NeuralSearchModel
from enum import Enum

app = FastAPI(
    title="Neural Search API для литературных произведений",
    description="API для нейронного поиска и автоматического присвоения тегов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_models = {}
try:
    # BigBird модель
    search_models[EmbeddingModelType.bigbird] = NeuralSearchModel(settings)
    print(f"BigBird модель инициализирована успешно")

    # MPNet модель
    mpnet_settings = settings.copy(update={
        "EMBEDDING_MODEL": EmbeddingModelType.multi,
        "MAX_LENGTH": settings.MPNET_MAX_LENGTH,
        "MPNET_MODEL_NAME": settings.MPNET_MODEL_NAME
    })
    search_models[EmbeddingModelType.multi] = NeuralSearchModel(mpnet_settings)
    print(f"MPNet модель инициализирована успешно")

except Exception as e:
    print(f"Ошибка инициализации моделей: {e}")
    raise

try:
    llm_explainer = OllamaExplainer(
        model_name="deepseek-r1:8b"
    )
    print(f"LLM инициализирован успешно")
except Exception as e:
    print(f"Ошибка инициализации LLM объяснителя: {e}")
    llm_explainer = None


class LLMMode(str, Enum):
    off = "off"
    query_only = "query_only"
    full = "full"


class SearchRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    fandoms: Optional[str] = None
    characters: Optional[str] = None
    story: Optional[str] = None
    top_k: Optional[int] = 5
    embedding_model: EmbeddingModelType = EmbeddingModelType.bigbird

    llm_mode: LLMMode = Field(
        LLMMode.full,
        description="Режим использования LLM"
    )


class NewWorkRequest(BaseModel):
    work_id: str = Field(..., min_length=1, description="ID произведения")
    title: str = Field(..., description="Название произведения")
    summary: Optional[str] = Field(None, description="Краткое описание")
    tags: Optional[List[str]] = Field([], description="Теги")
    fandoms: Optional[List[str]] = Field([], description="Фэндомы")
    characters: Optional[List[str]] = Field([], description="Персонажи")
    categories: Optional[List[str]] = Field([], description="Категории")
    relationships: Optional[List[str]] = Field([], description="Отношения/пейринги")
    text_sample: Optional[str] = Field(None, description="Отрывок текста")


class DeveloperInfo(BaseModel):
    name: str
    group: str
    email: str
    role: str


@app.get("/")
async def root():
    return {
        "message": "Neural Search API для литературных произведений",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search (POST)",
            "add_work": "/add-work (POST)",
            "developers": "/developers (GET)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    health_info = {
        "status": "healthy",
        "models_available": list(search_models.keys()),
    }

    for model_type, model in search_models.items():
        try:
            model_info = {
                "embeddings_shape": model.embeddings.shape if hasattr(model, 'embeddings') else None,
                "dataset_size": len(model.dataset) if hasattr(model, 'dataset') else 0,
                "device": str(model.device),
                "model_type": str(model.embedding_model_type)
            }

            test_emb = model.get_query_embedding("test")
            model_info["encoder_working"] = True
            model_info["test_embedding_shape"] = test_emb.shape

        except Exception as e:
            model_info = {
                "error": str(e),
                "encoder_working": False
            }

        health_info[f"{model_type.value}_model"] = model_info

    if llm_explainer:
        health_info["llm_available"] = True
    else:
        health_info["llm_available"] = False
        health_info["llm_error"] = "LLM объяснитель не инициализирован"

    return health_info


@app.post("/search")
async def search_similar_works(request: SearchRequest):
    try:
        if request.embedding_model not in search_models:
            raise HTTPException(
                status_code=400,
                detail=f"Модель {request.embedding_model} не доступна"
            )

        search_model = search_models[request.embedding_model]

        query_text = search_model.prepare_query_text(
            request.title or "",
            request.summary or "",
            request.fandoms or "",
            request.characters or "",
            request.story or ""
        )

        if not query_text or query_text.strip() == "":
            return {
                "query": request.dict(),
                "results_count": 0,
                "results": [],
                "llm_explanation": None,
                "warning": "Пустой запрос"
            }

        results = search_model.search(query_text, top_k=request.top_k)

        llm_answer = None
        if llm_explainer and request.llm_mode != LLMMode.off:
            try:
                llm_answer = llm_explainer.explain(
                    query_text=query_text,
                    retrieved_works=results,
                    include_full_text=(request.llm_mode == LLMMode.full)
                )
            except Exception as e:
                print(f"Ошибка LLM объяснителя: {e}")
                llm_answer = f"Ошибка LLM: {str(e)}"

        return {
            "query": {
                "title": request.title,
                "summary": request.summary,
                "fandoms": request.fandoms,
                "characters": request.characters,
                "story": request.story,
                "embedding_model": request.embedding_model
            },
            "results_count": len(results),
            "results": results,
            "llm_explanation": llm_answer
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


@app.post("/add-work")
async def add_new_work(work: NewWorkRequest):
    try:
        if not work.work_id or not work.work_id.strip():
            raise HTTPException(status_code=400, detail="work_id обязателен")

        for model_type, model in search_models.items():
            if str(work.work_id) in model.dataset["work_id"].astype(str).values:
                raise HTTPException(
                    status_code=409,
                    detail=f"Работа с таким work_id уже существует в модели {model_type.value}"
                )

        work_data = {
            "work_id": work.work_id,
            "title": work.title,
            "summary": work.summary or "",
            "tags": work.tags or [],
            "fandoms": work.fandoms or [],
            "characters": work.characters or [],
            "categories": work.categories or [],
            "relationships": work.relationships or [],
            "text_sample": work.text_sample or ""
        }

        results = {}
        for model_type, model in search_models.items():
            try:
                success, result = model.add_new_work(work_data)
                if not success:
                    results[model_type.value] = {
                        "success": False,
                        "error": result
                    }
                else:
                    results[model_type.value] = {
                        "success": True,
                        "work_id": result
                    }
            except Exception as e:
                results[model_type.value] = {
                    "success": False,
                    "error": str(e)
                }

        all_success = all(r["success"] for r in results.values())
        if not all_success:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Ошибка при добавлении работы в некоторые модели",
                    "results": results
                }
            )

        return {
            "success": True,
            "message": "Работа успешно добавлена и эмбеддинги обновлены для всех моделей",
            "work_id": work.work_id,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка добавления работы: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/dataset-info")
async def get_dataset_info():
    info = {}
    for model_type, model in search_models.items():
        try:
            info[model_type.value] = {
                "total_works": len(model.dataset),
                "total_tags": len(model.mlb.classes_) if hasattr(model, 'mlb') else 0,
                "embeddings_dimension": model.embeddings.shape[1] if hasattr(model, 'embeddings') else 0,
                "total_embeddings": model.embeddings.shape[0] if hasattr(model, 'embeddings') else 0,
                "model_type": str(model.embedding_model_type)
            }
        except Exception as e:
            info[model_type.value] = {
                "error": str(e)
            }
    return info


if __name__ == "__main__":
    print(f"Запуск API на {settings.HOST}:{settings.PORT}")
    print(f"Доступные модели: {list(search_models.keys())}")
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )