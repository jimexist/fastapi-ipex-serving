import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal

import intel_extension_for_pytorch as ipex
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    pipeline,
)


class ServerSetting(BaseSettings):
    model_name_or_path: str | None
    mp_context: Literal["spawn", "fork"] = "spawn"
    dtype: Literal["bfloat16", "fp16", "float32"] = "bfloat16"


class PredictRequest(BaseModel):
    inputs: str | list[str]
    parameters: dict | None


server_settings = ServerSetting()

global_ctx = {}


def handle_request(request: PredictRequest, truncation: bool = True):
    global classification_pipe
    with torch.no_grad(), torch.cpu.amp.autocast():
        return classification_pipe(request.inputs, truncation=truncation)


def init_model(settings: ServerSetting):
    model_name = settings.model_name_or_path or "/repository"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )
    model = model.eval()

    if settings.dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif settings.dtype == "fp16":
        model_dtype = torch.float16
    elif settings.dtype == "float32":
        model_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {settings.dtype}")
    model = ipex.optimize(model, dtype=model_dtype)
    logging.info(
        f"Model '{model_name}' loaded and optimized with {settings.dtype} dtype"
    )

    global classification_pipe
    classification_pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer
    )


def create_process_pool(settings: ServerSetting) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=settings.max_workers,
        mp_context=multiprocessing.get_context(settings.mp_context),
        initializer=init_model,
        initargs=(settings,),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global_ctx["process_pool"] = create_process_pool(settings=server_settings)
    yield
    process_pool: ProcessPoolExecutor = global_ctx["process_pool"]
    process_pool.shutdown()
    del global_ctx["process_pool"]


app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/predict")
async def predict(req: PredictRequest):
    process_pool: ProcessPoolExecutor = global_ctx["process_pool"]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(process_pool, handle_request, req)
