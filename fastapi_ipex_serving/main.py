import asyncio
import logging
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal

import intel_extension_for_pytorch as ipex
import torch
from aiolimiter import AsyncLimiter
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
    cpu_max_workers: int = 8
    rate_limit_per_second: int = 64
    mp_context: Literal["spawn", "fork"] = "spawn"
    dtype: Literal["bfloat16", "fp16", "float32"] = "bfloat16"

    class Config:
        env_prefix = "SERVER_"


class PredictRequest(BaseModel):
    inputs: str | list[str]
    parameters: dict | None


server_settings = ServerSetting()

global_ctx = {}


def handle_request(request: PredictRequest, truncation: bool = True):
    start = time.perf_counter()
    global classification_pipe
    with torch.no_grad(), torch.cpu.amp.autocast():
        result = classification_pipe(request.inputs, truncation=truncation)
    elapsed = time.perf_counter() - start
    return result, elapsed


def init_model(settings: ServerSetting):
    model_name = "/repository"
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

    global classification_pipe
    classification_pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer
    )


def create_process_pool(settings: ServerSetting) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=settings.cpu_max_workers,
        mp_context=multiprocessing.get_context(settings.mp_context),
        initializer=init_model,
        initargs=(settings,),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global_ctx["process_pool"] = create_process_pool(settings=server_settings)
    global_ctx["rate_limit"] = AsyncLimiter(
        10 * server_settings.rate_limit_per_second, 10
    )
    yield
    process_pool: ProcessPoolExecutor = global_ctx["process_pool"]
    process_pool.shutdown()
    del global_ctx["process_pool"]


app = FastAPI(lifespan=lifespan)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(
    "%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s"
)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/info")
async def info():
    return {
        "cpu_max_workers": server_settings.cpu_max_workers,
        "rate_limit_per_second": server_settings.rate_limit_per_second,
        "mp_context": server_settings.mp_context,
        "dtype": server_settings.dtype,
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    rate_limit: AsyncLimiter = global_ctx["rate_limit"]
    process_pool: ProcessPoolExecutor = global_ctx["process_pool"]
    async with rate_limit:
        loop = asyncio.get_running_loop()
        result, elapsed = await loop.run_in_executor(process_pool, handle_request, req)
        logger.info(f"Model inference time: {elapsed:.3f}s")
        return result
