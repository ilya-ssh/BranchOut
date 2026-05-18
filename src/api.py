from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import contextlib
import inspect
import json
import os
import uuid
from dotenv import load_dotenv
from typing import Dict, Optional, Any
import uvicorn

from src.pydantic_schemas import VNGenerationRequest, VNGenerationResponse
from src.orchestrator import VNOrchestrator
from src.orchestrator_simple import VNOrchestratorSimple
from src.router import ModelRouter
from src.apicallhandler import OpenRouterClient
from src.logger import setup_logging, app_logger
from src.progress import progress_manager

client: Optional[OpenRouterClient] = None
orchestrator: Optional[object] = None
generation_tasks: Dict[str, asyncio.Task] = {}


def _pick_orchestrator(client: OpenRouterClient, router: ModelRouter) -> object:
    variant = str(os.getenv("PIPELINE_VARIANT", "full")).strip().lower()
    if variant in ("worse", "simple", "baseline"):
        return VNOrchestratorSimple(client, router)
    return VNOrchestrator(client, router)


def _request_to_generate_kwargs(request: VNGenerationRequest) -> Dict[str, Any]:
    return {
        "user_prompt": request.user_prompt,
        "story_length": request.story_length,
        "char_list": request.char_list,
        "loc_list": request.loc_list,
        "setting": request.setting_override,
        "max_branches": request.max_branches,
        "tone": request.tone,
        "artstyle": request.artstyle,
        "generate_images": request.generate_images if request.generate_images is not None else True,
        "time_choice": request.time_choice,
        "genre_choice": request.genre_choice,
        "tone_choice_ru": request.tone_choice,
        "mc_name": request.mc_name,
        "mc_description": request.mc_description,
        "extra_character_names": request.extra_character_names,
        "plot_prefs": request.plot_prefs,
        "plot_freeform": request.plot_freeform,
        "graphic_style_ru": request.graphic_style,
    }


def _supports_progress_callbacks(obj: object) -> bool:
    try:
        sig = inspect.signature(obj.generate_vn)
    except Exception:
        return False
    params = sig.parameters
    return "generation_id" in params and "progress_callback" in params


def _sse_pack(data: Dict[str, Any], event_name: str = "progress") -> str:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _run_generation_task(generation_id: str, request: VNGenerationRequest) -> None:
    global orchestrator, client

    if orchestrator is None or client is None:
        await progress_manager.fail(
            generation_id,
            error="Service not initialized",
            message="Service not initialized",
        )
        return

    try:
        await progress_manager.update(
            generation_id,
            status="running",
            stage="started",
            message="Generation task started",
            percent=1,
        )

        kwargs = _request_to_generate_kwargs(request)

        if _supports_progress_callbacks(orchestrator):
            kwargs["generation_id"] = generation_id
            kwargs["progress_callback"] = progress_manager.from_orchestrator
        else:
            #фоллбэк
            await progress_manager.update(
                generation_id,
                status="running",
                stage="running",
                message="Pipeline started (limited progress mode)",
                percent=5,
            )

        result = await orchestrator.generate_vn(**kwargs)

        await progress_manager.complete(
            generation_id,
            result=result,
            message="Visual novel generated successfully",
        )

    except Exception as e:
        app_logger.error(f"Background generation failed for {generation_id}: {e}", exc_info=True)
        await progress_manager.fail(
            generation_id,
            error=str(e),
            message="Generation failed",
        )
    finally:
        generation_tasks.pop(generation_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, orchestrator

    load_dotenv()
    setup_logging()
    app_logger.info("BranchOut запускается")

    api_key = os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    if "openrouter.ai" in base_url and not api_key:
        app_logger.error("API_KEY не найден для OpenRouter")
        raise ValueError("API_KEY не найден в .env")
    client = OpenRouterClient(api_key=api_key, base_url=base_url)
    router = ModelRouter()
    orchestrator = _pick_orchestrator(client, router)

    yield

    app_logger.info("Выключаемся...")

    for task in list(generation_tasks.values()):
        task.cancel()

    for task in list(generation_tasks.values()):
        with contextlib.suppress(asyncio.CancelledError):
            await task

    if client:
        await client.close()
        app_logger.info("Клиент выключился")


app = FastAPI(
    title="BranchOut",
    description="Мультиагентная система генерации визуальных новелл",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "online", "service": "VN Generator API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "client_initialized": client is not None,
        "orchestrator_initialized": orchestrator is not None,
        "total_api_calls": client.call_count if client else 0,
        "total_tokens_used": client.total_tokens_used if client else 0,
        "pipeline_variant": str(os.getenv("PIPELINE_VARIANT", "full")).strip().lower(),
        "active_background_generations": len(generation_tasks),
    }


#легаси
@app.post("/generate", response_model=VNGenerationResponse)
async def generate_vn(request: VNGenerationRequest):
    try:
        if orchestrator is None or client is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        app_logger.info("Блокирующий запрос генерации")
        app_logger.info(
            f"Форма: time={request.time_choice}, genre={request.genre_choice}, "
            f"tone_ru={request.tone_choice}, graphic_style={request.graphic_style}"
        )
        app_logger.info(
            f"MC: {request.mc_name}, has_mc_desc={bool(request.mc_description)}, "
            f"extra_chars={len(request.extra_character_names or [])}"
        )
        app_logger.info(f"Plot prefs?={bool(request.plot_prefs)}, plot_freeform?={bool(request.plot_freeform)}")
        app_logger.info(f"Промпт юзера: {request.user_prompt[:200]}...")
        app_logger.info(f"Длина истории (raw): {request.story_length}")
        app_logger.info(f"Max branches: {request.max_branches}")
        app_logger.info(f"Generate images: {request.generate_images}")
        app_logger.info(f"PIPELINE_VARIANT: {str(os.getenv('PIPELINE_VARIANT','full')).strip().lower()}")
        app_logger.info("=" * 60)

        if not request.mc_name:
            raise HTTPException(status_code=400, detail="Поле 'mc_name' (Имя ГГ) обязательно")

        result = await orchestrator.generate_vn(**_request_to_generate_kwargs(request))  # type: ignore[attr-defined]

        app_logger.info("=" * 60)
        app_logger.info(f"Генерация завершена: {result.get('generation_id')}")
        app_logger.info(f"Вызовов API: {client.call_count}")
        app_logger.info(f"Токенов использовано: {client.total_tokens_used}")
        app_logger.info("=" * 60)

        return VNGenerationResponse(
            status="success",
            message="Visual novel generated successfully",
            generation_id=result.get("generation_id", "unknown"),
            context=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Ошибка генерации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


#асинковый для веба
@app.post("/generate/start")
async def start_generate_vn(request: VNGenerationRequest):
    try:
        if orchestrator is None or client is None:
            raise HTTPException(status_code=503, detail="Service not initialized")
        if not request.mc_name:
            raise HTTPException(status_code=400, detail="Поле 'mc_name' (Имя ГГ) обязательно")
        generation_id = f"job_{uuid.uuid4().hex}"
        await progress_manager.create(
            generation_id,
            message="Request accepted",
            extra={
                "pipeline_variant": str(os.getenv("PIPELINE_VARIANT", "full")).strip().lower(),
                "mc_name": request.mc_name,
                "story_length": request.story_length,
            },
        )
        task = asyncio.create_task(_run_generation_task(generation_id, request))
        generation_tasks[generation_id] = task
        return {
            "status": "started",
            "message": "Generation started",
            "generation_id": generation_id,
            "progress_url": f"/generate/progress/{generation_id}",
            "events_url": f"/generate/events/{generation_id}",
            "result_url": f"/generate/result/{generation_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Ошибка запуска фоновой генерации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка запуска генерации: {str(e)}")


@app.get("/generate/progress/{generation_id}")
async def get_generation_progress(generation_id: str):
    state = await progress_manager.get(generation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Generation ID not found")
    return state


@app.get("/generate/result/{generation_id}")
async def get_generation_result(generation_id: str):
    state = await progress_manager.get(generation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Generation ID not found")

    if state["status"] != "completed":
        return {
            "status": state["status"],
            "message": "Result is not ready yet",
            "generation_id": generation_id,
        }

    result = await progress_manager.get_result(generation_id)
    return {
        "status": "success",
        "generation_id": generation_id,
        "context": result,
    }


@app.get("/generate/events/{generation_id}")
async def stream_generation_events(generation_id: str):
    state = await progress_manager.get(generation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Generation ID not found")

    async def event_stream():
        queue = await progress_manager.subscribe(generation_id)
        try:
            initial = await progress_manager.get(generation_id)
            if initial is not None:
                yield _sse_pack(initial, event_name="progress")

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _sse_pack(event, event_name="progress")

                    if event.get("status") in {"completed", "failed"}:
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            await progress_manager.unsubscribe(generation_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate/test/{agent_type}")
async def test_single_agent(agent_type: str, params: Dict):
    try:
        app_logger.info(f"Тестируем одного агента: {agent_type}")
        app_logger.info(f"Параметры: {params}")

        return {
            "status": "not_implemented",
            "message": "В работе",
            "agent_type": agent_type,
            "params": params,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        app_logger.error(f"Ошибка {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_available_models():
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    models = orchestrator.router.get_all_models()  # type: ignore[attr-defined]
    return {
        "models": {
            key: {
                "name": config.name,
                "display_name": config.display_name,
                "context_window": config.context_window,
                "cost_per_1k_tokens": config.cost_per_1k_tokens,
                "best_for": config.best_for,
                "supports_function_calling": config.supports_function_calling,
            }
            for key, config in models.items()
        }
    }


@app.get("/routing")
async def get_routing_config():
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {
        "routing": orchestrator.router.routing,  # type: ignore[attr-defined]
        "available_model_types": list(orchestrator.router.MODELS.keys()),  # type: ignore[attr-defined]
    }


@app.post("/routing/{agent_name}")
async def update_agent_routing(agent_name: str, model_type: str):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    try:
        orchestrator.router.update_routing(agent_name, model_type)  # type: ignore[attr-defined]
        return {
            "status": "success",
            "message": f"Updated routing for {agent_name} to {model_type}",
            "current_routing": orchestrator.router.routing,  # type: ignore[attr-defined]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/stats")
async def get_stats():
    if not client:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return {
        "total_api_calls": client.call_count,
        "total_tokens_used": client.total_tokens_used,
        "estimated_total_cost": round(client.total_tokens_used * 0.003 / 1000, 4),
        "pipeline_variant": str(os.getenv("PIPELINE_VARIANT", "full")).strip().lower(),
        "active_background_generations": len(generation_tasks),
    }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")