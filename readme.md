# Multi-agent System for Visual Novel Generation

This repository contains the source code for the Master's thesis project:

Multi-agent System for Visual Novel Generation  
Russian title: "Мультиагентная система для генерации визуальных новелл"

Institution:
National Research University Higher School of Economics (HSE University)  
Saint Petersburg School of Humanities and Arts

Authors:
Valerii V. Sukmaniuk  
Ilia A. Shakhov


## If you want to see generation examples, check the `output/` directory; for a full playable novel, open the generated `.html` file in Chrome (located in the 'web' subfolder).

## Project summary

The project implements a multi-agent pipeline for automatic generation of visual novels from a user prompt.

The system takes a natural language request and produces a machine-readable visual novel artifact that may include:

- normalized user request
- setting description
- story outline
- branching structure
- character list and character relations
- location list and location graph
- scene contracts
- final scene texts
- interactive choices
- visual assets for characters and backgrounds
- playable HTML export

The main research idea is to replace one-shot story generation with a staged multi-agent pipeline with planning, validation, memory, and controlled visual generation.


## Current repository scope

This repository contains the backend generation code and local HTML export logic.

Main technologies used in the current codebase:

- Python
- FastAPI
- Uvicorn
- Pydantic
- httpx
- OpenRouter-compatible LLM API
- local ComfyUI for image generation
- FLUX-based image workflows through ComfyUI
- HTML export for generated visual novels


## Core files

Main files in the current repository:

- `src/api.py`  
  FastAPI application and HTTP endpoints

- `src/orchestrator.py`  
  Full multi-agent pipeline

- `src/orchestrator_simple.py`  
  Simplified baseline pipeline

- `src/build.py`  
  Local visual asset generation and HTML export

- `src/apicallhandler.py`  
  Unified OpenRouter-compatible client for text, embeddings, and image calls

- `src/router.py`  
  Model routing by agent type

- `src/toolbox.py`  
  Tool definitions used by agents

- `src/agents.py`  
  Agent execution logic

- `src/prompts.py`  
  System prompts for all pipeline stages

- `src/pydantic_schemas.py`  
  Core data schemas for the whole pipeline

- `src/postprocess.py`  
  Choice postprocessing (legacy)

- `src/progress.py`  
  Background task progress tracking


## Repository structure

- `src/`
  - `__init__.py`
  - `agents.py`
  - `api.py`
  - `apicallhandler.py`
  - `build.py`
  - `logger.py`
  - `orchestrator.py`
  - `orchestrator_simple.py`
  - `postprocess.py`
  - `progress.py`
  - `prompts.py`
  - `pydantic_schemas.py`
  - `router.py`
  - `toolbox.py`
  - `utils/`
    - `__init__.py`
    - `artifacts.py`
    - `decorators.py`
    - `fileio.py`
    - `names.py`

Experimental runner files is also present:
- `src/main.py`

This is useful for local experiments, but the main service entry point is `src/api.py`.


## Requirements

Install via `requirements.txt` 

If you use local visual generation, you also need:
- a working ComfyUI installation
- the required image workflows
- reference folders for poses and depth/composition templates
- local model setup for your ComfyUI workflows



## Required external resources for visual generation

For the current codebase, the image and HTML export stage expects the following files or directories, unless you override them in `.env`.

Default names used by the code:

- `Person.json`
- `Location.json`
- `pose-ref/`
- `deph-ref/`


## Environment variables

Create a `.env` file in the project root.

Minimal text-only setup:

```env
API_KEY=your_api_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1
PIPELINE_VARIANT=full
OUTPUT_DIR=output
```

Minimal setup with local image export:

```env
API_KEY=your_api_key_here
LLM_BASE_URL=https://openrouter.ai/api/v1

PIPELINE_VARIANT=full
OUTPUT_DIR=output

COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_INPUT_ROOT=/absolute/path/to/ComfyUI/input

PERSON_WORKFLOW=Person.json
LOCATION_WORKFLOW=Location.json
POSE_REF_DIR=pose-reph
DEPTH_REF_DIR=deph-reph

PROMPT_REWRITE_MODEL=openai/gpt-4o-mini
POSTPROCESS_CHOICES=true
CHOICE_PATCH_USE_LLM=true
```

Other routing and tuning variables may also be used by the system, but the values above are enough for a standard run.


## How to run the API

Start the FastAPI server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Or run the module directly:

```bash
python -m src.api
```

If the service starts correctly, the API will be available at:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`


## Main API endpoints

Basic endpoints:

- `GET /`
- `GET /health`
- `GET /stats`
- `GET /models`
- `GET /routing`

Generation endpoints:

- `POST /generate`
  Blocking generation call

- `POST /generate/start`
  Starts generation in the background

- `GET /generate/progress/{generation_id}`
  Returns current progress state

- `GET /generate/result/{generation_id}`
  Returns final result when ready

- `GET /generate/events/{generation_id}`
  Server-Sent Events stream for progress updates


## Minimal blocking request example

`/generate` requires `mc_name`.

Example request body:

```json
{
  "user_prompt": "Generate a fantasy visual novel about a haunted lighthouse town.",
  "story_length": "medium",
  "max_branches": 3,
  "generate_images": false,
  "time_choice": "средневековье",
  "genre_choice": "фентези",
  "tone_choice": "грустный",
  "mc_name": "Lira",
  "mc_description": "A young cartographer apprentice afraid of losing her memory."
}
```

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Generate a fantasy visual novel about a haunted lighthouse town.",
    "story_length": "medium",
    "max_branches": 3,
    "generate_images": false,
    "time_choice": "средневековье",
    "genre_choice": "фентези",
    "tone_choice": "грустный",
    "mc_name": "Lira",
    "mc_description": "A young cartographer apprentice afraid of losing her memory."
  }'
```


## How the full pipeline works

In simplified form, the full pipeline does the following:

1. Normalize the user request
2. Generate the setting
3. Generate the story outline
4. Extract plot threads
5. Plan branching
6. Generate character and location knowledge
7. Infer location affordances
8. Build scene contracts
9. Validate and enrich contracts
10. Write main route scenes
11. Write branch scenes
12. Save `final.json`
13. If `generate_images=true`, run the local build/export stage from `src/build.py`
14. Save image assets, manifest, and playable HTML export


## Output files

Generation artifacts are saved under:

```text
output/<generation_id>/
```

Depending on the run, this directory may contain:

- `final.json`
- `checkpoints/`
- `events.jsonl`
- `scenes/`
- `state/`
- `contracts/`
- `context/`
- `critics/`

If `generate_images=true` in the full pipeline, it may also contain:

```text
output/<generation_id>/web/
```

with:

- `index.html`
- `manifest.json`
- `assets/characters/...`
- `assets/backgrounds/...`


## Standalone visual export

`src/build.py` can also be used separately on an already generated `final.json`.

Example:

```bash
python -m src.build output/<generation_id>/final.json \
  --comfy-input-root /absolute/path/to/ComfyUI/input
```


## Logging

Logs are written to the `logs/` directory.

Main log files:
- `logs/app_debug.log`
- `logs/api_calls.log`
- `logs/errors.log`
- `logs/agent_activity.log`


## Notes for thesis use

This repository is a research prototype created as part of a Master's thesis.

It is intended to demonstrate:
- multi-agent decomposition of long-form narrative generation
- typed intermediate artifacts
- contract-based scene planning
- memory and branch isolation
- controlled visual asset generation
- backend integration through FastAPI

The implementation is experimental and may require local adjustment of:
- model routing
- environment variables
- ComfyUI workflows
- reference assets
- output paths


## Contact / authors

Authors:
- Valerii Sukmaniuk
- Ilia Shakhov
