from __future__ import annotations

import argparse
import asyncio
import copy
import html
import json
import os
import random
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

from src.apicallhandler import OpenRouterClient
from src.logger import setup_logging, app_logger


TARGET_MEGAPIXELS = 1.0736
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_REWRITE_MODEL = "openai/gpt-4o-mini"
#промпты на английском, для FLUX лучше
CHAR_NEGATIVE = (
    "multiple characters, busy background, scenery, props, furniture, weapon focus, "
    "text, logo, watermark, blurry, low detail, extra limbs, extra fingers, deformed hands, "
    "cropped body, duplicate person, messy background"
)

BG_NEGATIVE = (
    "people, person, character, portrait, close-up, text, logo, watermark, blurry, low detail, "
    "warped perspective, distorted architecture, cluttered foreground, UI, speech bubble"
)

CHAR_REWRITE_SYSTEM = """
You rewrite a character appearance into an English image prompt for a local ComfyUI visual novel sprite workflow.

Return strict JSON:
{
  "positive_prompt": "...",
  "negative_prompt": "..."
}

Rules:
- English only.
- Do NOT mention pose, stance, arm position, gesture, body orientation, camera angle, framing, shot type, composition control, or perspective control.
- Focus on identity, apparent age, face, hair, eyes, build, clothing, materials, colors, iconic details, emotional aura.
- This is a SINGLE visual novel sprite.
- White background is REQUIRED.
- Add strong style markers naturally: visual novel sprite, anime illustration, crisp lineart, polished cel shading, high detail, clean silhouette.
- No props unless absolutely essential to the character identity.
- No text, watermark, UI.
- The positive prompt should end naturally with isolated character / pure white background wording.
- The negative prompt should help avoid extra bodies, busy backgrounds, blur and anatomy issues.
""".strip()

BG_PLAN_SYSTEM = """
You are preparing a ComfyUI background generation plan for a visual novel.

Return strict JSON:
{
  "template_rel_path": "...",
  "positive_prompt": "...",
  "negative_prompt": "..."
}

You must:
1) choose the best composition/depth reference template from available_templates,
2) rewrite the location description into a strong English background prompt.

Rules:
- template_rel_path must be EXACTLY one of the provided rel_path values.
- Choose template by spatial layout and composition, not by color palette.
- English only.
- No characters, no people, no text, no UI.
- Strong style markers: visual novel background, anime background art, environmental concept art, cinematic lighting, atmospheric depth, highly detailed, polished, 16:9.
- Respect location_affordance:
  - if enterable=false or scale=object, keep it clearly exterior / outside-facing,
  - do not imply interior rooms, ceilings, hallways, walls enclosing the viewer unless the affordance allows it.
- Do not mention the filename literally inside the positive prompt.
""".strip()


@dataclass
class TemplateRef:
    rel_path: str
    group: str
    name: str


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def slugify(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(text or ""), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() or "item"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", str(text or "").lower())


def unique_preserve_order(items: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items or []:
        s = str(item or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def parse_json_relaxed(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}


def unwrap_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("context"), dict):
        return payload["context"]
    return payload if isinstance(payload, dict) else {}


def get_descriptions(block: Any) -> List[str]:
    if isinstance(block, dict):
        vals = block.get("descriptions")
        if isinstance(vals, list):
            return [str(x or "") for x in vals]
    return []


def build_setting_text(setting: Dict[str, Any]) -> str:
    if not isinstance(setting, dict):
        return ""

    parts: List[str] = []
    if setting.get("setting"):
        parts.append(str(setting["setting"]))
    if setting.get("world_rules"):
        parts.append(f"World rules: {setting['world_rules']}")
    if setting.get("genre"):
        parts.append(f"Genre: {setting['genre']}")
    if setting.get("time_period"):
        parts.append(f"Time period: {setting['time_period']}")
    return "\n".join(parts)


def resolve_existing_path(raw: str, extra_bases: Optional[List[Path]] = None) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()

    bases = [Path.cwd(), *(extra_bases or [])]
    for base in bases:
        candidate = (base / p).resolve()
        if candidate.exists():
            return candidate

    return (Path.cwd() / p).resolve()


def relative_comfy_input(path: Path, comfy_input_root: Path) -> str:
    return path.resolve().relative_to(comfy_input_root.resolve()).as_posix()


def sync_reference_tree(src_dir: Path, comfy_input_root: Path) -> Path:
    src_dir = src_dir.resolve()
    comfy_input_root = comfy_input_root.resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {src_dir}")

    try:
        src_dir.relative_to(comfy_input_root)
        return src_dir
    except Exception:
        pass

    dst_dir = comfy_input_root / src_dir.name

    for src_path in src_dir.rglob("*"):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            (not dst_path.exists())
            or dst_path.stat().st_size != src_path.stat().st_size
            or int(dst_path.stat().st_mtime) < int(src_path.stat().st_mtime)
        ):
            shutil.copy2(src_path, dst_path)

    return dst_dir


def discover_pose_refs(pose_dir: Path, comfy_input_root: Path) -> List[str]:
    refs: List[str] = []
    for p in sorted(pose_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            refs.append(relative_comfy_input(p, comfy_input_root))
    return refs


def discover_depth_templates(depth_dir: Path, comfy_input_root: Path) -> List[TemplateRef]:
    out: List[TemplateRef] = []
    for p in sorted(depth_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = relative_comfy_input(p, comfy_input_root)
        rel_inside = p.relative_to(depth_dir)
        group = rel_inside.parts[0] if len(rel_inside.parts) > 1 else depth_dir.name
        out.append(TemplateRef(rel_path=rel, group=group, name=p.name))
    return out


def fallback_character_prompt(name: str, raw_description: str) -> str:
    raw = clean_text(raw_description)
    return (
        f"{name}, {raw}. single character visual novel sprite, anime illustration, "
        f"crisp clean lineart, polished cel shading, high detail, elegant color design, "
        f"clean silhouette, isolated character, pure white background, no text, no watermark"
    )


def fallback_background_prompt(location_name: str, raw_description: str, affordance: Dict[str, Any]) -> str:
    raw = clean_text(raw_description)
    kind = str((affordance or {}).get("kind") or "mixed")
    scale = str((affordance or {}).get("scale") or "area")
    return (
        f"{location_name}. {raw}. visual novel background, anime background art, environmental concept art, "
        f"cinematic lighting, atmospheric depth, highly detailed, polished, clean composition, 16:9 wide background, "
        f"{kind} environment, {scale} scale, no characters, no people, no text, no watermark"
    )


def choose_template_fallback(
    location_name: str,
    raw_description: str,
    affordance: Dict[str, Any],
    templates: List[TemplateRef],
) -> TemplateRef:
    if not templates:
        raise ValueError("No depth templates found")

    text = " ".join(
        [
            location_name or "",
            raw_description or "",
            json.dumps(affordance or {}, ensure_ascii=False),
        ]
    ).lower()

    want_indoor = str((affordance or {}).get("kind") or "").lower() == "indoor" or any(
        kw in text
        for kw in [
            "room",
            "corridor",
            "hallway",
            "cafe",
            "classroom",
            "cabinet",
            "interior",
            "inside",
            "door",
            "window",
            "school",
            "office",
        ]
    )

    pool = [
        t for t in templates
        if (want_indoor and "indoor" in t.group.lower())
        or ((not want_indoor) and "indoor" not in t.group.lower())
    ]
    if not pool:
        pool = templates

    desc_tokens = set(tokenize(text))
    boosts = [
        (["forest", "woods", "tree"], ["forest", "woods"]),
        (["corridor", "hallway", "hall"], ["corridor", "coridor"]),
        (["cafe", "coffee"], ["cafe"]),
        (["city", "street", "road", "building"], ["city", "street", "buildings"]),
        (["mountain", "path"], ["mountain", "path"]),
        (["banner"], ["banner"]),
        (["tower", "wall"], ["tower", "wall"]),
        (["room"], ["room"]),
        (["object", "statue", "monolith"], ["object"]),
        (["left"], ["left"]),
        (["right"], ["right"]),
        (["center", "centre"], ["center", "centre"]),
    ]

    scored = []
    for t in pool:
        fname = t.name.lower()
        fname_tokens = set(tokenize(fname))

        score = len(desc_tokens & fname_tokens)
        if want_indoor and "indoor" in t.group.lower():
            score += 3
        if (not want_indoor) and "indoor" not in t.group.lower():
            score += 3
        if str((affordance or {}).get("scale") or "").lower() == "object" and "object" in fname:
            score += 4

        for needles, matches in boosts:
            if any(n in text for n in needles) and any(m in fname for m in matches):
                score += 4

        scored.append((score, t.rel_path, t))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


class ComfyUIClient:
    def __init__(self, base_url: str, timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        response = await self.http.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
        )
        response.raise_for_status()
        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return str(prompt_id)

    async def wait_for_history(
        self,
        prompt_id: str,
        timeout_s: float = 3600.0,
        poll_s: float = 1.5,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            response = await self.http.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            data = response.json() or {}
            item = data.get(prompt_id)

            if item:
                status = item.get("status") or {}
                if str(status.get("status_str") or "").lower() == "error":
                    raise RuntimeError(f"ComfyUI prompt failed: {json.dumps(status, ensure_ascii=False)}")

                if item.get("outputs"):
                    return item

                if status.get("completed") and not item.get("outputs"):
                    raise RuntimeError(
                        f"ComfyUI prompt completed without outputs: "
                        f"{json.dumps(item, ensure_ascii=False)[:1000]}"
                    )

            await asyncio.sleep(poll_s)

        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

    @staticmethod
    def extract_images(history_item: Dict[str, Any], preferred_node_id: str = "9") -> List[Dict[str, Any]]:
        outputs = history_item.get("outputs") or {}

        preferred = outputs.get(str(preferred_node_id)) or {}
        preferred_images = preferred.get("images") or []
        if preferred_images:
            return preferred_images

        flat: List[Dict[str, Any]] = []
        for node_data in outputs.values():
            flat.extend(node_data.get("images") or [])
        return flat

    async def download_image(self, image_meta: Dict[str, Any], dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "filename": image_meta.get("filename", ""),
            "subfolder": image_meta.get("subfolder", ""),
            "type": image_meta.get("type", "output"),
        }

        response = await self.http.get(f"{self.base_url}/view", params=params)
        response.raise_for_status()
        dst.write_bytes(response.content)

    async def run_and_save(
        self,
        workflow: Dict[str, Any],
        dst: Path,
        preferred_node_id: str = "9",
    ) -> Dict[str, Any]:
        prompt_id = await self.queue_prompt(workflow)
        history_item = await self.wait_for_history(prompt_id)
        images = self.extract_images(history_item, preferred_node_id=preferred_node_id)

        if not images:
            raise RuntimeError(f"No images returned for ComfyUI prompt {prompt_id}")

        await self.download_image(images[0], dst)

        return {
            "prompt_id": prompt_id,
            "image_meta": images[0],
            "history": history_item,
        }


async def llm_json(
    client: OpenRouterClient,
    model: str,
    system_prompt: str,
    payload: Dict[str, Any],
    operation_name: str,
) -> Dict[str, Any]:
    response = await client.generate_completion(
        model=model,
        temperature=0.2,
        system_prompt=system_prompt,
        prompt=json.dumps(payload, ensure_ascii=False),
        response_format={"type": "json_object"},
        operation_name=operation_name,
    )
    raw = (((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    return parse_json_relaxed(raw)


async def rewrite_character_prompt(
    llm_client: OpenRouterClient,
    model: str,
    *,
    name: str,
    raw_description: str,
    setting_text: str,
) -> tuple[str, str]:
    payload = {
        "character_name": name,
        "raw_description": raw_description,
        "setting": setting_text,
    }
    data = await llm_json(
        llm_client,
        model,
        CHAR_REWRITE_SYSTEM,
        payload,
        operation_name=f"rewrite_char_prompt_{slugify(name)}",
    )

    positive = clean_text(data.get("positive_prompt") or "")
    negative = clean_text(data.get("negative_prompt") or "")

    if not positive:
        positive = fallback_character_prompt(name, raw_description)
    if not negative:
        negative = CHAR_NEGATIVE

    return positive, negative


async def plan_location_background(
    llm_client: OpenRouterClient,
    model: str,
    *,
    location_name: str,
    raw_description: str,
    setting_text: str,
    affordance: Dict[str, Any],
    templates: List[TemplateRef],
) -> tuple[TemplateRef, str, str]:
    payload = {
        "location_name": location_name,
        "raw_description": raw_description,
        "setting": setting_text,
        "location_affordance": affordance,
        "available_templates": [
            {"rel_path": t.rel_path, "group": t.group, "name": t.name}
            for t in templates
        ],
    }

    data = await llm_json(
        llm_client,
        model,
        BG_PLAN_SYSTEM,
        payload,
        operation_name=f"rewrite_bg_prompt_{slugify(location_name)}",
    )

    template_rel = clean_text(data.get("template_rel_path") or "")
    chosen = next((t for t in templates if t.rel_path == template_rel), None)
    if chosen is None:
        chosen = choose_template_fallback(location_name, raw_description, affordance, templates)

    positive = clean_text(data.get("positive_prompt") or "")
    negative = clean_text(data.get("negative_prompt") or "")

    if not positive:
        positive = fallback_background_prompt(location_name, raw_description, affordance)
    if not negative:
        negative = BG_NEGATIVE

    return chosen, positive, negative


def load_workflow(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_nodes(workflow: Dict[str, Any], node_ids: List[str], label: str) -> None:
    missing = [node_id for node_id in node_ids if node_id not in workflow]
    if missing:
        raise KeyError(f"{label} workflow is missing expected nodes: {missing}")


def patch_1080p_nodes(workflow: Dict[str, Any]) -> None:
    if "74" in workflow:
        workflow["74"].setdefault("inputs", {})
        workflow["74"]["inputs"]["megapixels"] = TARGET_MEGAPIXELS
        workflow["74"]["inputs"]["resolution_steps"] = 1

    if "73" in workflow:
        workflow["73"].setdefault("inputs", {})
        workflow["73"]["inputs"]["displaytext"] = "W: 1920, H: 1080"


def build_person_workflow(
    base: Dict[str, Any],
    *,
    positive_prompt: str,
    negative_prompt: str,
    pose_image: str,
    filename_prefix: str,
    seed: int,
) -> Dict[str, Any]:
    workflow = copy.deepcopy(base)
    require_nodes(workflow, ["3", "6", "7", "9", "74", "78"], "Person")
    patch_1080p_nodes(workflow)

    workflow["3"]["inputs"]["seed"] = int(seed)
    workflow["6"]["inputs"]["text"] = positive_prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["9"]["inputs"]["filename_prefix"] = filename_prefix
    workflow["78"]["inputs"]["image"] = pose_image
    return workflow


def build_location_workflow(
    base: Dict[str, Any],
    *,
    positive_prompt: str,
    negative_prompt: str,
    ref_image: str,
    filename_prefix: str,
    seed: int,
) -> Dict[str, Any]:
    workflow = copy.deepcopy(base)
    require_nodes(workflow, ["3", "6", "7", "9", "74", "78"], "Location")
    patch_1080p_nodes(workflow)

    workflow["3"]["inputs"]["seed"] = int(seed)
    workflow["6"]["inputs"]["text"] = positive_prompt
    workflow["7"]["inputs"]["text"] = negative_prompt
    workflow["9"]["inputs"]["filename_prefix"] = filename_prefix
    workflow["78"]["inputs"]["image"] = ref_image
    return workflow


def validate_context(ctx: Dict[str, Any]) -> None:
    if not isinstance(ctx, dict):
        raise ValueError("Input JSON must be an object")

    for key in ("char_list", "loc_list", "scene_contracts", "scenes"):
        if not ctx.get(key):
            raise ValueError(f"Missing or empty '{key}' in generation JSON")


def build_web_data(ctx: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    contracts = ctx.get("scene_contracts") or []
    scenes = ctx.get("scenes") or {}

    contracts_by_id: Dict[str, Dict[str, Any]] = {}
    branch_map: Dict[str, List[Dict[str, Any]]] = {}

    for contract in contracts:
        if not isinstance(contract, dict) or not contract.get("id"):
            continue
        scene_id = str(contract["id"])
        branch_id = str(contract.get("branch_id") or "main")
        contracts_by_id[scene_id] = contract
        branch_map.setdefault(branch_id, []).append(contract)

    if "main" not in branch_map or not branch_map["main"]:
        raise ValueError("Could not find main branch contracts")

    next_scene_map: Dict[str, str] = {}
    for branch_id, items in branch_map.items():
        items.sort(key=lambda x: int(x.get("branch_order") or 0))
        for i in range(len(items) - 1):
            next_scene_map[str(items[i]["id"])] = str(items[i + 1]["id"])

    main_sorted = sorted(branch_map["main"], key=lambda x: int(x.get("branch_order") or 0))
    start_scene_id = str(main_sorted[0]["id"])

    web_scenes: Dict[str, Any] = {}
    for scene_id, script in scenes.items():
        sid = str(scene_id)
        contract = contracts_by_id.get(sid, {})
        if not isinstance(script, dict):
            continue

        web_scenes[sid] = {
            "id": sid,
            "location": contract.get("location"),
            "pov_character": contract.get("pov_character"),
            "present_characters": contract.get("present_characters") or [],
            "lines": script.get("lines") or [],
            "choices": script.get("choices") or [],
            "summary": script.get("summary") or contract.get("summary") or "",
            "branch_id": script.get("branch_id") or contract.get("branch_id") or "main",
            "branch_order": script.get("branch_order") or contract.get("branch_order") or 0,
        }

    return {
        "title": f"VN - {ctx.get('generation_id') or 'export'}",
        "generation_id": ctx.get("generation_id"),
        "start_scene_id": start_scene_id,
        "next_scene_map": next_scene_map,
        "scenes": web_scenes,
        "assets": {
            "characters": {
                name: item["web_path"]
                for name, item in (manifest.get("characters") or {}).items()
            },
            "locations": {
                name: item["web_path"]
                for name, item in (manifest.get("locations") or {}).items()
            },
        },
    }


def render_html(web_data: Dict[str, Any]) -> str:
    data_json = json.dumps(web_data, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(str(web_data.get("title") or "Visual Novel"))

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: dark;
      --ui-bg: rgba(8, 11, 17, 0.78);
      --ui-border: rgba(255,255,255,0.14);
      --text: #f4f7fb;
      --muted: #b9c2d0;
      --accent: #8ab4ff;
      --accent2: #c084fc;
    }

    * { box-sizing: border-box; }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      background: #000;
      overflow: hidden;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    #game {
      position: relative;
      width: 100vw;
      height: 100vh;
      background: #000;
    }

    #bg {
      position: absolute;
      inset: 0;
      background-size: cover;
      background-position: center center;
      background-repeat: no-repeat;
      transition: background-image 180ms ease-in-out;
      filter: saturate(1.04) contrast(1.02);
    }

    #bg::after {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(to top, rgba(0,0,0,0.68) 0%, rgba(0,0,0,0.16) 38%, rgba(0,0,0,0.34) 100%);
    }

    #sprite-layer {
      position: absolute;
      inset: 0;
      overflow: hidden;
      pointer-events: none;
    }

    .sprite {
      position: absolute;
      bottom: 18vh;
      transform: translateX(-50%);
      object-fit: contain;
      max-width: 42vw;
      transition: opacity 120ms ease, transform 120ms ease, filter 120ms ease;
      filter: drop-shadow(0 16px 28px rgba(0,0,0,0.34));
    }

    .sprite.inactive {
      opacity: 0.62;
      filter: brightness(0.82) grayscale(0.06) drop-shadow(0 14px 24px rgba(0,0,0,0.30));
    }

    .sprite.active {
      opacity: 1;
      transform: translateX(-50%) scale(1.015);
      filter: drop-shadow(0 18px 34px rgba(0,0,0,0.42));
    }

    #location {
      position: absolute;
      top: 18px;
      right: 18px;
      z-index: 4;
      padding: 9px 13px;
      border-radius: 999px;
      border: 1px solid var(--ui-border);
      background: rgba(10, 14, 21, 0.56);
      color: var(--muted);
      font-size: 13px;
      backdrop-filter: blur(8px);
    }

    #ui {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 5;
      padding: 20px;
    }

    #namebox {
      display: none;
      margin: 0 0 8px 14px;
      width: fit-content;
      padding: 8px 14px;
      border-radius: 14px;
      background: rgba(20, 26, 38, 0.9);
      border: 1px solid var(--ui-border);
      color: #fff;
      font-weight: 700;
      font-size: 15px;
    }

    #textbox {
      min-height: 166px;
      padding: 18px 20px 16px;
      border-radius: 18px;
      background: var(--ui-bg);
      border: 1px solid var(--ui-border);
      backdrop-filter: blur(10px);
      box-shadow: 0 18px 46px rgba(0,0,0,0.30);
    }

    #text {
      min-height: 98px;
      color: var(--text);
      font-size: 24px;
      line-height: 1.6;
      white-space: pre-wrap;
    }

    #hint {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }

    #choices {
      display: none;
      margin-top: 14px;
      gap: 10px;
      grid-template-columns: 1fr;
    }

    .choice-btn {
      width: 100%;
      padding: 15px 16px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(18, 24, 36, 0.92);
      color: #fff;
      text-align: left;
      font-size: 17px;
      cursor: pointer;
      transition: transform 90ms ease, background 90ms ease, border-color 90ms ease;
    }

    .choice-btn:hover {
      transform: translateY(-1px);
      background: rgba(28, 36, 56, 0.96);
      border-color: rgba(138, 180, 255, 0.55);
    }

    .choice-btn.fake {
      border-style: dashed;
    }

    #ending {
      position: absolute;
      inset: 0;
      z-index: 20;
      display: none;
      place-items: center;
      background: rgba(0,0,0,0.62);
    }

    #ending.show {
      display: grid;
    }

    #ending .card {
      width: min(92vw, 460px);
      padding: 28px;
      border-radius: 20px;
      background: rgba(12, 16, 24, 0.96);
      border: 1px solid rgba(255,255,255,0.12);
      text-align: center;
      box-shadow: 0 18px 54px rgba(0,0,0,0.44);
    }

    #ending h1 {
      margin: 0 0 8px;
      font-size: 34px;
    }

    #ending p {
      margin: 0 0 20px;
      color: var(--muted);
    }

    #restart-btn {
      border: none;
      border-radius: 14px;
      padding: 12px 18px;
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
    }

    @media (max-width: 900px) {
      #text { font-size: 20px; }
      .sprite { bottom: 22vh; max-width: 62vw; }
    }
  </style>
</head>
<body>
  <div id="game">
    <div id="bg"></div>
    <div id="sprite-layer"></div>
    <div id="location"></div>

    <div id="ui">
      <div id="namebox"></div>
      <div id="textbox">
        <div id="text"></div>
        <div id="hint">Click / Space / Enter</div>
      </div>
      <div id="choices"></div>
    </div>

    <div id="ending">
      <div class="card">
        <h1>The End</h1>
        <p>You reached the end of this route.</p>
        <button id="restart-btn" type="button">Restart</button>
      </div>
    </div>
  </div>

  <script id="vn-data" type="application/json">__VN_DATA__</script>
  <script>
    const VN_DATA = JSON.parse(document.getElementById("vn-data").textContent);

    const dom = {
      game: document.getElementById("game"),
      bg: document.getElementById("bg"),
      spriteLayer: document.getElementById("sprite-layer"),
      location: document.getElementById("location"),
      namebox: document.getElementById("namebox"),
      text: document.getElementById("text"),
      hint: document.getElementById("hint"),
      choices: document.getElementById("choices"),
      ending: document.getElementById("ending"),
      restart: document.getElementById("restart-btn"),
    };

    const state = {
      sceneId: VN_DATA.start_scene_id,
      lineIndex: -1,
      waitingChoice: false,
      ended: false,
    };

    function normalizeName(value) {
      return String(value || "").trim().toLowerCase();
    }

    function sceneById(sceneId) {
      return VN_DATA.scenes[String(sceneId)] || null;
    }

    function positionsFor(count) {
      if (count <= 0) return [];
      if (count === 1) return [50];
      if (count === 2) return [32, 68];
      if (count === 3) return [18, 50, 82];
      if (count === 4) return [12, 37, 63, 88];

      const start = 10;
      const end = 90;
      const step = (end - start) / Math.max(1, count - 1);
      return Array.from({ length: count }, (_, i) => start + i * step);
    }

    function spriteHeight(count) {
      if (count >= 5) return 48;
      if (count === 4) return 54;
      if (count === 3) return 60;
      if (count === 2) return 68;
      return 76;
    }

    function setBackground(scene) {
      const location = scene && scene.location ? scene.location : "";
      dom.location.textContent = location || "";

      const src = location ? VN_DATA.assets.locations[location] : null;
      if (src) {
        dom.bg.style.backgroundImage =
          'linear-gradient(to top, rgba(0,0,0,0.22), rgba(0,0,0,0.08)), url("' + encodeURI(src) + '")';
      } else {
        dom.bg.style.backgroundImage =
          "radial-gradient(circle at top, rgba(70,90,130,0.65), rgba(0,0,0,0.95) 72%)";
      }
    }

    function renderSprites(scene, activeSpeaker) {
      dom.spriteLayer.innerHTML = "";
      if (!scene) return;

      const present = Array.isArray(scene.present_characters) ? scene.present_characters : [];
      const positions = positionsFor(present.length);
      const height = spriteHeight(present.length);

      present.forEach((name, idx) => {
        const src = VN_DATA.assets.characters[name];
        if (!src) return;

        const img = document.createElement("img");
        img.className = "sprite";
        img.src = encodeURI(src);
        img.alt = name;
        img.style.left = positions[idx] + "%";
        img.style.height = height + "vh";

        const isActive = activeSpeaker && normalizeName(activeSpeaker) === normalizeName(name);
        img.classList.add(isActive ? "active" : "inactive");
        img.style.zIndex = isActive ? "10" : String(idx + 1);

        dom.spriteLayer.appendChild(img);
      });
    }

    function findChoiceAfterLine(scene, lineIndex) {
      const choices = Array.isArray(scene.choices) ? scene.choices : [];
      return choices.find((choice) => Number(choice.appears_after_line) === Number(lineIndex)) || null;
    }

    function clearChoices() {
      dom.choices.innerHTML = "";
      dom.choices.style.display = "none";
      state.waitingChoice = false;
      dom.hint.textContent = "Click / Space / Enter";
    }

    function showChoices(choice) {
      state.waitingChoice = true;
      dom.choices.innerHTML = "";
      dom.choices.style.display = "grid";
      dom.hint.textContent = "Choose";

      (choice.options || []).forEach((opt) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "choice-btn" + (opt.is_fake ? " fake" : "");
        btn.textContent = opt.text || "Continue";
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          chooseOption(opt);
        });
        dom.choices.appendChild(btn);
      });
    }

    function renderLine(scene, line) {
      const type = String(line.type || "narration");
      const speaker = line.speaker || scene.pov_character || "";
      dom.text.textContent = line.text || "";

      if (type === "narration") {
        dom.namebox.style.display = "none";
        renderSprites(scene, null);
        return;
      }

      dom.namebox.style.display = "inline-block";
      dom.namebox.textContent = type === "thought" ? (speaker + " (thought)") : speaker;
      renderSprites(scene, speaker);
    }

    function nextSceneId(scene) {
      return VN_DATA.next_scene_map[String(scene.id)] || null;
    }

    function endGame() {
      state.ended = true;
      state.waitingChoice = true;
      dom.ending.classList.add("show");
      dom.hint.textContent = "Route complete";
    }

    function hideEnding() {
      dom.ending.classList.remove("show");
      state.ended = false;
    }

    function goToScene(sceneId) {
      const scene = sceneById(sceneId);
      if (!scene) {
        endGame();
        return;
      }

      state.sceneId = String(sceneId);
      state.lineIndex = -1;
      clearChoices();
      setBackground(scene);
      renderSprites(scene, null);
      advance();
    }

    function chooseOption(opt) {
      clearChoices();
      const target = opt && opt.leads_to_scene_id ? String(opt.leads_to_scene_id) : null;
      if (!target) {
        endGame();
        return;
      }
      goToScene(target);
    }

    function advance() {
      if (state.waitingChoice || state.ended) return;

      const scene = sceneById(state.sceneId);
      if (!scene) {
        endGame();
        return;
      }

      if (state.lineIndex + 1 < (scene.lines || []).length) {
        state.lineIndex += 1;
        const line = scene.lines[state.lineIndex];
        renderLine(scene, line);

        const choice = findChoiceAfterLine(scene, state.lineIndex);
        if (choice) {
          showChoices(choice);
        }
        return;
      }

      const nextId = nextSceneId(scene);
      if (nextId) {
        goToScene(nextId);
        return;
      }

      endGame();
    }

    function preloadAssets() {
      const paths = [];
      Object.values(VN_DATA.assets.characters || {}).forEach((p) => paths.push(p));
      Object.values(VN_DATA.assets.locations || {}).forEach((p) => paths.push(p));

      paths.forEach((src) => {
        const img = new Image();
        img.src = encodeURI(src);
      });
    }

    dom.game.addEventListener("click", (ev) => {
      if (ev.target.closest("#choices")) return;
      if (ev.target.closest("#restart-btn")) return;
      advance();
    });

    window.addEventListener("keydown", (ev) => {
      if (state.ended) return;
      if (ev.code === "Space" || ev.code === "Enter") {
        ev.preventDefault();
        advance();
      }
    });

    dom.restart.addEventListener("click", (ev) => {
      ev.stopPropagation();
      hideEnding();
      goToScene(VN_DATA.start_scene_id);
    });

    preloadAssets();
    goToScene(VN_DATA.start_scene_id);
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", title).replace("__VN_DATA__", data_json)


async def generate_character_assets(
    *,
    ctx: Dict[str, Any],
    llm_client: OpenRouterClient,
    comfy_client: ComfyUIClient,
    rewrite_model: str,
    person_workflow: Dict[str, Any],
    pose_refs: List[str],
    out_dir: Path,
    rng: random.Random,
    manifest: Dict[str, Any],
) -> None:
    char_list = unique_preserve_order(ctx.get("char_list") or [])
    descriptions = get_descriptions(ctx.get("char_appearance"))
    setting_text = build_setting_text(ctx.get("setting") or {})

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(char_list, start=1):
        raw_desc = descriptions[idx - 1] if idx - 1 < len(descriptions) and descriptions[idx - 1].strip() else name
        positive, negative = await rewrite_character_prompt(
            llm_client,
            rewrite_model,
            name=name,
            raw_description=raw_desc,
            setting_text=setting_text,
        )

        pose_rel = rng.choice(pose_refs)
        seed = rng.randint(1, 2**63 - 1)
        filename_prefix = f"vnexp_char_{idx:03d}_{slugify(name)}"

        workflow = build_person_workflow(
            person_workflow,
            positive_prompt=positive,
            negative_prompt=negative,
            pose_image=pose_rel,
            filename_prefix=filename_prefix,
            seed=seed,
        )

        out_path = out_dir / f"{idx:03d}_{slugify(name)}.png"
        app_logger.info(f"[CHAR {idx}/{len(char_list)}] {name} | pose={pose_rel}")

        result = await comfy_client.run_and_save(workflow, out_path, preferred_node_id="9")

        manifest["characters"][name] = {
            "web_path": out_path.relative_to(out_dir.parent.parent).as_posix(),
            "absolute_path": str(out_path),
            "prompt": positive,
            "negative_prompt": negative,
            "reference": pose_rel,
            "seed": seed,
            "comfy_prompt_id": result["prompt_id"],
            "comfy_output": result["image_meta"],
        }


async def generate_location_assets(
    *,
    ctx: Dict[str, Any],
    llm_client: OpenRouterClient,
    comfy_client: ComfyUIClient,
    rewrite_model: str,
    location_workflow: Dict[str, Any],
    templates: List[TemplateRef],
    out_dir: Path,
    rng: random.Random,
    manifest: Dict[str, Any],
) -> None:
    loc_list = unique_preserve_order(ctx.get("loc_list") or [])
    descriptions = get_descriptions(ctx.get("loc_description"))
    loc_canons = ctx.get("loc_canons") or {}
    loc_affordances = ctx.get("loc_affordances") or {}
    setting_text = build_setting_text(ctx.get("setting") or {})

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, location_name in enumerate(loc_list, start=1):
        raw_desc = loc_canons.get(location_name)
        if not raw_desc:
            raw_desc = descriptions[idx - 1] if idx - 1 < len(descriptions) and descriptions[idx - 1].strip() else location_name

        affordance = loc_affordances.get(location_name) if isinstance(loc_affordances, dict) else {}
        if not isinstance(affordance, dict):
            affordance = {}

        template, positive, negative = await plan_location_background(
            llm_client,
            rewrite_model,
            location_name=location_name,
            raw_description=raw_desc,
            setting_text=setting_text,
            affordance=affordance,
            templates=templates,
        )

        seed = rng.randint(1, 2**63 - 1)
        filename_prefix = f"vnexp_bg_{idx:03d}_{slugify(location_name)}"

        workflow = build_location_workflow(
            location_workflow,
            positive_prompt=positive,
            negative_prompt=negative,
            ref_image=template.rel_path,
            filename_prefix=filename_prefix,
            seed=seed,
        )

        out_path = out_dir / f"{idx:03d}_{slugify(location_name)}.png"
        app_logger.info(
            f"[BG {idx}/{len(loc_list)}] {location_name} | template={template.rel_path}"
        )

        result = await comfy_client.run_and_save(workflow, out_path, preferred_node_id="9")

        manifest["locations"][location_name] = {
            "web_path": out_path.relative_to(out_dir.parent.parent).as_posix(),
            "absolute_path": str(out_path),
            "prompt": positive,
            "negative_prompt": negative,
            "reference": template.rel_path,
            "seed": seed,
            "comfy_prompt_id": result["prompt_id"],
            "comfy_output": result["image_meta"],
            "affordance": affordance,
        }


def default_out_dir(input_json: Path) -> Path:
    if input_json.name == "final.json":
        return input_json.parent / "web"
    return input_json.parent / f"{input_json.stem}_web"
async def export_web_from_json(
    input_json: str | Path,
    *,
    out_dir: str | Path | None = None,
    comfy_url: Optional[str] = None,
    comfy_input_root: Optional[str] = None,
    person_workflow: Optional[str] = None,
    location_workflow: Optional[str] = None,
    pose_ref_dir: Optional[str] = None,
    depth_ref_dir: Optional[str] = None,
    rewrite_model: Optional[str] = None,
    seed: int = 1337,
    llm_client: Optional[OpenRouterClient] = None,
) -> Dict[str, Any]:
    load_dotenv()

    input_json_path = resolve_existing_path(str(input_json))
    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    ctx = unwrap_context(payload)
    validate_context(ctx)

    out_dir_path = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else default_out_dir(input_json_path).resolve()
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    comfy_url = comfy_url or os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

    comfy_input_root_raw = comfy_input_root or os.getenv("COMFYUI_INPUT_ROOT")
    if not comfy_input_root_raw:
        raise ValueError("COMFYUI_INPUT_ROOT is required")

    comfy_input_root_path = Path(comfy_input_root_raw).expanduser().resolve()
    if not comfy_input_root_path.exists():
        raise FileNotFoundError(f"ComfyUI input root not found: {comfy_input_root_path}")

    person_workflow_path = resolve_existing_path(
        str(person_workflow or os.getenv("PERSON_WORKFLOW", "Person.json")),
        extra_bases=[input_json_path.parent],
    )
    location_workflow_path = resolve_existing_path(
        str(location_workflow or os.getenv("LOCATION_WORKFLOW", "Location.json")),
        extra_bases=[input_json_path.parent],
    )
    pose_ref_src = resolve_existing_path(
        str(pose_ref_dir or os.getenv("POSE_REF_DIR", "pose-ref")),
        extra_bases=[input_json_path.parent],
    )
    depth_ref_src = resolve_existing_path(
        str(depth_ref_dir or os.getenv("DEPTH_REF_DIR", "deph-ref")),
        extra_bases=[input_json_path.parent],
    )

    rewrite_model_name = rewrite_model or os.getenv("PROMPT_REWRITE_MODEL", DEFAULT_REWRITE_MODEL)

    person_workflow_obj = load_workflow(person_workflow_path)
    location_workflow_obj = load_workflow(location_workflow_path)

    synced_pose_dir = sync_reference_tree(pose_ref_src, comfy_input_root_path)
    synced_depth_dir = sync_reference_tree(depth_ref_src, comfy_input_root_path)

    pose_refs = discover_pose_refs(synced_pose_dir, comfy_input_root_path)
    templates = discover_depth_templates(synced_depth_dir, comfy_input_root_path)

    if not pose_refs:
        raise ValueError(f"No pose refs found in {synced_pose_dir}")
    if not templates:
        raise ValueError(f"No depth templates found in {synced_depth_dir}")

    active_llm_client = llm_client
    own_llm_client = active_llm_client is None

    if active_llm_client is None:
        api_key = os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        if "openrouter.ai" in base_url and not api_key:
            raise ValueError("API_KEY or LLM_API_KEY is required for prompt rewriting through OpenRouter")
        active_llm_client = OpenRouterClient(api_key=api_key, base_url=base_url)

    comfy_client = ComfyUIClient(comfy_url)
    rng = random.Random(seed)

    assets_root = out_dir_path / "assets"
    char_assets_dir = assets_root / "characters"
    bg_assets_dir = assets_root / "backgrounds"

    manifest: Dict[str, Any] = {
        "generation_id": ctx.get("generation_id"),
        "source_json": str(input_json_path),
        "characters": {},
        "locations": {},
        "settings": {
            "rewrite_model": rewrite_model_name,
            "comfy_url": comfy_url,
            "comfy_input_root": str(comfy_input_root_path),
            "person_workflow": str(person_workflow_path),
            "location_workflow": str(location_workflow_path),
            "pose_ref_dir": str(synced_pose_dir),
            "depth_ref_dir": str(synced_depth_dir),
        },
    }

    try:
        app_logger.info("=" * 60)
        app_logger.info(f"INPUT JSON: {input_json_path}")
        app_logger.info(f"OUTPUT DIR: {out_dir_path}")
        app_logger.info(f"POSE REFS: {synced_pose_dir}")
        app_logger.info(f"DEPTH REFS: {synced_depth_dir}")
        app_logger.info(f"COMFYUI: {comfy_url}")
        app_logger.info("=" * 60)

        await generate_character_assets(
            ctx=ctx,
            llm_client=active_llm_client,
            comfy_client=comfy_client,
            rewrite_model=rewrite_model_name,
            person_workflow=person_workflow_obj,
            pose_refs=pose_refs,
            out_dir=char_assets_dir,
            rng=rng,
            manifest=manifest,
        )

        await generate_location_assets(
            ctx=ctx,
            llm_client=active_llm_client,
            comfy_client=comfy_client,
            rewrite_model=rewrite_model_name,
            location_workflow=location_workflow_obj,
            templates=templates,
            out_dir=bg_assets_dir,
            rng=rng,
            manifest=manifest,
        )

        web_data = build_web_data(ctx, manifest)
        html_text = render_html(web_data)

        index_html_path = out_dir_path / "index.html"
        manifest_path = out_dir_path / "manifest.json"

        index_html_path.write_text(html_text, encoding="utf-8")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "out_dir": str(out_dir_path),
            "index_html": str(index_html_path),
            "manifest": str(manifest_path),
            "assets": {
                "characters": {
                    name: {
                        "web_path": item.get("web_path"),
                        "absolute_path": item.get("absolute_path"),
                    }
                    for name, item in (manifest.get("characters") or {}).items()
                },
                "locations": {
                    name: {
                        "web_path": item.get("web_path"),
                        "absolute_path": item.get("absolute_path"),
                    }
                    for name, item in (manifest.get("locations") or {}).items()
                },
            },
        }

    finally:
        if own_llm_client and active_llm_client is not None:
            await active_llm_client.close()
        await comfy_client.aclose()

async def amain() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate local ComfyUI VN assets and export a playable HTML VN from a generated VN JSON.",
    )
    parser.add_argument("input_json", help="Path to final.json or exported generation JSON")
    parser.add_argument("--out-dir", default=None, help="Output directory for index.html and assets")
    parser.add_argument("--comfy-url", default=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-input-root", default=os.getenv("COMFYUI_INPUT_ROOT"), required=not bool(os.getenv("COMFYUI_INPUT_ROOT")))
    parser.add_argument("--person-workflow", default="Person.json")
    parser.add_argument("--location-workflow", default="Location.json")
    parser.add_argument("--pose-ref-dir", default="pose-ref")
    parser.add_argument("--depth-ref-dir", default="deph-ref")
    parser.add_argument("--rewrite-model", default=os.getenv("PROMPT_REWRITE_MODEL", DEFAULT_REWRITE_MODEL))
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()
    setup_logging("INFO")

    input_json = resolve_existing_path(args.input_json)
    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    ctx = unwrap_context(payload)
    validate_context(ctx)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir(input_json).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    comfy_input_root = Path(args.comfy_input_root).expanduser().resolve()
    if not comfy_input_root.exists():
        raise FileNotFoundError(f"ComfyUI input root not found: {comfy_input_root}")

    person_workflow_path = resolve_existing_path(args.person_workflow, extra_bases=[input_json.parent])
    location_workflow_path = resolve_existing_path(args.location_workflow, extra_bases=[input_json.parent])
    pose_ref_src = resolve_existing_path(args.pose_ref_dir, extra_bases=[input_json.parent])
    depth_ref_src = resolve_existing_path(args.depth_ref_dir, extra_bases=[input_json.parent])

    person_workflow = load_workflow(person_workflow_path)
    location_workflow = load_workflow(location_workflow_path)

    synced_pose_dir = sync_reference_tree(pose_ref_src, comfy_input_root)
    synced_depth_dir = sync_reference_tree(depth_ref_src, comfy_input_root)

    pose_refs = discover_pose_refs(synced_pose_dir, comfy_input_root)
    templates = discover_depth_templates(synced_depth_dir, comfy_input_root)

    if not pose_refs:
        raise ValueError(f"No pose refs found in {synced_pose_dir}")
    if not templates:
        raise ValueError(f"No depth templates found in {synced_depth_dir}")

    api_key = os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    if "openrouter.ai" in base_url and not api_key:
        raise ValueError("API_KEY or LLM_API_KEY is required for prompt rewriting through OpenRouter")

    llm_client = OpenRouterClient(api_key=api_key, base_url=base_url)
    comfy_client = ComfyUIClient(args.comfy_url)
    rng = random.Random(args.seed)

    assets_root = out_dir / "assets"
    char_assets_dir = assets_root / "characters"
    bg_assets_dir = assets_root / "backgrounds"

    manifest: Dict[str, Any] = {
        "generation_id": ctx.get("generation_id"),
        "source_json": str(input_json),
        "characters": {},
        "locations": {},
        "settings": {
            "rewrite_model": args.rewrite_model,
            "comfy_url": args.comfy_url,
            "comfy_input_root": str(comfy_input_root),
            "person_workflow": str(person_workflow_path),
            "location_workflow": str(location_workflow_path),
            "pose_ref_dir": str(synced_pose_dir),
            "depth_ref_dir": str(synced_depth_dir),
        },
    }

    try:
        app_logger.info("=" * 60)
        app_logger.info(f"INPUT JSON: {input_json}")
        app_logger.info(f"OUTPUT DIR: {out_dir}")
        app_logger.info(f"POSE REFS: {synced_pose_dir}")
        app_logger.info(f"DEPTH REFS: {synced_depth_dir}")
        app_logger.info(f"COMFYUI: {args.comfy_url}")
        app_logger.info("=" * 60)

        await generate_character_assets(
            ctx=ctx,
            llm_client=llm_client,
            comfy_client=comfy_client,
            rewrite_model=args.rewrite_model,
            person_workflow=person_workflow,
            pose_refs=pose_refs,
            out_dir=char_assets_dir,
            rng=rng,
            manifest=manifest,
        )

        await generate_location_assets(
            ctx=ctx,
            llm_client=llm_client,
            comfy_client=comfy_client,
            rewrite_model=args.rewrite_model,
            location_workflow=location_workflow,
            templates=templates,
            out_dir=bg_assets_dir,
            rng=rng,
            manifest=manifest,
        )

        web_data = build_web_data(ctx, manifest)
        html_text = render_html(web_data)

        (out_dir / "index.html").write_text(html_text, encoding="utf-8")
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        app_logger.info("=" * 60)
        app_logger.info("DONE")
        app_logger.info(f"HTML: {out_dir / 'index.html'}")
        app_logger.info(f"MANIFEST: {out_dir / 'manifest.json'}")
        app_logger.info("=" * 60)

    finally:
        await llm_client.close()
        await comfy_client.aclose()


if __name__ == "__main__":
    asyncio.run(amain())