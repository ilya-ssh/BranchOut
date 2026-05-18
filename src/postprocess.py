#ЛЕГАСИ КОД!
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.apicallhandler import OpenRouterClient
from src.logger import setup_logging, app_logger


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _parse_json_relaxed(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}


def _shorten(text: str, limit: int = 120) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if len(s) <= limit:
        return s
    s = s[:limit].rsplit(" ", 1)[0].strip()
    return s + "..."


def _first_clause(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.split(r"[.;!?]\s+", s, maxsplit=1)[0].strip()
    return _shorten(s, 120)


def _extract_last_lines(scene_script: Dict[str, Any], n: int = 2) -> List[str]:
    lines = scene_script.get("lines") or []
    out: List[str] = []
    for ln in lines[-n:]:
        if not isinstance(ln, dict):
            continue
        tp = str(ln.get("type") or "")
        sp = ln.get("speaker")
        text = str(ln.get("text") or "").strip()
        if not text:
            continue
        prefix = f"{sp}: " if sp else ""
        out.append(_shorten(f"[{tp}] {prefix}{text}", 180))
    return out


def _extract_first_lines(scene_script: Optional[Dict[str, Any]], n: int = 2) -> List[str]:
    if not scene_script:
        return []
    lines = scene_script.get("lines") or []
    out: List[str] = []
    for ln in lines[:n]:
        if not isinstance(ln, dict):
            continue
        tp = str(ln.get("type") or "")
        sp = ln.get("speaker")
        text = str(ln.get("text") or "").strip()
        if not text:
            continue
        prefix = f"{sp}: " if sp else ""
        out.append(_shorten(f"[{tp}] {prefix}{text}", 180))
    return out


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except Exception:
        return int(default)


def _clean_button_text(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    s = s.strip(" \t\r\n")
    s = re.sub(r"[.。…]+$", "", s).strip()
    max_chars = _env_int("CHOICE_TEXT_MAX_CHARS", 72)
    return _shorten(s, max_chars)


def _build_maps(data: Dict[str, Any]) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, str],
    Dict[str, List[Dict[str, Any]]],
]:
    beats_by_id: Dict[str, Dict[str, Any]] = {}
    for b in ((data.get("outline") or {}).get("beats") or []):
        if isinstance(b, dict) and b.get("id"):
            beats_by_id[str(b["id"])] = b

    scenes_by_id: Dict[str, Dict[str, Any]] = {}
    for sid, sc in (data.get("scenes") or {}).items():
        if isinstance(sc, dict):
            scenes_by_id[str(sid)] = sc

    contracts_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    main_beat_to_scene: Dict[str, str] = {}
    branch_first_scene: Dict[str, str] = {}

    all_contracts = data.get("scene_contracts") or []
    for c in all_contracts:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        bid = str(c.get("branch_id") or "main")
        contracts_by_branch.setdefault(bid, []).append(c)

    for bid, arr in contracts_by_branch.items():
        arr.sort(key=lambda x: int(x.get("branch_order") or 0))
        if bid == "main":
            for c in arr:
                beat_id = str(c.get("beat_id") or "")
                if beat_id and beat_id not in main_beat_to_scene:
                    main_beat_to_scene[beat_id] = str(c["id"])
        else:
            if arr:
                branch_first_scene[bid] = str(arr[0]["id"])

    branches_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    branches = ((data.get("branching") or {}).get("branches") or [])
    for br in branches:
        if not isinstance(br, dict):
            continue
        if str(br.get("id") or "") == "main":
            continue

        from_scene_id = br.get("from_scene_id")
        if not from_scene_id:
            from_beat_id = str(br.get("from_beat_id") or "")
            from_scene_id = main_beat_to_scene.get(from_beat_id)

        if from_scene_id:
            branches_by_scene.setdefault(str(from_scene_id), []).append(br)

    return beats_by_id, scenes_by_id, contracts_by_branch, branch_first_scene, branches_by_scene


def _get_char_type(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    root_ct = data.get("char_type")
    if isinstance(root_ct, dict):
        return root_ct

    st = data.get("story_state_main") or {}
    world = st.get("world") or {}
    ct = world.get("char_type")
    return ct if isinstance(ct, dict) else None


def _fallback_main_text(next_main_contract: Optional[Dict[str, Any]], next_main_script: Optional[Dict[str, Any]]) -> str:
    summary = _first_clause((next_main_script or {}).get("summary") or (next_main_contract or {}).get("summary") or "")
    low = summary.lower()

    rules = [
        ("расслед", "Продолжить расследование"),
        ("довер", "Довериться и проверить"),
        ("побег", "Искать путь к бегству"),
        ("бежать", "Попробовать уйти сейчас"),
        ("сделк", "Обсудить условия сделки"),
        ("спас", "Сначала спасти близкого"),
        ("тайн", "Докопаться до правды"),
        ("архив", "Идти за следующей зацепкой"),
        ("маяк", "Вернуться к маяку"),
        ("карта", "Проверить карту ещё раз"),
    ]
    for needle, text in rules:
        if needle in low:
            return text

    if summary and len(summary) >= 12:
        return summary

    return "Продолжить по намеченному плану"


def _fallback_branch_text(branch_spec: Dict[str, Any], branch_first_contract: Optional[Dict[str, Any]], branch_first_script: Optional[Dict[str, Any]]) -> str:
    summary = _first_clause((branch_first_script or {}).get("summary") or (branch_first_contract or {}).get("summary") or "")
    title = _first_clause(branch_spec.get("title") or "")
    desc = _first_clause(branch_spec.get("description") or "")

    for candidate in [summary, desc, title]:
        if candidate and len(candidate) >= 12 and "ветка" not in candidate.lower():
            return candidate

    ending_tone = str(branch_spec.get("ending_tone") or "").strip().lower()
    if ending_tone == "good":
        return "Рискнуть ради лучшего исхода"
    if ending_tone == "bad":
        return "Пойти на опасный шаг"
    if ending_tone == "bittersweet":
        return "Выбрать путь с ценой"
    if ending_tone == "open":
        return "Оставить исход неопределённым"

    return "Сделать альтернативный выбор"


async def _generate_contextual_choice_texts(
    client: OpenRouterClient,
    *,
    pov_character: str,
    prev_scene: Optional[Dict[str, Any]],
    current_scene_contract: Dict[str, Any],
    current_scene_script: Dict[str, Any],
    next_main_contract: Dict[str, Any],
    next_main_script: Optional[Dict[str, Any]],
    branch_targets: List[Dict[str, Any]],
    char_type: Optional[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    """
    Returns JSON:
    {
      "options": [
        {"branch_id":"main","text":"..."},
        {"branch_id":"branch_01","text":"..."}
      ]
    }
    """

    prompt = """
Ты редактор кнопок выбора для визуальной новеллы.

Тебе дают JSON с контекстом:
- pov_character: POV-персонаж ТЕКУЩЕЙ сцены (это тот, кто принимает решение)
- prev_scene: конец предыдущей сцены (может быть null)
- current_scene: конец текущей сцены (именно ПОСЛЕ этого появляется меню)
- next_main_scene: начало следующей сцены по main
- branch_targets: для каждой ветки — начало ПЕРВОЙ сцены ветки (это непосредственное следствие выбора)
- char_type: роли персонажей (может помочь формулировкам)

Задача:
Придумай КОРОТКИЕ, ЕСТЕСТВЕННЫЕ тексты кнопок выбора в конце current_scene.

Верни строго JSON:
{
  "options": [
    {"branch_id": "main", "text": "..." },
    {"branch_id": "branch_01", "text": "..." }
  ]
}

Правила:
- Должна быть ровно 1 опция для "main" и по 1 опции для КАЖДОГО branch_id из branch_targets.
- Текст — интерфейсный, обычно 3–10 слов. Без кавычек. Желательно без точки в конце.
- Формулируй как РЕШЕНИЕ/ДЕЙСТВИЕ POV-персонажа (инфинитивы/императивы допустимы).
- Не используй общие фразы: "выбрать другой путь", "изменить ход событий", "шаг в неизвестность", "рискнуть", "продолжить".
- Разные кнопки должны быть реально РАЗНЫМИ по намерению (чёткая развилка).
- Не спойлери финал напрямую.
- Не придумывай новых персонажей/локаций — используй только информацию из входа.
- Только JSON.
""".strip()

    before_n = max(1, min(_env_int("CHOICE_CONTEXT_BEFORE_LINES", 2), 8))
    after_n = max(1, min(_env_int("CHOICE_CONTEXT_AFTER_LINES", 2), 8))

    payload = {
        "pov_character": pov_character,
        "prev_scene": (
            {
                "summary": (prev_scene or {}).get("summary"),
                "last_lines": _extract_last_lines(prev_scene or {}, n=before_n),
            }
            if prev_scene
            else None
        ),
        "current_scene": {
            "contract": {
                "id": current_scene_contract.get("id"),
                "location": current_scene_contract.get("location"),
                "pov_character": current_scene_contract.get("pov_character"),
                "present_characters": current_scene_contract.get("present_characters"),
                "summary": current_scene_contract.get("summary"),
            },
            "script": {
                "summary": current_scene_script.get("summary"),
                "last_lines": _extract_last_lines(current_scene_script, n=before_n),
            },
        },
        "next_main_scene": {
            "contract": {
                "id": next_main_contract.get("id"),
                "location": next_main_contract.get("location"),
                "pov_character": next_main_contract.get("pov_character"),
                "present_characters": next_main_contract.get("present_characters"),
                "summary": next_main_contract.get("summary"),
            },
            "script": {
                "summary": (next_main_script or {}).get("summary") if next_main_script else None,
                "first_lines": _extract_first_lines(next_main_script, n=after_n),
            },
        },
        "branch_targets": branch_targets,
        "char_type": char_type or {},
    }

    resp = await client.generate_completion(
        model=model,
        temperature=0.3,
        system_prompt=prompt,
        prompt=json.dumps(payload, ensure_ascii=False),
        response_format={"type": "json_object"},
        operation_name=f"choice_patch_{current_scene_contract.get('id')}",
    )

    raw = (resp["choices"][0]["message"]["content"] or "").strip()
    return _parse_json_relaxed(raw)


def _strip_fake_options_everywhere(data: Dict[str, Any]) -> int:
    """
    Remove any is_fake=true options and drop menus that end up with <2 options.
    Returns number of removed fake options.
    """
    removed = 0
    scenes = data.get("scenes") or {}
    if not isinstance(scenes, dict):
        return 0

    for _, sc in scenes.items():
        if not isinstance(sc, dict):
            continue
        choices = sc.get("choices") or []
        if not isinstance(choices, list) or not choices:
            continue

        new_choices: List[Dict[str, Any]] = []
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            opts = ch.get("options") or []
            if not isinstance(opts, list):
                continue

            real_opts: List[Dict[str, Any]] = []
            for opt in opts:
                if not isinstance(opt, dict):
                    continue
                if bool(opt.get("is_fake", False)) is True:
                    removed += 1
                    continue
                opt["is_fake"] = False
                real_opts.append(opt)

            if len(real_opts) >= 2:
                ch["options"] = real_opts
                new_choices.append(ch)

        sc["choices"] = new_choices

    return removed


async def _rewrite_branch_choices(
    data: Dict[str, Any],
    scenes_by_id: Dict[str, Dict[str, Any]],
    contracts_by_branch: Dict[str, List[Dict[str, Any]]],
    branch_first_scene: Dict[str, str],
    branches_by_scene: Dict[str, List[Dict[str, Any]]],
    *,
    use_llm: bool,
    client: Optional[OpenRouterClient],
    model: str,
) -> int:
    """
    Creates/overwrites choices at divergence points ONLY (no fake buttons).
    Returns number of scenes where divergence choice was written.
    """
    patched_scenes = 0
    main_contracts = contracts_by_branch.get("main") or []
    main_idx_by_scene = {str(c["id"]): i for i, c in enumerate(main_contracts)}
    char_type = _get_char_type(data)

    before_n = max(1, min(_env_int("CHOICE_CONTEXT_BEFORE_LINES", 2), 8))
    after_n = max(1, min(_env_int("CHOICE_CONTEXT_AFTER_LINES", 2), 8))

    for scene_id, branch_specs in branches_by_scene.items():
        if scene_id not in scenes_by_id or scene_id not in main_idx_by_scene:
            continue

        idx = main_idx_by_scene[scene_id]
        if idx + 1 >= len(main_contracts):
            continue

        cur_contract = main_contracts[idx]
        next_main_contract = main_contracts[idx + 1]
        prev_contract = main_contracts[idx - 1] if idx - 1 >= 0 else None

        current_scene_script = scenes_by_id[scene_id]
        prev_scene_script = scenes_by_id.get(str(prev_contract["id"])) if prev_contract else None
        next_main_scene_script = scenes_by_id.get(str(next_main_contract["id"]))

        pov_character = str(cur_contract.get("pov_character") or "")

        branch_targets: List[Dict[str, Any]] = []
        branch_first_contract_by_id: Dict[str, Optional[Dict[str, Any]]] = {}
        branch_first_script_by_id: Dict[str, Optional[Dict[str, Any]]] = {}

        for br in sorted(branch_specs, key=lambda x: str(x.get("id") or "")):
            bid = str(br.get("id") or "").strip()
            if not bid:
                continue

            first_scene_id = branch_first_scene.get(bid)
            first_contract = (contracts_by_branch.get(bid) or [None])[0]
            first_script = scenes_by_id.get(str(first_scene_id)) if first_scene_id else None

            branch_first_contract_by_id[bid] = first_contract if isinstance(first_contract, dict) else None
            branch_first_script_by_id[bid] = first_script if isinstance(first_script, dict) else None

            if not first_scene_id or not isinstance(first_contract, dict):
                continue

            branch_targets.append(
                {
                    "branch_id": bid,
                    "branch_spec": {
                        "id": bid,
                        "title": br.get("title"),
                        "description": br.get("description"),
                        "ending_tone": br.get("ending_tone"),
                        "from_beat_id": br.get("from_beat_id"),
                        "from_scene_id": br.get("from_scene_id"),
                    },
                    "first_scene": {
                        "scene_id": first_scene_id,
                        "contract": {
                            "id": first_contract.get("id"),
                            "location": first_contract.get("location"),
                            "pov_character": first_contract.get("pov_character"),
                            "present_characters": first_contract.get("present_characters"),
                            "summary": first_contract.get("summary"),
                        },
                        "script": {
                            "summary": (first_script or {}).get("summary") if first_script else None,
                            "first_lines": _extract_first_lines(first_script, n=after_n),
                        },
                    },
                }
            )

        text_map: Dict[str, str] = {}
        if use_llm and client is not None and pov_character:
            try:
                generated = await _generate_contextual_choice_texts(
                    client,
                    pov_character=pov_character,
                    prev_scene=prev_scene_script,
                    current_scene_contract=cur_contract,
                    current_scene_script=current_scene_script,
                    next_main_contract=next_main_contract,
                    next_main_script=next_main_scene_script,
                    branch_targets=branch_targets,
                    char_type=char_type,
                    model=model,
                )
                for item in (generated.get("options") or []):
                    if not isinstance(item, dict):
                        continue
                    bid = str(item.get("branch_id") or "").strip()
                    txt = _clean_button_text(item.get("text") or "")
                    if bid and txt:
                        text_map[bid] = txt
            except Exception as e:
                app_logger.warning(f"choice rewrite failed for {scene_id}: {e}")

        main_text = _clean_button_text(text_map.get("main") or "") or _clean_button_text(
            _fallback_main_text(next_main_contract, next_main_scene_script)
        )

        options: List[Dict[str, Any]] = []
        options.append(
            {
                "id": "opt_main",
                "text": main_text,
                "leads_to_scene_id": str(next_main_contract["id"]),
                "leads_to_branch_id": "main",
                "is_fake": False,
            }
        )

        for br in sorted(branch_specs, key=lambda x: str(x.get("id") or "")):
            bid = str(br.get("id") or "").strip()
            if not bid:
                continue

            first_scene_id = branch_first_scene.get(bid)
            if not first_scene_id:
                continue

            first_contract = branch_first_contract_by_id.get(bid)
            first_script = branch_first_script_by_id.get(bid)

            txt = _clean_button_text(text_map.get(bid) or "") or _clean_button_text(
                _fallback_branch_text(br, first_contract, first_script)
            )

            options.append(
                {
                    "id": f"opt_{bid}",
                    "text": txt,
                    "leads_to_scene_id": str(first_scene_id),
                    "leads_to_branch_id": bid,
                    "is_fake": False,
                }
            )

        if len(options) < 2:
            current_scene_script["choices"] = []
            continue

        cur_last = _extract_last_lines(current_scene_script, n=before_n)
        cur_summary = current_scene_script.get("summary")
        cur_loc = cur_contract.get("location")

        for opt in options:
            target_scene_id = str(opt.get("leads_to_scene_id") or "")
            target_script = scenes_by_id.get(target_scene_id)
            target_contract: Optional[Dict[str, Any]] = None

            if str(opt.get("leads_to_branch_id") or "") == "main":
                target_contract = next_main_contract
            else:
                bid = str(opt.get("leads_to_branch_id") or "")
                arr = contracts_by_branch.get(bid) or []
                if arr and isinstance(arr[0], dict) and str(arr[0].get("id") or "") == target_scene_id:
                    target_contract = arr[0]
                else:
                    for c in arr:
                        if isinstance(c, dict) and str(c.get("id") or "") == target_scene_id:
                            target_contract = c
                            break

            opt["pov_character"] = pov_character
            opt["target_pov_character"] = (target_contract or {}).get("pov_character") if target_contract else None

            opt["context_before"] = {
                "scene_id": str(scene_id),
                "location": cur_loc,
                "pov_character": pov_character,
                "summary": cur_summary,
                "last_lines": cur_last,
            }
            opt["context_after"] = {
                "scene_id": target_scene_id,
                "location": (target_contract or {}).get("location") if target_contract else None,
                "pov_character": (target_contract or {}).get("pov_character") if target_contract else None,
                "summary": (target_script or {}).get("summary") if isinstance(target_script, dict) else None,
                "first_lines": _extract_first_lines(target_script, n=after_n),
            }

        current_scene_script["choices"] = [
            {
                "id": f"choice_branch_{scene_id}",
                "appears_after_line": max(0, len(current_scene_script.get("lines") or []) - 1),
                "options": options,
            }
        ]
        patched_scenes += 1

    return patched_scenes


async def patch_choices_payload(
    data: Dict[str, Any],
    *,
    use_llm: bool = True,
    fake_every: int = 0,  #не используется
    client: Optional[OpenRouterClient] = None,
    model: Optional[str] = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:

    _ = fake_every

    beats_by_id, scenes_by_id, contracts_by_branch, branch_first_scene, branches_by_scene = _build_maps(data)

    model_name = model or os.getenv("CHOICE_PATCH_MODEL", "google/gemini-2.5-flash")

    use_llm_effective = bool(use_llm and client is not None)

    removed_fake = _strip_fake_options_everywhere(data)

    patched = await _rewrite_branch_choices(
        data,
        scenes_by_id,
        contracts_by_branch,
        branch_first_scene,
        branches_by_scene,
        use_llm=use_llm_effective,
        client=client,
        model=model_name,
    )

    removed_fake += _strip_fake_options_everywhere(data)

    data.setdefault("postprocess", {})
    data["postprocess"]["choices_patched"] = True
    data["postprocess"]["choices_patch_mode"] = "llm" if use_llm_effective else "heuristic"
    data["postprocess"]["choice_patch_model"] = model_name
    data["postprocess"]["fake_buttons"] = "disabled"
    data["postprocess"]["fake_options_removed"] = int(removed_fake)
    data["postprocess"]["branch_choice_scenes_patched"] = int(patched)
    data["postprocess"]["rng_seed"] = int(rng_seed)
    data["postprocess"]["context_before_lines"] = _env_int("CHOICE_CONTEXT_BEFORE_LINES", 2)
    data["postprocess"]["context_after_lines"] = _env_int("CHOICE_CONTEXT_AFTER_LINES", 2)

    return data


async def patch_choices(
    input_json: Path,
    *,
    output_json: Optional[Path],
    use_llm: bool,
    fake_every: int,
) -> Path:
    """
    File-based wrapper (kept for backwards compatibility with your CLI),
    but fake_every is now deprecated because we never inject fake choices.
    """
    data = _load_json(input_json)

    client: Optional[OpenRouterClient] = None
    model = os.getenv("CHOICE_PATCH_MODEL", "google/gemini-2.5-flash")
    rng_seed = _env_int("CHOICE_PATCH_RNG_SEED", 42)

    try:
        if use_llm:
            load_dotenv()
            api_key = os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
            base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

            if "openrouter.ai" in base_url and not api_key:
                app_logger.warning("API key not found; falling back to heuristic mode")
                use_llm = False
            else:
                client = OpenRouterClient(api_key=api_key, base_url=base_url)

        data = await patch_choices_payload(
            data,
            use_llm=use_llm,
            fake_every=fake_every,
            client=client if use_llm else None,
            model=model,
            rng_seed=rng_seed,
        )

        out_path = output_json
        if out_path is None:
            if input_json.stem == "final":
                out_path = input_json.with_name("final_choices_patched.json")
            else:
                out_path = input_json.with_name(input_json.stem + "_choices_patched.json")

        _save_json(out_path, data)
        return out_path

    finally:
        if client is not None:
            await client.close()


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Patch VN json with real (non-fake) branch choices + POV context")
    parser.add_argument("input_json", help="Path to final.json or exported generation json")
    parser.add_argument("--output", help="Path to save patched json", default=None)
    parser.add_argument("--no-llm", action="store_true", help="Do not call LLM, use heuristic labels only")
    parser.add_argument(
        "--fake-every",
        type=int,
        default=0,
        help="DEPRECATED. Fake choices are no longer inserted (kept for backward compatibility).",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out = await patch_choices(
        input_path,
        output_json=Path(args.output) if args.output else None,
        use_llm=not args.no_llm,
        fake_every=max(0, int(args.fake_every)),
    )
    app_logger.info(f"Patched file saved to: {out}")


if __name__ == "__main__":
    asyncio.run(_amain())