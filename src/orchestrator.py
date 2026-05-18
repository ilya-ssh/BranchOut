from __future__ import annotations
from src.postprocess import patch_choices_payload
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
import json
from json import JSONDecodeError
import os
import math
import base64
import re
import inspect
from pathlib import Path
import random
from collections import Counter, defaultdict, deque
import hashlib
import uuid

import httpx
from pydantic import ValidationError

from src.build import export_web_from_json
from src.agents import Agent
from src.toolbox import Toolbox
from src.apicallhandler import OpenRouterClient, TRACE_HOOK
from src.router import ModelRouter
from src.pydantic_schemas import (
    Setting,
    StoryOutlineFull,
    OutlineBeat,
    SceneContract,
    SceneScript,
    SceneChoice,
    SceneChoiceOption,
    CharacterAppearance,
    LocationDescription,
    UserRequest,
    CharacterImage,
    LocationImage,
    BranchSpec,
    BranchingInfo,
    PlotPreferences,
    StoryState,
    SceneLine,
    LocationAffordance,
    ChoiceContract,
    ChoiceOptionPlan,
)
from src.logger import app_logger
from src.utils.artifacts import ArtifactStore
from src.utils.names import NameCanonicalizer
from src.prompts import (
    contract_location_critic_prompt,
    branch_tail_rewriter_prompt,
    contract_consistency_critic_prompt,
    char_prompt,
    loc_prompt,
    setting_prompt,
    outline_prompt,
    scene_plan_prompt,
    writer_prompt,
    rag_context_prompt,
    critic_prompt,
    user_request_prompt,
    character_critic_prompt,
    branch_planner_prompt,
    choice_planner_prompt,
    plot_thread_extractor_prompt,
    scene_microplanner_prompt,
    scene_editor_prompt,
    scene_contract_reenricher_prompt,
    location_affordance_prompt,
    contract_location_critic_prompt,
)


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class RAGIndex:
    def __init__(self, client: OpenRouterClient, embed_model: Optional[str] = None):
        self.client = client
        self.embed_model = embed_model
        self.items: List[Dict[str, Any]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._doc_len_sum: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[0-9A-Za-zА-Яа-яЁё_]+", text.lower())
        return [t for t in toks if len(t) > 1]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    def _bm25_score(self, q_tokens: List[str], item: Dict[str, Any], k1: float = 1.2, b: float = 0.75) -> float:
        N = len(self.items)
        if N <= 0:
            return 0.0
        avgdl = (self._doc_len_sum / N) if N else 1.0
        if avgdl <= 0.0:
            avgdl = 1.0

        tf: Counter = item.get("tf") or Counter()
        dl: int = len(item.get("tokens") or [])
        if dl <= 0:
            dl = 1

        score = 0.0
        for t in q_tokens:
            df = self._df.get(t, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            f = tf.get(t, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * (f * (k1 + 1.0) / denom)
        return float(score)



    def _remove_stats(self, tokens: List[str]) -> None:
        uniq = set(tokens)
        for t in uniq:
            self._df[t] = max(0, int(self._df.get(t, 0)) - 1)
        self._doc_len_sum = max(0, self._doc_len_sum - len(tokens))

    def _add_stats(self, tokens: List[str]) -> None:
        uniq = set(tokens)
        for t in uniq:
            self._df[t] += 1
        self._doc_len_sum += len(tokens)

    async def upsert_item(self, item_id: str, kind: str, text: str) -> None:
        if not text or not text.strip():
            return

        for idx, it in enumerate(list(self.items)):
            if it.get("id") == item_id and it.get("kind") == kind:
                self._remove_stats(it.get("tokens") or [])
                self.items.pop(idx)
                break

        tokens = self._tokenize(text)
        tf = Counter(tokens)

        embedding: List[float] = []
        if self.embed_model:
            embedding = await self.client.generate_embedding(
                text=text,
                model=self.embed_model,
                operation_name=f"embed_{kind}_{item_id}",
            )

        self._add_stats(tokens)
        self.items.append(
            {
                "id": item_id,
                "kind": kind,
                "text": text,
                "embedding": embedding,
                "tokens": tokens,
                "tf": tf,
            }
        )

    async def add_item(self, item_id: str, kind: str, text: str) -> None:
        await self.upsert_item(item_id, kind, text)

    def import_items(self, items: List[Dict[str, Any]], kind_filter: Optional[str] = None) -> None:
        for it in items:
            if not isinstance(it, dict):
                continue
            if kind_filter and it.get("kind") != kind_filter:
                continue
            item_id = it.get("id")
            kind = it.get("kind")
            text = it.get("text")
            if not item_id or not kind or not text:
                continue

            if any(x.get("id") == item_id and x.get("kind") == kind for x in self.items):
                continue

            tokens = it.get("tokens") or self._tokenize(text)
            tf = it.get("tf") or Counter(tokens)
            emb = it.get("embedding") or []

            self._add_stats(tokens)
            self.items.append(
                {
                    "id": item_id,
                    "kind": kind,
                    "text": text,
                    "embedding": emb,
                    "tokens": tokens,
                    "tf": tf,
                }
            )

    def clone(self) -> "RAGIndex":
        other = RAGIndex(self.client, self.embed_model)
        other.import_items(list(self.items))
        return other

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        kinds: Optional[List[str]] = None,
        alpha: float = 0.7,
        q_tokens: Optional[List[str]] = None,
        q_emb: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.items or not (query_text or "").strip():
            return []

        q_tokens = q_tokens or self._tokenize(query_text)

        lex: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.items:
            if kinds and item.get("kind") not in kinds:
                continue
            bm_raw = self._bm25_score(q_tokens, item)
            lex.append((bm_raw, item))

        max_bm = max((s for s, _ in lex), default=0.0)
        if max_bm <= 0.0:
            max_bm = 1.0

        use_emb = bool(self.embed_model)
        if use_emb and q_emb is None:
            q_emb = await self.client.generate_embedding(
                text=query_text,
                model=self.embed_model,
                operation_name="embed_query",
            )

        scored: List[Tuple[float, Dict[str, Any], float, float, float]] = []
        for bm_raw, item in lex:
            bm_norm = float(bm_raw / max_bm)
            cos = 0.0
            if use_emb and q_emb is not None:
                cos = self._cosine(q_emb, item.get("embedding") or [])
            score = alpha * cos + (1.0 - alpha) * bm_norm
            scored.append((score, item, cos, bm_raw, bm_norm))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        return [
            {
                "id": it["id"],
                "kind": it["kind"],
                "text": it["text"],
                "score": float(score),
                "embedding_score": float(cos),
                "lexical_score_raw": float(bm_raw),
                "lexical_score_norm": float(bm_norm),
            }
            for score, it, cos, bm_raw, bm_norm in top
        ]


class RAGBundle:
    def __init__(
        self,
        client: OpenRouterClient,
        embed_model: Optional[str],
        world_index: Optional[RAGIndex] = None,
        char_index: Optional[RAGIndex] = None,
        thread_index: Optional[RAGIndex] = None,
    ):
        self.client = client
        self.embed_model = embed_model
        self.story = RAGIndex(client, embed_model)
        self.world = world_index or RAGIndex(client, embed_model)
        self.characters = char_index or RAGIndex(client, embed_model)
        self.threads = thread_index or RAGIndex(client, embed_model)


class VNOrchestrator:
    def __init__(self, client: OpenRouterClient, router: ModelRouter):
        self.client = client
        self.router = router

    @staticmethod
    def _strict_name_canon_enabled() -> bool:
        val = str(os.getenv("STRICT_NAME_CANON", "true")).strip().lower()
        return val in ("1", "true", "yes", "on")

    @staticmethod
    def _strict_location_gate_enabled() -> bool:
        val = str(os.getenv("STRICT_LOCATION_GATE", "true")).strip().lower()
        return val in ("1", "true", "yes", "on")

    @staticmethod
    def _strict_thread_closure_enabled() -> bool:
        val = str(os.getenv("STRICT_THREAD_CLOSURE", "false")).strip().lower()
        return val in ("1", "true", "yes", "on")

    @staticmethod
    def _story_checkpoint_every() -> int:
        env = os.getenv("STORY_CHECKPOINT_EVERY")
        if env:
            try:
                return max(2, min(int(env), 10))
            except Exception:
                pass
        return 4

    @staticmethod
    def _unwrap_last(v: Any) -> Any:
        return v[-1] if isinstance(v, list) and v else v

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in items or []:
            s = str(x or "").strip()
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out
    @staticmethod
    def _norm_text_list(value: Any, *, max_items: int = 8, max_len: int = 240) -> List[str]:
        if not isinstance(value, list):
            return []

        out: List[str] = []
        seen = set()

        for item in value:
            s = str(item or "").strip()
            if not s:
                continue

            if len(s) > max_len:
                s = s[:max_len].rsplit(" ", 1)[0].strip() or s[:max_len]

            key = s.casefold()
            if key in seen:
                continue

            seen.add(key)
            out.append(s)

            if len(out) >= max_items:
                break

        return out

    @staticmethod
    def _clip_text(value: Any, max_len: int = 280) -> str:
        s = str(value or "").strip()
        if len(s) > max_len:
            s = s[:max_len].rsplit(" ", 1)[0].strip() or s[:max_len]
        return s

    def _sanitize_contract_rich_fields(
        self,
        contract: SceneContract,
        *,
        known_thread_ids: Optional[set[str]] = None,
    ) -> None:
        contract.scene_goal = self._clip_text(contract.scene_goal, 240)
        contract.scene_conflict = self._clip_text(contract.scene_conflict, 260)
        contract.stakes = self._clip_text(contract.stakes, 260)
        contract.reveal = self._clip_text(contract.reveal, 260)
        contract.emotional_beat = self._clip_text(contract.emotional_beat, 180)

        contract.must_reference = self._norm_text_list(contract.must_reference, max_items=8, max_len=140)
        contract.entry_requirements = self._norm_text_list(contract.entry_requirements, max_items=8, max_len=180)
        contract.exit_targets = self._norm_text_list(contract.exit_targets, max_items=8, max_len=180)
        contract.continuity_notes = self._norm_text_list(contract.continuity_notes, max_items=8, max_len=180)

        tf = self._norm_text_list(contract.thread_focus, max_items=6, max_len=80)
        if known_thread_ids is not None:
            tf = [x for x in tf if x in known_thread_ids]
        contract.thread_focus = tf

    def _apply_rich_contract_patch(
        self,
        contract: SceneContract,
        patch: Dict[str, Any],
        *,
        known_thread_ids: Optional[set[str]] = None,
    ) -> bool:
        changed = False

        str_fields = [
            ("scene_goal", "new_scene_goal", 240),
            ("scene_conflict", "new_scene_conflict", 260),
            ("stakes", "new_stakes", 260),
            ("reveal", "new_reveal", 260),
            ("emotional_beat", "new_emotional_beat", 180),
        ]

        for attr, key, max_len in str_fields:
            if key not in patch or patch.get(key) is None:
                continue
            new_val = self._clip_text(patch.get(key), max_len=max_len)
            if new_val != getattr(contract, attr):
                setattr(contract, attr, new_val)
                changed = True

        list_fields = [
            ("must_reference", "new_must_reference", 8, 140, False),
            ("entry_requirements", "new_entry_requirements", 8, 180, False),
            ("exit_targets", "new_exit_targets", 8, 180, False),
            ("continuity_notes", "new_continuity_notes", 8, 180, False),
            ("thread_focus", "new_thread_focus", 6, 80, True),
        ]

        for attr, key, max_items, max_len, filter_threads in list_fields:
            if key not in patch or patch.get(key) is None:
                continue

            raw_val = patch.get(key)
            vals = self._norm_text_list(raw_val, max_items=max_items, max_len=max_len) if isinstance(raw_val, list) else []

            if filter_threads and known_thread_ids is not None:
                vals = [x for x in vals if x in known_thread_ids]

            if vals != getattr(contract, attr):
                setattr(contract, attr, vals)
                changed = True

        self._sanitize_contract_rich_fields(contract, known_thread_ids=known_thread_ids)
        return changed
    @staticmethod
    def _normalize_char_type(
        char_type: Any,
        char_list: List[str],
        *,
        canon: Optional[NameCanonicalizer] = None,
        protagonist: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        roles = [
            "Протагонист",
            "Искомый персонаж",
            "Антагонист",
            "Даритель",
            "Помощник",
            "Отправитель",
            "Ложный герой",
        ]

        out: Dict[str, List[str]] = {r: [] for r in roles}
        if not char_list:
            return out

        source: Dict[str, Any] = {}
        if isinstance(char_type, dict):
            nested = char_type.get("char_type")
            if isinstance(nested, dict):
                source = nested
            else:
                source = char_type

        used = set()

        def canon_name(x: Any) -> Optional[str]:
            s = str(x or "").strip()
            if not s:
                return None
            if s in char_list:
                return s
            if canon is not None:
                rr = canon.canonicalize_character(s, fallback=None)
                if rr.output and rr.output in char_list:
                    return rr.output
            return None

        forced_protagonist = protagonist if protagonist in char_list else char_list[0]
        out["Протагонист"] = [forced_protagonist]
        used.add(forced_protagonist)

        for role in roles:
            if role == "Протагонист":
                continue

            vals = source.get(role, [])
            if isinstance(vals, str):
                vals = [vals]
            if not isinstance(vals, list):
                vals = []

            for v in vals:
                nm = canon_name(v)
                if not nm or nm in used:
                    continue
                out[role].append(nm)
                used.add(nm)

        if not out["Искомый персонаж"]:
            candidate = next(
                (c for c in char_list if c != forced_protagonist and c not in used),
                None,
            )
            if candidate is None:
                candidate = next(
                    (c for c in char_list if c != forced_protagonist),
                    None,
                )

            if candidate:
                for role in roles:
                    if candidate in out[role]:
                        out[role] = [x for x in out[role] if x != candidate]
                out["Искомый персонаж"] = [candidate]
                used.add(candidate)

        for name in char_list:
            if name not in used:
                out["Помощник"].append(name)
                used.add(name)

        return out

    @staticmethod
    def _char_type_role_map(char_type: Optional[Dict[str, Any]]) -> Dict[str, str]:
        role_by_char: Dict[str, str] = {}
        if not isinstance(char_type, dict):
            return role_by_char

        for role, names in char_type.items():
            if isinstance(names, list):
                for name in names:
                    s = str(name or "").strip()
                    if s:
                        role_by_char[s] = str(role)
            elif isinstance(names, str):
                s = names.strip()
                if s:
                    role_by_char[s] = str(role)

        return role_by_char

    @staticmethod
    def _present_char_roles(
        char_type: Optional[Dict[str, Any]],
        present_characters: List[str],
    ) -> Dict[str, str]:
        role_map = VNOrchestrator._char_type_role_map(char_type)
        return {name: role_map[name] for name in present_characters if name in role_map}

    @staticmethod
    def _obj_to_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        return {}

    async def _emit_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]],
        *,
        generation_id: str,
        stage: str,
        message: str,
        percent: int,
        status: str = "running",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if progress_callback is None:
            return

        payload = {
            "generation_id": generation_id,
            "status": status,
            "stage": stage,
            "message": message,
            "percent": max(0, min(int(percent), 100)),
            "extra": extra or {},
        }

        try:
            maybe = progress_callback(payload)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception as e:
            app_logger.warning(f"Progress callback failed: {e}")

    @staticmethod
    def _normalize_purpose_value(purpose: str) -> str:
        p = (purpose or "").strip().casefold()

        mapping = {
            "introduction": "introduction",
            "setup": "setup",
            "inciting_incident": "inciting_incident",
            "turning_point": "turning_point",
            "reaction": "reaction",
            "sequel": "sequel",
            "rising_action": "rising_action",
            "conflict": "conflict",
            "midpoint": "midpoint",
            "crisis": "crisis",
            "climax": "climax",
            "resolution": "resolution",
            "ending": "resolution",
            "epilogue": "epilogue",
            "travel": "travel",
            "revelation": "revelation",
            "transformation": "epilogue",

            "переход": "travel",
            "дорога": "travel",
            "путь": "travel",

            "кульминация": "climax",
            "финал": "resolution",
            "развязка": "resolution",
            "эпилог": "epilogue",
            "начальная ситуация": "introduction",
            "запрет": "setup",
            "нарушение запрета, наказания или приказа или альтернатива исполнение приказа": "inciting_incident",
            "нарушение запрета": "inciting_incident",
            "введение антагониста": "setup",
            "антагонист предпринимает действие в отношении героя - насилие, обман, санкции": "conflict",
            "антагонист предпринимает действие в отношении героя": "conflict",
            "беда - кризисная ситуация, которая требует решения": "crisis",
            "беда": "crisis",
            "герой узнает о беде и начинает ей противодействовать": "turning_point",
            "герой уходит из дома/безопасного места": "turning_point",
            "герой встречает персонажа-дарителя - он может быть союзником или наоборот врагом, но он обладает артефактом, который может помочь главному герою": "setup",
            "герой встречает персонажа-дарителя": "setup",
            "герой получает от персонажа дарителя артефакт: герой может его добыть, украсть, получить любым другим способом. артефакт может быть знанием, предметом, человеком, помощником.": "revelation",
            "герой получает от персонажа дарителя артефакт": "revelation",
            "главный герой перемещается между локациями для достижения цели": "travel",
            "победа над антагонистом - главный герой каким-то способом побеждает главного антагониста: хитрость, физическая сила, магия, воля случая": "climax",
            "победа над антагонистом": "climax",
            "ликвидация беды - изначальная кризисная ситуация разрешается": "resolution",
            "ликвидация беды": "resolution",
            "возвращение героя": "resolution",
            "преследование героя": "conflict",
            "спасение героя": "sequel",
            "неузнанное прибытие персонажа": "travel",
            "трудная задача - появляется трудная задача, которая обязательно требует решения": "crisis",
            "трудная задача": "crisis",
            "решение задачи - ранее появившаяся трудная задача решена": "resolution",
            "решение задачи": "resolution",
            "узнавание героя": "revelation",
            "появление ложного героя под прикрытием - появление персонажа в команде главного героя, который на самом деле является его врагом.": "conflict",
            "появление ложного героя под прикрытием": "conflict",
            "обличение ложного героя - раскрытие личности и намерений ложного героя.": "climax",
            "обличение ложного героя": "climax",
            "трансфигурация персонажа - новая одежда, изменение внешности, социального статуса достатка или другое кардинальное изменение персонажа": "epilogue",
            "трансфигурация персонажа": "epilogue",
        }

        return mapping.get(p, p or "setup")

    def _max_locations_for_length(self, story_length: str) -> int:
        sl = (story_length or "medium").strip().lower()
        key = f"MAX_LOCATIONS_{sl.upper()}"
        env = os.getenv(key) or os.getenv("MAX_LOCATIONS")
        if env:
            try:
                v = int(env)
                return max(4, min(v, 40))
            except Exception:
                pass
        return {"short": 8, "medium": 12, "long": 16}.get(sl, 12)

    def _limit_locations(
        self,
        *,
        candidates: List[str],
        user_prompt: str,
        setting: Setting,
        outline: StoryOutlineFull,
        story_length: str,
        artifact_store: Optional[ArtifactStore] = None,
        reason: str = "auto_loc_list",
    ) -> List[str]:
        max_locs = self._max_locations_for_length(story_length)
        uniq = self._dedupe_preserve_order([str(x) for x in (candidates or [])])
        if len(uniq) <= max_locs:
            return uniq

        corpus_parts: List[str] = []
        corpus_parts.append(user_prompt or "")
        corpus_parts.append(setting.setting or "")
        corpus_parts.append(setting.world_rules or "")
        for b in (outline.beats or []):
            corpus_parts.append(b.title or "")
            corpus_parts.append(b.summary or "")
        corpus = "\n".join(corpus_parts).casefold()

        scored: List[Tuple[int, int, str]] = []
        for i, name in enumerate(uniq):
            n = name.casefold()
            score = 0
            if n:
                score += corpus.count(n)
                toks = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", n)
                for t in toks:
                    if len(t) >= 4:
                        score += 1 if t in corpus else 0
            scored.append((score, -i, name))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = {name for _, _, name in scored[:max_locs]}
        limited = [x for x in uniq if x in selected]

        if artifact_store is not None:
            artifact_store.event(
                "loc_list.limited",
                {
                    "reason": reason,
                    "max_locations": max_locs,
                    "before": len(uniq),
                    "after": len(limited),
                    "dropped": [x for x in uniq if x not in selected][:80],
                },
            )
            artifact_store.save(
                "checkpoints/loc_list_limited_preview.json",
                {"max_locations": max_locs, "limited": limited, "original": uniq[:200]},
            )

        return limited

    def _normalize_outline_order(
        self,
        outline: StoryOutlineFull,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> StoryOutlineFull:
        beats = list(outline.beats or [])
        if not beats:
            return outline

        for b in beats:
            b.purpose = self._normalize_purpose_value(b.purpose)

        orders = [b.order for b in beats]
        strictly_increasing = all(orders[i] < orders[i + 1] for i in range(len(orders) - 1))
        unique = len(set(orders)) == len(orders)

        if strictly_increasing and unique:
            outline.beats = sorted(beats, key=lambda b: b.order)
            return outline

        beats_sorted = sorted(beats, key=lambda b: (b.act, b.order, b.id))
        for idx, b in enumerate(beats_sorted, start=1):
            b.order = idx

        outline.beats = beats_sorted

        if artifact_store is not None:
            artifact_store.event(
                "outline.order_repaired",
                {
                    "reason": "non_unique_or_non_monotonic_order",
                    "original_orders_preview": orders[:60],
                    "new_orders_preview": [b.order for b in beats_sorted[:60]],
                },
            )

        return outline

    def _canon_contract_inplace(
        self,
        contract: SceneContract,
        canon: NameCanonicalizer,
        *,
        store: Optional[ArtifactStore] = None,
    ) -> None:
        loc_fb = canon.locations[0] if canon.locations else None
        rloc = canon.canonicalize_location(contract.location, fallback=loc_fb)
        if rloc.output:
            contract.location = rloc.output
        if store is not None and rloc.status in ("unknown", "fallback"):
            store.event(
                "name.location_canon",
                {
                    "scene_id": contract.id,
                    "input": rloc.input,
                    "output": rloc.output,
                    "status": rloc.status,
                    "detail": rloc.detail,
                },
            )

        pov_fb = canon.characters[0] if canon.characters else None
        rpov = canon.canonicalize_character(contract.pov_character, fallback=pov_fb)
        if rpov.output:
            contract.pov_character = rpov.output
        if store is not None and rpov.status in ("unknown", "fallback"):
            store.event(
                "name.pov_canon",
                {
                    "scene_id": contract.id,
                    "input": rpov.input,
                    "output": rpov.output,
                    "status": rpov.status,
                    "detail": rpov.detail,
                },
            )

        fixed: List[str] = []
        seen = set()
        for x in contract.present_characters or []:
            rr = canon.canonicalize_character(x, fallback=None)
            if rr.output is None:
                if store is not None:
                    store.event(
                        "name.present_drop_unknown",
                        {"scene_id": contract.id, "input": rr.input, "status": rr.status, "detail": rr.detail},
                    )
                continue
            if rr.output in seen:
                continue
            seen.add(rr.output)
            fixed.append(rr.output)

        if contract.pov_character and contract.pov_character not in seen:
            fixed.insert(0, contract.pov_character)

        contract.present_characters = fixed

    def _canon_script_inplace(
        self,
        script: SceneScript,
        contract: SceneContract,
        canon: NameCanonicalizer,
        *,
        store: Optional[ArtifactStore] = None,
    ) -> bool:
        had_unmatched = False

        pov = contract.pov_character
        present = contract.present_characters or []
        pov_fb = pov if pov else (canon.characters[0] if canon.characters else None)

        for i, line in enumerate(script.lines or []):
            if line.type == "narration":
                line.speaker = None
                continue

            if line.type == "thought":
                rr0 = canon.canonicalize_character(line.speaker or pov, fallback=None)
                if rr0.output is None:
                    had_unmatched = True
                rr = canon.canonicalize_character(line.speaker or pov, fallback=pov_fb)
                line.speaker = rr.output
                if store is not None and rr0.output is None:
                    store.event(
                        "name.thought_speaker_unmatched",
                        {
                            "scene_id": script.scene_id,
                            "line": i,
                            "input": rr0.input,
                            "output": rr.output,
                            "status": rr.status,
                            "detail": rr.detail,
                        },
                    )
                continue

            rr0 = canon.canonicalize_character(line.speaker, fallback=None)
            if rr0.output is not None:
                line.speaker = rr0.output
                continue

            had_unmatched = True
            if store is not None:
                store.event(
                    "name.dialogue_speaker_unmatched",
                    {
                        "scene_id": script.scene_id,
                        "line": i,
                        "input": rr0.input,
                        "output": None,
                        "status": rr0.status,
                        "detail": rr0.detail,
                    },
                )
            continue

        return had_unmatched

    async def _parse_json_with_repair(
        self,
        raw: str,
        model_name: str,
        operation_name: str,
        schema_hint: str,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Any]:
        raw_strip = (raw or "").strip()

        try:
            return json.loads(raw_strip)
        except JSONDecodeError:
            pass

        start = raw_strip.find("{")
        end = raw_strip.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw_strip[start : end + 1]
            try:
                return json.loads(candidate)
            except JSONDecodeError:
                app_logger.warning(f"{operation_name}: parse failed, invoking repair LLM...")
                if artifact_store is not None:
                    artifact_store.event(
                        "parse.repair_called",
                        {"operation_name": operation_name, "model": model_name, "raw_preview": raw_strip[:800]},
                    )

        repair_system = (
            "Ты помощник, который ЧИНИТ НЕВАЛИДНЫЙ JSON.\n"
            "Верни СТРОГО ОДИН ВАЛИДНЫЙ JSON-ОБЪЕКТ по schema_hint.\n"
            "Никакого Markdown, только JSON."
        )
        repair_payload = {"broken_json": raw_strip, "schema_hint": schema_hint}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.0,
            system_prompt=repair_system,
            prompt=json.dumps(repair_payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"{operation_name}_json_repair",
        )

        fixed = (resp["choices"][0]["message"]["content"] or "").strip()
        try:
            return json.loads(fixed)
        except JSONDecodeError:
            start = fixed.find("{")
            end = fixed.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(fixed[start : end + 1])
                except JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _build_loc_desc_map(loc_list: List[str], loc_description: Any) -> Dict[str, str]:
        descs: List[str] = []
        if isinstance(loc_description, dict):
            descs = loc_description.get("descriptions") or []
        else:
            descs = getattr(loc_description, "descriptions", None) or []
        out: Dict[str, str] = {}
        for name, desc in zip(loc_list or [], descs or []):
            out[str(name)] = str(desc or "")
        for name in loc_list or []:
            out.setdefault(str(name), "")
        return out

    async def _infer_location_affordances(
        self,
        setting: Setting,
        loc_list: List[str],
        loc_canons: Dict[str, str],
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Dict[str, Any]]:
        app_logger.info("Inferring location affordances (enterable/scale/kind)...")
        model_name = self.router.get_model_for_agent("outline_agent")

        payload = {
            "setting": setting.dict(),
            "loc_list": loc_list,
            "loc_canons": {k: (v[:1200] if isinstance(v, str) else "") for k, v in (loc_canons or {}).items()},
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=location_affordance_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="location_affordances",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"locations":[{"location":"...","kind":"outdoor","enterable":false,"scale":"object","notes":"..."}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "location_affordances_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        items = data.get("locations") or []
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                loc = str(it.get("location") or "").strip()
                if not loc:
                    continue
                if loc not in loc_list:
                    continue
                try:
                    aff = LocationAffordance(
                        location=loc,
                        kind=str(it.get("kind") or "mixed"),
                        enterable=bool(it.get("enterable", True)),
                        scale=str(it.get("scale") or "area"),
                        notes=str(it.get("notes") or ""),
                    )
                    out[loc] = aff.dict()
                except Exception:
                    out[loc] = {
                        "kind": str(it.get("kind") or "mixed"),
                        "enterable": bool(it.get("enterable", True)),
                        "scale": str(it.get("scale") or "area"),
                        "notes": str(it.get("notes") or ""),
                    }

        for loc in loc_list:
            out.setdefault(loc, {"kind": "mixed", "enterable": True, "scale": "area", "notes": ""})

        if artifact_store is not None:
            artifact_store.checkpoint("07b_location_affordances", {"affordances": out})
        return out

    async def _patch_scene_contracts_with_location_critic(
            self,
            scene_contracts: List[SceneContract],
            loc_list: List[str],
            loc_canons: Dict[str, str],
            loc_affordances: Dict[str, Dict[str, Any]],
            loc_graph: Optional[Any],
            plot_threads: Optional[List[Dict[str, Any]]] = None,
            artifact_store: Optional[ArtifactStore] = None,
            artifact_name: str = "contracts_patch",
    ) -> List[SceneContract]:
        if not scene_contracts:
            return scene_contracts

        app_logger.info("Running contract_location_critic (patch contracts for location consistency)...")
        model_name = self.router.get_model_for_agent("outline_agent")
        known_thread_ids = {
            str(t.get("id"))
            for t in (plot_threads or [])
            if isinstance(t, dict) and str(t.get("id") or "").strip()
        }

        payload = {
            "loc_list": loc_list,
            "loc_canons": {k: (v[:1200] if isinstance(v, str) else "") for k, v in (loc_canons or {}).items()},
            "loc_affordances": loc_affordances,
            "loc_graph": loc_graph,
            "plot_threads": plot_threads or [],
            "scene_contracts": [c.dict() for c in scene_contracts],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=contract_location_critic_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"contract_location_critic_{artifact_name}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = (
            '{"patches":[{"scene_id":"scene_010","new_location":null,"new_summary":null,'
            '"new_scene_goal":null,"new_scene_conflict":null,"new_stakes":null,"new_reveal":null,'
            '"new_emotional_beat":null,"new_must_reference":null,"new_entry_requirements":null,'
            '"new_exit_targets":null,"new_continuity_notes":null,"new_thread_focus":null,'
            '"reason":"..."}]}'
        )
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"contract_location_critic_{artifact_name}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        patches = data.get("patches") or []
        if artifact_store is not None:
            artifact_store.save(f"contracts/{artifact_name}.json", {"patches": patches})

        if not isinstance(patches, list) or not patches:
            for c in scene_contracts:
                self._sanitize_contract_rich_fields(c, known_thread_ids=known_thread_ids)
            return scene_contracts

        by_id: Dict[str, SceneContract] = {c.id: c for c in scene_contracts}
        applied: List[Dict[str, Any]] = []

        for p in patches[:200]:
            if not isinstance(p, dict):
                continue
            sid = str(p.get("scene_id") or "")
            if not sid or sid not in by_id:
                continue

            c = by_id[sid]
            changed = False

            new_loc = p.get("new_location")
            if isinstance(new_loc, str):
                nl = new_loc.strip()
                if nl and nl in loc_list and nl != c.location:
                    c.location = nl
                    changed = True

            new_sum = p.get("new_summary")
            if isinstance(new_sum, str):
                ns = new_sum.strip()
                if ns and ns != c.summary:
                    c.summary = ns
                    changed = True

            rich_changed = self._apply_rich_contract_patch(
                c,
                p,
                known_thread_ids=known_thread_ids,
            )
            changed = changed or rich_changed

            if changed:
                applied.append(
                    {
                        "scene_id": sid,
                        "new_location": c.location,
                        "new_summary": c.summary,
                        "rich_fields_patched": bool(rich_changed),
                        "reason": p.get("reason"),
                    }
                )

        for c in by_id.values():
            self._sanitize_contract_rich_fields(c, known_thread_ids=known_thread_ids)

        if artifact_store is not None and applied:
            artifact_store.save(f"contracts/{artifact_name}_applied.json", {"applied": applied})

        return list(by_id.values())

    async def _normalize_user_request(
        self,
        raw_prompt: str,
        explicit_length: Optional[str] = None,
        explicit_tone: Optional[str] = None,
        explicit_artstyle: Optional[str] = None,
        explicit_max_branches: Optional[int] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> UserRequest:
        app_logger.info("Normalizing user prompt into UserRequest...")
        model_name = self.router.get_model_for_agent("setting_agent")

        payload = {"user_prompt": raw_prompt}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=user_request_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="user_request_normalizer",
        )

        content = (resp["choices"][0]["message"]["content"] or "").strip()
        data = json.loads(content) if content else {}
        data["user_prompt"] = raw_prompt

        ur = UserRequest(**data)
        if explicit_length:
            ur.story_length = explicit_length
        if explicit_tone:
            ur.tone = explicit_tone
        if explicit_artstyle:
            ur.general_artstyle = explicit_artstyle
        if explicit_max_branches is not None:
            ur.max_branches = explicit_max_branches

        if artifact_store is not None:
            artifact_store.checkpoint("01_user_request_normalized", ur.dict())
        return ur

    async def _generate_setting(
            self,
            user_prompt: str,
            setting_override: str | None = None,
            time_choice: Optional[str] = None,
            genre_choice: Optional[str] = None,
            artifact_store: Optional[ArtifactStore] = None,
    ) -> Setting:
        app_logger.info("Generating setting...")
        model_name = self.router.get_model_for_agent("setting_agent")

        payload: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "setting_override": setting_override or "",
            "time_choice": time_choice,
            "genre_choice": genre_choice,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.4,
            system_prompt=setting_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="setting_from_prompt",
        )

        raw = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        raw = raw.strip()

        schema_hint = '{"setting":"...","genre":"...","time_period":"...","world_rules":"..."}'
        data = await self._parse_json_with_repair(
            raw=raw,
            model_name=model_name,
            operation_name="setting_from_prompt_parse",
            schema_hint=schema_hint,
            artifact_store=artifact_store,
        )

        if not isinstance(data, dict) or not data:
            raise ValueError("Failed to generate setting: empty/invalid JSON")

        setting = Setting(**data)

        if time_choice:
            setting.time_period = time_choice
        if genre_choice:
            setting.genre = genre_choice

        if artifact_store is not None:
            artifact_store.checkpoint("02_setting", setting.dict())
        return setting

    def _ensure_beat_coverage_contracts(
        self,
        *,
        beats: List[OutlineBeat],
        contracts: List[SceneContract],
        branch_id: str,
        id_prefix: str,
        default_loc: str,
        default_pov: str,
    ) -> List[SceneContract]:
        beats_in_order = [b.id for b in sorted(beats, key=lambda x: x.order)]
        by_beat: Dict[str, List[SceneContract]] = {}
        for c in contracts:
            by_beat.setdefault(c.beat_id, []).append(c)

        out: List[SceneContract] = []
        for bid in beats_in_order:
            existing = by_beat.get(bid) or []
            if existing:
                out.extend(sorted(existing, key=lambda x: x.branch_order))
            else:
                beat_obj = next((b for b in beats if b.id == bid), None)
                out.append(
                    SceneContract(
                        id="__TMP__",
                        beat_id=bid,
                        location=default_loc,
                        pov_character=default_pov,
                        present_characters=[default_pov],
                        summary=(beat_obj.summary if beat_obj else ""),
                        branch_id=branch_id,
                        branch_order=0,
                    )
                )

        for i, c in enumerate(out, start=1):
            c.branch_order = i
            c.id = f"{id_prefix}{i:03d}"

        return out

    async def _generate_outline(
        self,
        user_prompt: str,
        story_length: str,
        setting: Setting,
        plot_prefs: Optional[PlotPreferences] = None,
        plot_freeform: Optional[str] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> StoryOutlineFull:
        app_logger.info("Generating outline...")
        model_name = self.router.get_model_for_agent("outline_agent")

        payload: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "story_length": story_length,
            "setting": setting.dict(),
            "plot_prefs": plot_prefs.dict() if plot_prefs else None,
            "plot_freeform": plot_freeform,
        }

        schema_hint = (
            '{"theory":"three_act","beats":[{"id":"beat_01","act":1,"order":1,'
            '"title":"...","summary":"...","tension_level":"low","purpose":"setup"}]}'
        )

        last_resp: Optional[Dict[str, Any]] = None
        for attempt in range(1, 4):
            resp = await self.client.generate_completion(
                model=model_name,
                temperature=0.4 if attempt == 1 else 0.2,
                system_prompt=outline_prompt,
                prompt=json.dumps(payload, ensure_ascii=False),
                response_format={"type": "json_object"},
                operation_name=f"outline_from_setting_a{attempt}",
            )
            last_resp = resp

            msg = (((resp or {}).get("choices") or [{}])[0].get("message") or {})
            content = (msg.get("content") or "")
            content_stripped = content.strip() if isinstance(content, str) else ""

            if artifact_store is not None:
                artifact_store.save(
                    f"raw/outline_resp_a{attempt}.json",
                    {"resp": resp, "content_preview": content_stripped[:2000]},
                )

            if not content_stripped:
                app_logger.warning(f"outline_from_setting: empty content on attempt {attempt}, retrying...")
                continue

            data = await self._parse_json_with_repair(
                raw=content_stripped,
                model_name=model_name,
                operation_name=f"outline_from_setting_parse_a{attempt}",
                schema_hint=schema_hint,
                artifact_store=artifact_store,
            )

            if not isinstance(data, dict) or not data.get("beats"):
                app_logger.warning(
                    f"outline_from_setting: parsed JSON missing beats on attempt {attempt}, retrying..."
                )
                continue

            outline = StoryOutlineFull(**data)
            outline = self._normalize_outline_order(outline, artifact_store=artifact_store)
            outline.beats = sorted(outline.beats, key=lambda b: b.order)

            if artifact_store is not None:
                artifact_store.checkpoint("03_outline", outline.dict())
            return outline

        if artifact_store is not None:
            artifact_store.save("raw/outline_failed_last_resp.json", {"resp": last_resp})
        raise ValueError("Failed to generate outline: model returned empty/invalid JSON multiple times")

    async def _extract_plot_threads(
        self,
        user_prompt: str,
        setting: Setting,
        outline: StoryOutlineFull,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> List[Dict[str, Any]]:
        app_logger.info("Extracting plot threads...")
        model_name = self.router.get_model_for_agent("plot_thread_agent")

        payload = {
            "user_prompt": user_prompt,
            "setting": setting.dict(),
            "beats": [b.dict() for b in outline.beats],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=plot_thread_extractor_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="plot_thread_extractor",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"threads":[{"id":"thread_01","title":"...","description":"...","status":"open","anchors":["beat_01"]}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "plot_thread_extractor_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        threads = data.get("threads") or []
        if not isinstance(threads, list):
            return []

        norm: List[Dict[str, Any]] = []
        for i, t in enumerate(threads[:30], start=1):
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id") or f"thread_{i:02d}")
            norm.append(
                {
                    "id": tid,
                    "title": str(t.get("title") or tid),
                    "description": str(t.get("description") or ""),
                    "status": str(t.get("status") or "open"),
                    "anchors": t.get("anchors") if isinstance(t.get("anchors"), list) else [],
                    "priority": str(t.get("priority") or "major"),
                    "closure_signal": str(t.get("closure_signal") or ""),
                    "can_remain_open": bool(t.get("can_remain_open", False)),
                    "branch_scope": str(t.get("branch_scope") or "global"),
                    "branch_id": t.get("branch_id"),
                }
            )

        if artifact_store is not None:
            artifact_store.checkpoint("04_threads", {"threads": norm})
        return norm

    async def _plan_branches(
            self,
            outline: StoryOutlineFull,
            max_branches: int,
            tone: Optional[str],
            preferred_ending_types: Optional[List[str]] = None,
            artifact_store: Optional[ArtifactStore] = None,
    ) -> BranchingInfo:

        target_routes = max(1, min(int(max_branches or 1), 5))
        target_non_main = max(0, target_routes - 1)

        beats_sorted = sorted(outline.beats or [], key=lambda b: b.order)
        beat_ids_in_order = [b.id for b in beats_sorted]
        beat_id_set = set(beat_ids_in_order)
        idx_map = {bid: i for i, bid in enumerate(beat_ids_in_order)}

        def _main_spec() -> BranchSpec:
            return BranchSpec(
                id="main",
                from_beat_id=None,
                from_scene_id=None,
                kind="route",
                title="Основной маршрут",
                description="Каноничная история с хорошей, логичной развязкой.",
                ending_tone="good",
                is_canonical=True,
            )

        def _normalize_main_route(raw: Any) -> List[str]:
            if not isinstance(raw, list):
                return beat_ids_in_order[:]

            cleaned: List[str] = []
            seen = set()
            for x in raw:
                bid = str(x or "").strip()
                if not bid or bid not in beat_id_set or bid in seen:
                    continue
                seen.add(bid)
                cleaned.append(bid)

            if not cleaned:
                return beat_ids_in_order[:]

            cleaned.sort(key=lambda bid: idx_map[bid])

            last_idx = idx_map[cleaned[-1]]
            prefix = beat_ids_in_order[: last_idx + 1]
            if set(cleaned) != set(prefix):
                return beat_ids_in_order[:]

            return prefix

        def _parse_valid_non_main(raw_items: Any, main_route_beat_ids: List[str]) -> List[BranchSpec]:
            route_set = set(main_route_beat_ids[:-1])
            out: List[BranchSpec] = []
            seen = set()

            if not isinstance(raw_items, list):
                return out

            for br in raw_items:
                if not isinstance(br, dict):
                    continue

                from_beat_id = str(br.get("from_beat_id") or "").strip()
                if from_beat_id not in route_set:
                    continue

                title = str(br.get("title") or "").strip() or "Альтернативная ветка"
                description = str(br.get("description") or "").strip()
                ending_tone = str(br.get("ending_tone") or "neutral").strip() or "neutral"

                sig = (
                    from_beat_id,
                    title.casefold(),
                    description.casefold(),
                    ending_tone.casefold(),
                )
                if sig in seen:
                    continue
                seen.add(sig)

                out.append(
                    BranchSpec(
                        id="__tmp__",
                        from_beat_id=from_beat_id,
                        from_scene_id=None,
                        kind="ending",
                        title=title,
                        description=description,
                        ending_tone=ending_tone,
                        is_canonical=False,
                    )
                )

            return out

        def _map_pref_tones(values: Optional[List[str]]) -> List[str]:
            out: List[str] = []
            for v in values or []:
                s = str(v or "").strip().lower()
                if not s:
                    continue
                if "хэп" in s or "happy" in s or "good" in s or "светл" in s:
                    out.append("good")
                elif "плох" in s or "bad" in s or "dark" in s or "траг" in s:
                    out.append("bad")
                elif "горь" in s or "bittersweet" in s:
                    out.append("bittersweet")
                elif "open" in s or "открыт" in s:
                    out.append("open")
                else:
                    out.append("neutral")
            if not out:
                out = ["good", "bittersweet", "bad", "open", "neutral"]
            return out

        def _rank_candidate_beats(main_route_beat_ids: List[str]) -> List[str]:
            candidates: List[Tuple[int, float, str]] = []
            for bid in main_route_beat_ids[1:-1]:
                beat = next((b for b in beats_sorted if b.id == bid), None)
                purpose = self._normalize_purpose_value((beat.purpose if beat else "") or "")

                score = 0
                if purpose in {"turning_point", "revelation", "conflict", "rising_action", "midpoint", "crisis",
                               "climax"}:
                    score += 10
                if purpose in {"introduction", "setup", "reaction", "sequel", "travel", "resolution", "epilogue"}:
                    score -= 5

                pos = idx_map[bid] / max(1, len(beat_ids_in_order) - 1)
                if pos < 0.20:
                    score -= 4

                candidates.append((score, pos, bid))

            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            ranked = [bid for _, _, bid in candidates]

            if not ranked:
                ranked = main_route_beat_ids[1:-1] or main_route_beat_ids[:-1]

            return ranked

        def _fill_missing(existing: List[BranchSpec], main_route_beat_ids: List[str]) -> List[BranchSpec]:
            filled = list(existing)
            if len(filled) >= target_non_main:
                return filled[:target_non_main]

            ranked_beats = _rank_candidate_beats(main_route_beat_ids)
            if not ranked_beats:
                return filled[:target_non_main]

            tone_pool = _map_pref_tones(preferred_ending_types)

            guard = 0
            while len(filled) < target_non_main and guard < 100:
                i = len(filled)
                from_beat_id = ranked_beats[i % len(ranked_beats)]
                ending_tone = tone_pool[i % len(tone_pool)]

                filled.append(
                    BranchSpec(
                        id="__tmp__",
                        from_beat_id=from_beat_id,
                        from_scene_id=None,
                        kind="ending",
                        title=f"Альтернативная ветка {i + 1}",
                        description=f"Маршрут расходится после {from_beat_id} и ведёт к {ending_tone} финалу.",
                        ending_tone=ending_tone,
                        is_canonical=False,
                    )
                )
                guard += 1

            return filled[:target_non_main]

        if target_non_main <= 0:
            bi = BranchingInfo(
                max_branches=0,
                main_route_beat_ids=beat_ids_in_order,
                branches=[_main_spec()],
            )
            if artifact_store is not None:
                artifact_store.checkpoint("05_branching", bi.dict())
            return bi

        model_name = self.router.get_model_for_agent("outline_agent")
        schema_hint = (
            '{"main_route_beat_ids":["beat_01","beat_02"],'
            '"branches":[{"from_beat_id":"beat_06","title":"...","description":"...","ending_tone":"neutral"}]}'
        )

        best_valid_non_main: List[BranchSpec] = []
        best_main_route_beat_ids: List[str] = beat_ids_in_order[:]
        feedback: List[str] = []

        for attempt in range(1, 4):
            payload: Dict[str, Any] = {
                "beats": [b.dict() for b in beats_sorted],
                "target_non_main_branches": target_non_main,
                "tone": tone or "balanced",
                "preferred_ending_types": preferred_ending_types or [],
                "validation_feedback": feedback,
            }

            resp = await self.client.generate_completion(
                model=model_name,
                temperature=0.3 if attempt == 1 else 0.15,
                system_prompt=branch_planner_prompt,
                prompt=json.dumps(payload, ensure_ascii=False),
                response_format={"type": "json_object"},
                operation_name=f"branch_planner_a{attempt}",
            )

            raw = (resp["choices"][0]["message"]["content"] or "").strip()
            data = await self._parse_json_with_repair(
                raw,
                model_name,
                f"branch_planner_parse_a{attempt}",
                schema_hint,
                artifact_store=artifact_store,
            )

            main_route_beat_ids = _normalize_main_route(data.get("main_route_beat_ids"))
            valid_non_main = _parse_valid_non_main(data.get("branches") or [], main_route_beat_ids)

            if len(valid_non_main) > len(best_valid_non_main):
                best_valid_non_main = valid_non_main
                best_main_route_beat_ids = main_route_beat_ids

            if len(valid_non_main) == target_non_main:
                best_valid_non_main = valid_non_main
                best_main_route_beat_ids = main_route_beat_ids
                break

            feedback = [
                f"Need exactly {target_non_main} non-main branches.",
                f"Previous attempt produced {len(valid_non_main)} valid non-main branches after validation.",
                "Every from_beat_id must be inside main_route_beat_ids.",
                "Do not place divergence on the final beat of main_route_beat_ids.",
                "If the story feels linear, create late-act ending divergences instead of returning fewer branches.",
            ]

        final_non_main = _fill_missing(best_valid_non_main, best_main_route_beat_ids)

        renumbered_non_main: List[BranchSpec] = []
        for idx, br in enumerate(final_non_main, start=1):
            renumbered_non_main.append(
                BranchSpec(
                    id=f"branch_{idx:02d}",
                    from_beat_id=br.from_beat_id,
                    from_scene_id=None,
                    kind=br.kind,
                    title=br.title,
                    description=br.description,
                    ending_tone=br.ending_tone,
                    is_canonical=False,
                )
            )

        bi = BranchingInfo(
            max_branches=target_non_main,
            main_route_beat_ids=best_main_route_beat_ids,
            branches=[_main_spec(), *renumbered_non_main],
        )

        if artifact_store is not None:
            artifact_store.event(
                "branching.exact_enforced",
                {
                    "requested_non_main": target_non_main,
                    "final_non_main": len(renumbered_non_main),
                    "final_total_routes": len(bi.branches),
                },
            )
            artifact_store.checkpoint("05_branching", bi.dict())

        return bi
    async def _plan_choice_contracts(
        self,
        *,
        setting: Setting,
        outline_main: StoryOutlineFull,
        main_contracts: List[SceneContract],
        branching: BranchingInfo,
        branch_contracts_by_branch: Dict[str, List[SceneContract]],
        artifact_store: Optional[ArtifactStore] = None,
    ) -> List[ChoiceContract]:
        if not main_contracts or not branching or len(branching.branches or []) <= 1:
            return []

        model_name = self.router.get_model_for_agent("outline_agent")

        main_by_beat: Dict[str, SceneContract] = {c.beat_id: c for c in main_contracts}
        main_idx_by_scene: Dict[str, int] = {c.id: i for i, c in enumerate(main_contracts)}
        grouped_entries: Dict[str, List[Tuple[BranchSpec, SceneContract]]] = {}

        for br in branching.branches:
            if br.id == "main" or not br.from_beat_id:
                continue

            if not br.from_scene_id:
                base_scene = main_by_beat.get(br.from_beat_id)
                if base_scene is not None:
                    br.from_scene_id = base_scene.id

            if not br.from_scene_id:
                continue

            first_branch_scene = (branch_contracts_by_branch.get(br.id) or [None])[0]
            if first_branch_scene is None:
                continue

            grouped_entries.setdefault(br.from_scene_id, []).append((br, first_branch_scene))

        if not grouped_entries:
            return []

        candidates: List[Dict[str, Any]] = []
        for decision_scene_id, entries in grouped_entries.items():
            idx = main_idx_by_scene.get(decision_scene_id)
            if idx is None:
                continue

            decision_scene = main_contracts[idx]
            next_main_scene = main_contracts[idx + 1] if idx + 1 < len(main_contracts) else None
            setup_window = 2 if len(main_contracts) <= 12 else 3 if len(main_contracts) <= 24 else 4
            setup_suggested = [main_contracts[i].id for i in range(max(0, idx - setup_window), idx)]

            candidates.append(
                {
                    "decision_scene": decision_scene.dict(),
                    "next_main_scene": next_main_scene.dict() if next_main_scene else None,
                    "setup_scene_ids_suggested": setup_suggested,
                    "branches": [
                        {
                            "branch_spec": br.dict(),
                            "first_branch_scene": first_scene.dict(),
                        }
                        for br, first_scene in entries
                    ],
                }
            )

        payload = {
            "setting": setting.dict(),
            "main_outline": outline_main.dict(),
            "main_contracts": [c.dict() for c in main_contracts],
            "candidates": candidates,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=choice_planner_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="choice_contract_planner",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = (
            '{"choices":[{"id":"choice_01","decision_scene_id":"scene_010","from_beat_id":"beat_06",'
            '"decision_question":"...","why_now":"...","deadline_pressure":"...",'
            '"setup_scene_ids":["scene_008","scene_009"],'
            '"options":[{"id":"opt_main","branch_id":"main","text":"...","intent":"...",'
            '"perceived_cost":"...","immediate_consequence":"..."}]}]}'
        )
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "choice_contract_planner_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        raw_choices = data.get("choices") or []
        raw_by_scene: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_choices, list):
            for it in raw_choices:
                if not isinstance(it, dict):
                    continue
                sid = str(it.get("decision_scene_id") or "").strip()
                if sid:
                    raw_by_scene[sid] = it

        def _ui_text(v: Any, default: str) -> str:
            s = re.sub(r"\s+", " ", str(v or "").strip())
            if s:
                s = re.split(r"[.;!?]\s+", s, maxsplit=1)[0].strip()
            if len(s) > 72:
                s = s[:72].rsplit(" ", 1)[0].strip()
            return s or default

        out: List[ChoiceContract] = []

        for idx, candidate in enumerate(candidates, start=1):
            decision_scene = candidate["decision_scene"]
            decision_scene_id = str(decision_scene["id"])
            decision_idx = main_idx_by_scene[decision_scene_id]
            next_main_scene = candidate.get("next_main_scene")
            next_main_id = (
                str(next_main_scene["id"])
                if isinstance(next_main_scene, dict) and next_main_scene.get("id")
                else None
            )

            entries = grouped_entries.get(decision_scene_id) or []
            raw_choice = raw_by_scene.get(decision_scene_id) or {}

            raw_setup = raw_choice.get("setup_scene_ids") or []
            if not isinstance(raw_setup, list):
                raw_setup = []
            setup_scene_ids = []
            for sid in raw_setup:
                sid_s = str(sid or "").strip()
                if sid_s in main_idx_by_scene and main_idx_by_scene[sid_s] < decision_idx:
                    setup_scene_ids.append(sid_s)
            if not setup_scene_ids:
                setup_scene_ids = candidate.get("setup_scene_ids_suggested") or []

            raw_opts = raw_choice.get("options") or []
            if not isinstance(raw_opts, list):
                raw_opts = []
            raw_opt_by_branch = {}
            for op in raw_opts:
                if not isinstance(op, dict):
                    continue
                bid = str(op.get("branch_id") or "").strip()
                if bid:
                    raw_opt_by_branch[bid] = op

            options: List[ChoiceOptionPlan] = []

            if next_main_id:
                op_raw = raw_opt_by_branch.get("main") or {}
                options.append(
                    ChoiceOptionPlan(
                        id=str(op_raw.get("id") or "opt_main"),
                        branch_id="main",
                        text=_ui_text(
                            op_raw.get("text"),
                            _ui_text((next_main_scene or {}).get("summary"), "Держаться текущего плана"),
                        ),
                        intent=str(op_raw.get("intent") or "Следовать текущему плану"),
                        perceived_cost=str(op_raw.get("perceived_cost") or "Придётся отказаться от альтернативы"),
                        immediate_consequence=str(
                            op_raw.get("immediate_consequence")
                            or (next_main_scene or {}).get("summary")
                            or ""
                        ),
                        target_scene_id=next_main_id,
                    )
                )

            for br, first_scene in entries:
                op_raw = raw_opt_by_branch.get(br.id) or {}
                default_text = br.title or first_scene.summary or f"Выбрать {br.id}"
                options.append(
                    ChoiceOptionPlan(
                        id=str(op_raw.get("id") or f"opt_{br.id}"),
                        branch_id=br.id,
                        text=_ui_text(op_raw.get("text"), _ui_text(default_text, f"Выбрать {br.id}")),
                        intent=str(op_raw.get("intent") or br.description or first_scene.summary or ""),
                        perceived_cost=str(
                            op_raw.get("perceived_cost") or "Это изменит маршрут и усложнит ситуацию"
                        ),
                        immediate_consequence=str(
                            op_raw.get("immediate_consequence") or first_scene.summary or ""
                        ),
                        target_scene_id=first_scene.id,
                    )
                )

            if len(options) < 2:
                continue

            out.append(
                ChoiceContract(
                    id=str(raw_choice.get("id") or f"choice_{idx:02d}"),
                    decision_scene_id=decision_scene_id,
                    from_beat_id=str(raw_choice.get("from_beat_id") or decision_scene.get("beat_id") or ""),
                    decision_question=str(raw_choice.get("decision_question") or decision_scene.get("summary") or ""),
                    why_now=str(raw_choice.get("why_now") or decision_scene.get("summary") or ""),
                    deadline_pressure=str(raw_choice.get("deadline_pressure") or ""),
                    setup_scene_ids=setup_scene_ids,
                    options=options,
                )
            )

        if artifact_store is not None:
            artifact_store.checkpoint("08b_choice_contracts", {"choices": [c.dict() for c in out]})

        return out

    def _build_choice_context_by_scene(
        self,
        main_contracts: List[SceneContract],
        choice_contracts: List[ChoiceContract],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        idx_by_scene = {c.id: i for i, c in enumerate(main_contracts)}

        for cc in choice_contracts or []:
            cc_dict = cc.dict() if hasattr(cc, "dict") else self._obj_to_dict(cc)
            decision_scene_id = str(cc_dict.get("decision_scene_id") or "").strip()
            if not decision_scene_id:
                continue

            out[decision_scene_id] = {"scene_role": "decision", **cc_dict}

            setup_scene_ids = cc_dict.get("setup_scene_ids") or []
            if not setup_scene_ids:
                idx = idx_by_scene.get(decision_scene_id)
                if idx is not None:
                    setup_scene_ids = [main_contracts[i].id for i in range(max(0, idx - 2), idx)]

            for sid in setup_scene_ids:
                sid_s = str(sid or "").strip()
                if not sid_s or sid_s == decision_scene_id:
                    continue

                bucket = out.setdefault(sid_s, {"scene_role": "setup", "upcoming_choices": []})
                if bucket.get("scene_role") == "decision":
                    continue

                bucket.setdefault("upcoming_choices", []).append(
                    {
                        "id": cc_dict.get("id"),
                        "decision_scene_id": decision_scene_id,
                        "decision_question": cc_dict.get("decision_question"),
                        "why_now": cc_dict.get("why_now"),
                        "deadline_pressure": cc_dict.get("deadline_pressure"),
                        "options": cc_dict.get("options") or [],
                    }
                )

        return out

    def _inject_branch_choices(
        self,
        main_scripts: Dict[str, SceneScript],
        choice_contracts: List[ChoiceContract],
    ) -> None:
        for cc in choice_contracts or []:
            script = main_scripts.get(cc.decision_scene_id)
            if script is None:
                continue

            options: List[SceneChoiceOption] = []
            for op in cc.options or []:
                target_scene_id = str(op.target_scene_id or "").strip()
                if not target_scene_id:
                    continue

                options.append(
                    SceneChoiceOption(
                        id=op.id,
                        text=op.text,
                        leads_to_scene_id=target_scene_id,
                        leads_to_branch_id=op.branch_id,
                        is_fake=False,
                    )
                )

            if len(options) < 2:
                continue

            script.choices = [
                SceneChoice(
                    id=cc.id,
                    appears_after_line=max(0, len(script.lines) - 1),
                    options=options,
                )
            ]
    async def _ensure_char_list(
        self,
        user_prompt: str,
        outline: StoryOutlineFull,
        char_list: List[str] | None,
    ) -> List[str]:
        if char_list:
            return char_list

        app_logger.info("No char_list provided. Generating character names...")
        model_name = self.router.get_model_for_agent("char_agent")

        system_prompt = (
            "Ты придумываешь список имён персонажей для визуальной новеллы.\n"
            'Выведи строгий JSON-объект вида:\n{"characters": ["Имя1", "Имя2", "..."]}\n'
            "Не используй Markdown, только JSON."
        )

        payload = {"user_prompt": user_prompt, "beats": [b.dict() for b in outline.beats]}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.6,
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="char_name_list",
        )

        data = json.loads(resp["choices"][0]["message"]["content"])
        chars = data.get("characters", [])
        if not isinstance(chars, list) or not chars:
            raise ValueError("Failed to generate character list")
        return [str(x) for x in chars]

    async def _ensure_loc_list(
        self,
        user_prompt: str,
        outline: StoryOutlineFull,
        loc_list: List[str] | None,
    ) -> List[str]:
        if loc_list:
            return loc_list

        app_logger.info("No loc_list provided. Generating location names...")
        model_name = self.router.get_model_for_agent("loc_agent")

        system_prompt = (
            "Ты придумываешь список локаций для визуальной новеллы.\n"
            'Выведи строгий JSON-объект вида:\n{"locations": ["Локация1", "Локация2", "..."]}\n'
            "Не используй Markdown, только JSON."
        )

        payload = {"user_prompt": user_prompt, "beats": [b.dict() for b in outline.beats]}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.6,
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="loc_name_list",
        )

        data = json.loads(resp["choices"][0]["message"]["content"])
        locs = data.get("locations", [])
        if not isinstance(locs, list) or not locs:
            raise ValueError("Failed to generate location list")
        return [str(x) for x in locs]

    async def _run_char_agent(
        self,
        char_list: List[str],
        setting: Setting,
        hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        app_logger.info("Running char_agent...")
        char_toolbox = (
            Toolbox()
            .add_tool_schema("schema_char_graph")
            .add_tool_schema("schema_char_appearance")
            .add_tool_schema("schema_char_type")
        )
        char_supports = self.router.supports_function_calling("char_agent")

        char_agent = Agent(
            api_key=self.client.api_key,
            model=self.router.get_model_for_agent("char_agent"),
            temperature=0.1,
            toolbox=char_toolbox,
            freedom="required",
            reasoning="high",
            supports_tools=char_supports,
        ).set_role(char_prompt)

        char_agent.set_input(char_list=char_list, setting=setting.setting, hints=hints or {})
        return await char_agent.run(self.client)

    def _trim_beats_to_single_ending(self, beats: List[OutlineBeat]) -> List[OutlineBeat]:

        beats_sorted = sorted(beats or [], key=lambda b: b.order)
        if not beats_sorted:
            return beats_sorted

        terminal = {"resolution", "epilogue"}
        major_restart = {"turning_point", "rising_action", "conflict", "midpoint", "crisis", "climax"}

        end_idx: Optional[int] = None
        seen_terminal = False

        for idx, b in enumerate(beats_sorted):
            p = self._normalize_purpose_value(b.purpose or "")

            if p in terminal:
                seen_terminal = True
                end_idx = idx
                continue

            if seen_terminal and p in major_restart:
                break

            if seen_terminal:
                end_idx = idx

        return beats_sorted if end_idx is None else beats_sorted[: end_idx + 1]

    async def _run_loc_agent(
        self,
        loc_list: List[str],
        setting: Setting,
    ) -> Dict[str, Any]:
        app_logger.info("Running loc_agent...")
        loc_toolbox = Toolbox().add_tool_schema("schema_loc_description").add_tool_schema("schema_loc_graph")
        loc_supports = self.router.supports_function_calling("loc_agent")

        loc_agent = Agent(
            api_key=self.client.api_key,
            model=self.router.get_model_for_agent("loc_agent"),
            temperature=0.1,
            toolbox=loc_toolbox,
            freedom="required",
            reasoning="high",
            supports_tools=loc_supports,
        ).set_role(loc_prompt)

        loc_agent.set_input(loc_list=loc_list, setting=setting.setting)
        return await loc_agent.run(self.client)

    async def _ensure_char_appearance_complete(
            self,
            *,
            char_list: List[str],
            char_appearance: Any,
            setting: Setting,
            hints: Optional[Dict[str, str]] = None,
            artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Any]:
        n = len(char_list or [])
        descs: List[str] = []
        if isinstance(char_appearance, dict):
            v = char_appearance.get("descriptions")
            descs = v if isinstance(v, list) else []
        elif hasattr(char_appearance, "descriptions"):
            descs = list(getattr(char_appearance, "descriptions") or [])

        bad = (n > 0 and len(descs) != n) or (n > 0 and sum(1 for d in descs if not str(d or "").strip()) > n // 2)
        if not bad:
            descs = [str(x or "") for x in descs]
            if len(descs) < n:
                descs += [""] * (n - len(descs))
            if len(descs) > n:
                descs = descs[:n]
            return {"descriptions": descs}

        app_logger.warning(
            f"char_appearance invalid (len={len(descs)} expected={n}); regenerating via Toolbox.char_appearance")

        tb = Toolbox()
        tb.model = self.router.get_model_for_agent("char_agent")
        ca = await tb.char_appearance(
            self.client,
            char_list=char_list,
            setting=setting.setting,
            hints=hints or {},
        )
        payload = ca.dict() if hasattr(ca, "dict") else (
            ca.model_dump() if hasattr(ca, "model_dump") else {"descriptions": []})

        out = payload.get("descriptions") if isinstance(payload, dict) else []
        out = out if isinstance(out, list) else []
        out = [str(x or "") for x in out]
        if len(out) < n:
            out += [""] * (n - len(out))
        if len(out) > n:
            out = out[:n]

        if artifact_store is not None:
            artifact_store.event("char_appearance.regenerated", {"expected": n, "got": len(out)})

        return {"descriptions": out}
    async def _ensure_loc_description_complete(
            self,
            *,
            loc_list: List[str],
            loc_description: Any,
            setting: Setting,
            artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Any]:
        n = len(loc_list or [])
        descs: List[str] = []
        if isinstance(loc_description, dict):
            v = loc_description.get("descriptions")
            descs = v if isinstance(v, list) else []
        elif hasattr(loc_description, "descriptions"):
            descs = list(getattr(loc_description, "descriptions") or [])

        bad = (n > 0 and len(descs) != n) or (n > 0 and sum(1 for d in descs if not str(d or "").strip()) > n // 2)
        if not bad:
            descs = [str(x or "") for x in descs]
            if len(descs) < n:
                descs += [""] * (n - len(descs))
            if len(descs) > n:
                descs = descs[:n]
            return {"descriptions": descs}

        app_logger.warning(
            f"loc_description invalid (len={len(descs)} expected={n}); regenerating via Toolbox.loc_description"
        )

        tb = Toolbox()
        tb.model = self.router.get_model_for_agent("loc_agent")
        ld = await tb.loc_description(
            self.client,
            loc_list=loc_list,
            setting=setting.setting,
        )
        payload = ld.dict() if hasattr(ld, "dict") else (
            ld.model_dump() if hasattr(ld, "model_dump") else {"descriptions": []}
        )

        out = payload.get("descriptions") if isinstance(payload, dict) else []
        out = out if isinstance(out, list) else []
        out = [str(x or "") for x in out]
        if len(out) < n:
            out += [""] * (n - len(out))
        if len(out) > n:
            out = out[:n]

        if artifact_store is not None:
            artifact_store.event("loc_description.regenerated", {"expected": n, "got": len(out)})

        return {"descriptions": out}
    def _beats_for_main_route(
        self,
        outline: StoryOutlineFull,
        branching: Optional[BranchingInfo] = None,
    ) -> StoryOutlineFull:
        beats_sorted = sorted(outline.beats, key=lambda b: b.order)
        if not beats_sorted:
            return StoryOutlineFull(theory=outline.theory, beats=[])

        if branching is not None and branching.main_route_beat_ids:
            route_set = {str(x).strip() for x in branching.main_route_beat_ids if str(x).strip()}
            used = [b for b in beats_sorted if b.id in route_set]
            if used:
                return StoryOutlineFull(theory=outline.theory, beats=used)

        used = self._trim_beats_to_single_ending(beats_sorted)
        return StoryOutlineFull(theory=outline.theory, beats=used)

    async def _generate_scene_contracts_main(
            self,
            outline: StoryOutlineFull,
            char_list: List[str],
            loc_list: List[str],
            story_length: str,
            char_type: Optional[Dict[str, List[str]]] = None,
            branching: Optional[BranchingInfo] = None,
            plot_threads: Optional[List[Dict[str, Any]]] = None,
            artifact_store: Optional[ArtifactStore] = None,
    ) -> List[SceneContract]:
        app_logger.info("Generating scene contracts (main)...")
        model_name = self.router.get_model_for_agent("outline_agent")
        mc_name = char_list[0] if char_list else None
        outline_main = outline
        beats_sorted = sorted(outline_main.beats, key=lambda b: b.order)
        payload = {
            "outline": outline_main.dict(),
            "char_list": char_list,
            "loc_list": loc_list,
            "story_length": story_length,
            "mc_name": mc_name,
            "char_type": char_type or {},
            "branching_info": branching.dict() if branching else None,
            "plot_threads": plot_threads or [],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.4,
            system_prompt=scene_plan_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="scene_contracts_main",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = (
            '{"scenes":[{"beat_id":"beat_01","location":"...","pov_character":"...","present_characters":["..."],'
            '"summary":"...","scene_goal":"...","scene_conflict":"...","stakes":"...","reveal":"...",'
            '"emotional_beat":"...","must_reference":["..."],"entry_requirements":["..."],'
            '"exit_targets":["..."],"continuity_notes":["..."],"thread_focus":["thread_01"]}]}'
        )
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "scene_contracts_main_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        raw_scenes = data.get("scenes") or []

        if not isinstance(raw_scenes, list) or not raw_scenes:
            pov = mc_name or (char_list[0] if char_list else "Protagonist")
            default_loc = loc_list[0] if loc_list else "Default Location"
            contracts = [
                SceneContract(
                    id=f"scene_{i+1:03d}",
                    beat_id=b.id,
                    location=default_loc,
                    pov_character=pov,
                    present_characters=[pov],
                    summary=b.summary,
                    branch_id="main",
                    branch_order=i + 1,
                )
                for i, b in enumerate(beats_sorted)
            ]
            default_loc = loc_list[0] if loc_list else "Default Location"
            default_pov = mc_name or (char_list[0] if char_list else "Protagonist")

            contracts = self._ensure_beat_coverage_contracts(
                beats=beats_sorted,
                contracts=contracts,
                branch_id="main",
                id_prefix="scene_",
                default_loc=default_loc,
                default_pov=default_pov,
            )

            if artifact_store is not None:
                artifact_store.checkpoint("08_scene_contracts_main", [c.dict() for c in contracts])
            return contracts

        contracts: List[SceneContract] = []
        for idx, s in enumerate(raw_scenes):
            if not isinstance(s, dict):
                continue
            beat_id = s.get("beat_id") or (beats_sorted[idx].id if idx < len(beats_sorted) else beats_sorted[-1].id)
            location = s.get("location") or (loc_list[0] if loc_list else "Default Location")
            pov_character = s.get("pov_character") or (mc_name or (char_list[0] if char_list else "Protagonist"))
            present = s.get("present_characters")
            if not isinstance(present, list) or not present:
                present = [pov_character]
            summary = str(s.get("summary") or "")
            scene_goal = str(s.get("scene_goal") or "")
            scene_conflict = str(s.get("scene_conflict") or "")
            stakes = str(s.get("stakes") or "")
            reveal = str(s.get("reveal") or "")
            emotional_beat = str(s.get("emotional_beat") or "")

            contracts.append(
                SceneContract(
                    id=f"scene_{idx + 1:03d}",
                    beat_id=str(beat_id),
                    location=str(location),
                    pov_character=str(pov_character),
                    present_characters=[str(x) for x in present],
                    summary=summary,
                    scene_goal=scene_goal,
                    scene_conflict=scene_conflict,
                    stakes=stakes,
                    reveal=reveal,
                    emotional_beat=emotional_beat,
                    must_reference=self._norm_text_list(s.get("must_reference"), max_items=8),
                    entry_requirements=self._norm_text_list(s.get("entry_requirements"), max_items=8),
                    exit_targets=self._norm_text_list(s.get("exit_targets"), max_items=8),
                    continuity_notes=self._norm_text_list(s.get("continuity_notes"), max_items=8),
                    thread_focus=self._norm_text_list(s.get("thread_focus"), max_items=6, max_len=80),
                    branch_id="main",
                    branch_order=idx + 1,
                )
            )

        default_loc = loc_list[0] if loc_list else "Default Location"
        default_pov = mc_name or (char_list[0] if char_list else "Protagonist")

        contracts = self._ensure_beat_coverage_contracts(
            beats=beats_sorted,
            contracts=contracts,
            branch_id="main",
            id_prefix="scene_",
            default_loc=default_loc,
            default_pov=default_pov,
        )

        if artifact_store is not None:
            artifact_store.checkpoint("08_scene_contracts_main", [c.dict() for c in contracts])
        return contracts

    async def _generate_scene_contracts_for_branch(
            self,
            outline: StoryOutlineFull,
            char_list: List[str],
            loc_list: List[str],
            story_length: str,
            branch: BranchSpec,
            setting: Setting,
            char_type: Optional[Dict[str, List[str]]] = None,
            artifact_store: Optional[ArtifactStore] = None,
            divergence_scene_contract: Optional[SceneContract] = None,
            plot_threads: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[StoryOutlineFull, List[SceneContract]]:
        if not branch.from_beat_id:
            return StoryOutlineFull(theory=outline.theory, beats=[]), []

        beats_sorted = sorted(outline.beats, key=lambda b: b.order)
        order_map = {b.id: b.order for b in beats_sorted}
        if branch.from_beat_id not in order_map:
            return StoryOutlineFull(theory=outline.theory, beats=[]), []

        split_order = order_map[branch.from_beat_id]
        tail_beats = [b for b in beats_sorted if b.order > split_order]
        tail_beats = self._trim_beats_to_single_ending(tail_beats)
        if not tail_beats:
            return StoryOutlineFull(theory=outline.theory, beats=[]), []

        model_name = self.router.get_model_for_agent("outline_agent")
        mc_name = char_list[0] if char_list else None

        outline_tail = StoryOutlineFull(theory=outline.theory, beats=tail_beats)
        outline_tail = await self._rewrite_tail_outline_for_branch(
            setting=setting,
            outline_tail=outline_tail,
            branch=branch,
            char_list=char_list,
            loc_list=loc_list,
            artifact_store=artifact_store,
        )
        tail_beats_rewritten = sorted(outline_tail.beats, key=lambda b: b.order)
        prelude_beats = [b for b in beats_sorted if b.order <= split_order]
        branch_outline = StoryOutlineFull(
            theory=outline.theory,
            beats=prelude_beats + tail_beats_rewritten,
        )
        payload = {
            "outline": outline_tail.dict(),
            "char_list": char_list,
            "loc_list": loc_list,
            "story_length": story_length,
            "mc_name": mc_name,
            "char_type": char_type or {},
            "branch_context": {
                "branch_id": branch.id,
                "from_beat_id": branch.from_beat_id,
                "title": branch.title,
                "description": branch.description,
                "ending_tone": branch.ending_tone,
            },
            "divergence_scene_contract": divergence_scene_contract.dict() if divergence_scene_contract else None,
            "plot_threads": plot_threads or [],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.4,
            system_prompt=scene_plan_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"scene_contracts_{branch.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = (
            '{"scenes":[{"beat_id":"beat_01","location":"...","pov_character":"...","present_characters":["..."],'
            '"summary":"...","scene_goal":"...","scene_conflict":"...","stakes":"...","reveal":"...",'
            '"emotional_beat":"...","must_reference":["..."],"entry_requirements":["..."],'
            '"exit_targets":["..."],"continuity_notes":["..."],"thread_focus":["thread_01"]}]}'
        )
        data = await self._parse_json_with_repair(raw, model_name, f"scene_contracts_{branch.id}_parse", schema_hint)
        raw_scenes = data.get("scenes") or []
        if not isinstance(raw_scenes, list) or not raw_scenes:
             return branch_outline, []

        contracts: List[SceneContract] = []
        for idx, s in enumerate(raw_scenes):
            if not isinstance(s, dict):
                continue
            beat_id = s.get("beat_id") or (
                tail_beats_rewritten[idx].id if idx < len(tail_beats_rewritten) else tail_beats_rewritten[-1].id
            )
            location = s.get("location") or (loc_list[0] if loc_list else "Unknown Location")
            pov_character = s.get("pov_character") or mc_name or (char_list[0] if char_list else "Protagonist")
            present = s.get("present_characters")
            if not isinstance(present, list) or not present:
                present = [pov_character]
            summary = str(s.get("summary") or "")
            scene_goal = str(s.get("scene_goal") or "")
            scene_conflict = str(s.get("scene_conflict") or "")
            stakes = str(s.get("stakes") or "")
            reveal = str(s.get("reveal") or "")
            emotional_beat = str(s.get("emotional_beat") or "")

            contracts.append(
                SceneContract(
                    id=f"{branch.id}_scene_{idx + 1:03d}",
                    beat_id=str(beat_id),
                    location=str(location),
                    pov_character=str(pov_character),
                    present_characters=[str(x) for x in present],
                    summary=summary,
                    scene_goal=scene_goal,
                    scene_conflict=scene_conflict,
                    stakes=stakes,
                    reveal=reveal,
                    emotional_beat=emotional_beat,
                    must_reference=self._norm_text_list(s.get("must_reference"), max_items=8),
                    entry_requirements=self._norm_text_list(s.get("entry_requirements"), max_items=8),
                    exit_targets=self._norm_text_list(s.get("exit_targets"), max_items=8),
                    continuity_notes=self._norm_text_list(s.get("continuity_notes"), max_items=8),
                    thread_focus=self._norm_text_list(s.get("thread_focus"), max_items=6, max_len=80),
                    branch_id=branch.id,
                    branch_order=idx + 1,
                )
            )

        default_loc = loc_list[0] if loc_list else "Unknown Location"
        default_pov = mc_name or (char_list[0] if char_list else "Protagonist")

        contracts = self._ensure_beat_coverage_contracts(
            beats=tail_beats_rewritten,
            contracts=contracts,
            branch_id=branch.id,
            id_prefix=f"{branch.id}_scene_",
            default_loc=default_loc,
            default_pov=default_pov,
        )

        return branch_outline, contracts

    @staticmethod
    def _extract_last_lines(script: SceneScript, n: int = 3) -> List[str]:
        out: List[str] = []
        if not script or not script.lines:
            return out
        for line in script.lines[-n:]:
            speaker_prefix = f"{line.speaker}: " if line.speaker else ""
            out.append(f"[{line.type}] {speaker_prefix}{line.text}")
        return out

    @staticmethod
    def _neighbors_from_loc_graph(loc_graph: Any, loc: str, limit: int = 12) -> List[str]:
        if not loc_graph or not loc:
            return []
        edges = loc_graph.get("edges") if isinstance(loc_graph, dict) else getattr(loc_graph, "edges", None)
        if not isinstance(edges, list):
            return []
        neigh: List[str] = []
        for e in edges:
            if isinstance(e, dict):
                src = e.get("source")
                tgt = e.get("target")
                bidir = bool(e.get("bidirectional", False))
            else:
                src = getattr(e, "source", None)
                tgt = getattr(e, "target", None)
                bidir = bool(getattr(e, "bidirectional", False))
            if src == loc and tgt:
                neigh.append(str(tgt))
            if tgt == loc and src and bidir:
                neigh.append(str(src))
            if len(neigh) >= limit:
                break
        seen = set()
        uniq: List[str] = []
        for x in neigh:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    @staticmethod
    def _present_relationship_notes(char_graph: Any, present_characters: List[str], limit: int = 12) -> List[str]:

        if not char_graph or not present_characters or len(present_characters) < 2:
            return []

        edges = char_graph.get("edges") if isinstance(char_graph, dict) else getattr(char_graph, "edges", None)
        if not isinstance(edges, list):
            return []

        present = set(str(x) for x in present_characters)
        out: List[str] = []

        for e in edges:
            if isinstance(e, dict):
                src = e.get("source")
                tgt = e.get("target")
                label = e.get("label")
                directed = bool(e.get("directed", True))
            else:
                src = getattr(e, "source", None)
                tgt = getattr(e, "target", None)
                label = getattr(e, "label", None)
                directed = bool(getattr(e, "directed", True))

            if str(src) in present and str(tgt) in present:
                arrow = "->" if directed else "<->"
                out.append(f"{src} {arrow} {tgt}: {label}")
                if len(out) >= limit:
                    break

        return out

    def _loc_adjacency(self, loc_graph: Any) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        if not loc_graph:
            return out

        edges = loc_graph.get("edges") if isinstance(loc_graph, dict) else getattr(loc_graph, "edges", None)
        if not isinstance(edges, list):
            return out

        for e in edges:
            if isinstance(e, dict):
                src = e.get("source")
                tgt = e.get("target")
                bidir = bool(e.get("bidirectional", False))
            else:
                src = getattr(e, "source", None)
                tgt = getattr(e, "target", None)
                bidir = bool(getattr(e, "bidirectional", False))

            if src and tgt:
                out.setdefault(str(src), []).append(str(tgt))
                if bidir:
                    out.setdefault(str(tgt), []).append(str(src))

        for k, lst in list(out.items()):
            seen = set()
            uniq: List[str] = []
            for x in lst:
                if x in seen:
                    continue
                seen.add(x)
                uniq.append(x)
            out[k] = uniq

        return out

    def _shortest_loc_path(
        self,
        loc_graph: Any,
        src: Optional[str],
        dst: Optional[str],
        max_depth: int = 5,
    ) -> List[str]:
        if not src or not dst:
            return [x for x in [src, dst] if x]
        if src == dst:
            return [src]

        adj = self._loc_adjacency(loc_graph)
        if src not in adj:
            return [src, dst]

        q = deque([[src]])
        seen = {src}

        while q:
            path = q.popleft()
            node = path[-1]
            if len(path) - 1 > max_depth:
                continue
            for nb in adj.get(node, []):
                if nb == dst:
                    return path + [nb]
                if nb not in seen:
                    seen.add(nb)
                    q.append(path + [nb])

        return [src, dst]

    @staticmethod
    def _indoor_markers_found(text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        markers = [
            "вош", "внутр", "двер", "стен", "потол", "комнат", "коридор", "бункер",
            "кабинет", "лестниц", "туннел", "помещени", "забаррикад", "закрыл", "запер"
        ]
        return any(m in t for m in markers)

    def _sanitize_contracts_schema(
            self,
            contracts: List[SceneContract],
            *,
            char_list: List[str],
            loc_list: List[str],
    ) -> List[SceneContract]:
        if not contracts:
            return contracts

        char_set = set(char_list or [])
        loc_set = set(loc_list or [])

        out: List[SceneContract] = []
        for c in contracts:

            if c.location not in loc_set and loc_list:
                c.location = loc_list[0]


            if c.pov_character not in char_set and char_list:
                c.pov_character = char_list[0]

            pres: List[str] = []
            seen = set()
            for x in (c.present_characters or []):
                if x in char_set and x not in seen:
                    pres.append(x)
                    seen.add(x)
            if c.pov_character and c.pov_character in char_set and c.pov_character not in seen:
                pres.insert(0, c.pov_character)
            if not pres and c.pov_character:
                pres = [c.pov_character]

            c.present_characters = pres
            out.append(c)

        return out

    async def _patch_scene_contracts_with_consistency_critic(
            self,
            *,
            scene_contracts: List[SceneContract],
            setting: Setting,
            char_list: List[str],
            loc_list: List[str],
            loc_canons: Dict[str, str],
            loc_affordances: Dict[str, Dict[str, Any]],
            loc_graph: Optional[Any],
            plot_threads: Optional[List[Dict[str, Any]]] = None,
            artifact_store: Optional[ArtifactStore] = None,
            artifact_name: str = "contracts_qc",
            branch_context: Optional[Dict[str, Any]] = None,
    ) -> List[SceneContract]:
        if not scene_contracts:
            return scene_contracts

        model_name = self.router.get_model_for_agent("qc_agent")
        known_thread_ids = {
            str(t.get("id"))
            for t in (plot_threads or [])
            if isinstance(t, dict) and str(t.get("id") or "").strip()
        }

        payload = {
            "setting": {"setting": setting.setting, "world_rules": setting.world_rules},
            "char_list": char_list,
            "loc_list": loc_list,
            "loc_canons": {k: (v[:1200] if isinstance(v, str) else "") for k, v in (loc_canons or {}).items()},
            "loc_affordances": loc_affordances or {},
            "loc_graph": loc_graph,
            "plot_threads": plot_threads or [],
            "scene_contracts": [c.dict() for c in scene_contracts],
            "branch_context": branch_context,
            "constraints": {
                "max_same_location_run": int(os.getenv("QC_MAX_SAME_LOCATION_RUN", "2") or "2"),
                "prefer_fix_summary_over_location": True,
            },
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.0,
            system_prompt=contract_consistency_critic_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"contract_consistency_critic_{artifact_name}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = (
            '{"patches":[{"scene_id":"scene_001","new_location":null,"new_pov_character":null,'
            '"new_present_characters":null,"new_summary":null,"new_scene_goal":null,'
            '"new_scene_conflict":null,"new_stakes":null,"new_reveal":null,"new_emotional_beat":null,'
            '"new_must_reference":null,"new_entry_requirements":null,"new_exit_targets":null,'
            '"new_continuity_notes":null,"new_thread_focus":null,"reason":"...","confidence":0.8}]}'
        )
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"contract_consistency_critic_{artifact_name}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        patches = data.get("patches") or []
        if artifact_store is not None:
            artifact_store.save(f"contracts/{artifact_name}_qc_patches.json", {"patches": patches})

        if not isinstance(patches, list) or not patches:
            for c in scene_contracts:
                self._sanitize_contract_rich_fields(c, known_thread_ids=known_thread_ids)
            return scene_contracts

        by_id: Dict[str, SceneContract] = {c.id: c for c in scene_contracts}
        applied: List[Dict[str, Any]] = []
        min_conf = float(os.getenv("QC_MIN_CONFIDENCE", "0.6") or "0.6")

        for p in patches[:250]:
            if not isinstance(p, dict):
                continue
            sid = str(p.get("scene_id") or "")
            if sid not in by_id:
                continue

            conf = p.get("confidence")
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if conf_f < min_conf:
                continue

            c = by_id[sid]
            changed = False

            new_loc = p.get("new_location")
            if isinstance(new_loc, str):
                nl = new_loc.strip()
                if nl and nl in loc_list and nl != c.location:
                    c.location = nl
                    changed = True

            new_pov = p.get("new_pov_character")
            if isinstance(new_pov, str):
                np = new_pov.strip()
                if np and np in char_list and np != c.pov_character:
                    c.pov_character = np
                    changed = True

            new_pres = p.get("new_present_characters")
            if isinstance(new_pres, list):
                pres = [str(x).strip() for x in new_pres if isinstance(x, str) and str(x).strip() in char_list]
                if c.pov_character and c.pov_character not in pres:
                    pres.insert(0, c.pov_character)
                if pres and pres != c.present_characters:
                    c.present_characters = pres
                    changed = True

            new_sum = p.get("new_summary")
            if isinstance(new_sum, str):
                ns = new_sum.strip()
                if ns and ns != c.summary:
                    c.summary = ns
                    changed = True

            rich_changed = self._apply_rich_contract_patch(
                c,
                p,
                known_thread_ids=known_thread_ids,
            )
            changed = changed or rich_changed

            if changed:
                applied.append(
                    {
                        "scene_id": sid,
                        "confidence": conf_f,
                        "rich_fields_patched": bool(rich_changed),
                        "reason": p.get("reason"),
                    }
                )

        for c in by_id.values():
            self._sanitize_contract_rich_fields(c, known_thread_ids=known_thread_ids)

        if artifact_store is not None and applied:
            artifact_store.save(f"contracts/{artifact_name}_qc_applied.json", {"applied": applied})

        return list(by_id.values())
    async def _re_enrich_scene_contracts(
        self,
        *,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contracts: List[SceneContract],
        plot_threads: Optional[List[Dict[str, Any]]] = None,
        artifact_store: Optional[ArtifactStore] = None,
        artifact_name: str = "main",
        branch_context: Optional[Dict[str, Any]] = None,
    ) -> List[SceneContract]:
        if not scene_contracts:
            return scene_contracts

        model_name = self.router.get_model_for_agent("outline_agent")
        known_thread_ids = {
            str(t.get("id"))
            for t in (plot_threads or [])
            if isinstance(t, dict) and str(t.get("id") or "").strip()
        }

        payload = {
            "setting": setting.dict(),
            "outline": outline.dict(),
            "plot_threads": plot_threads or [],
            "branch_context": branch_context,
            "scene_contracts": [c.dict() for c in scene_contracts],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=scene_contract_reenricher_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"scene_contract_reenricher_{artifact_name}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = (
            '{"scenes":[{"scene_id":"scene_001","scene_goal":"...","scene_conflict":"...",'
            '"stakes":"...","reveal":"...","emotional_beat":"...","must_reference":["..."],'
            '"entry_requirements":["..."],"exit_targets":["..."],"continuity_notes":["..."],'
            '"thread_focus":["thread_01"]}]}'
        )
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"scene_contract_reenricher_{artifact_name}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        items = data.get("scenes") or []
        by_id: Dict[str, SceneContract] = {c.id: c for c in scene_contracts}
        applied: List[Dict[str, Any]] = []

        if isinstance(items, list):
            for it in items[:300]:
                if not isinstance(it, dict):
                    continue
                sid = str(it.get("scene_id") or "").strip()
                if not sid or sid not in by_id:
                    continue

                c = by_id[sid]
                rich_patch = {
                    "new_scene_goal": it.get("scene_goal"),
                    "new_scene_conflict": it.get("scene_conflict"),
                    "new_stakes": it.get("stakes"),
                    "new_reveal": it.get("reveal"),
                    "new_emotional_beat": it.get("emotional_beat"),
                    "new_must_reference": it.get("must_reference"),
                    "new_entry_requirements": it.get("entry_requirements"),
                    "new_exit_targets": it.get("exit_targets"),
                    "new_continuity_notes": it.get("continuity_notes"),
                    "new_thread_focus": it.get("thread_focus"),
                }

                changed = self._apply_rich_contract_patch(
                    c,
                    rich_patch,
                    known_thread_ids=known_thread_ids,
                )
                if changed:
                    applied.append({"scene_id": sid})

        for c in by_id.values():
            self._sanitize_contract_rich_fields(c, known_thread_ids=known_thread_ids)

        if artifact_store is not None:
            artifact_store.save(
                f"contracts/{artifact_name}_reenriched.json",
                {
                    "applied": applied,
                    "contracts": [c.dict() for c in by_id.values()],
                },
            )

        return list(by_id.values())

    async def _rewrite_tail_outline_for_branch(
            self,
            *,
            setting: Setting,
            outline_tail: StoryOutlineFull,
            branch: BranchSpec,
            char_list: List[str],
            loc_list: List[str],
            artifact_store: Optional[ArtifactStore] = None,
    ) -> StoryOutlineFull:
        model_name = self.router.get_model_for_agent("branch_rewriter_agent")

        payload = {
            "setting": setting.dict(),
            "tail_outline": outline_tail.dict(),
            "branch_context": {
                "branch_id": branch.id,
                "title": branch.title,
                "description": branch.description,
                "ending_tone": branch.ending_tone,
            },
            "char_list": char_list,
            "loc_list": loc_list,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=branch_tail_rewriter_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"branch_tail_rewriter_{branch.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"beats":[{"id":"beat_13","act":3,"order":13,"title":"...","summary":"...","tension_level":"high","purpose":"climax"}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"branch_tail_rewriter_{branch.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        beats_new = data.get("beats")
        if not isinstance(beats_new, list):
            return outline_tail

        old = {b.id: b for b in outline_tail.beats}
        out_beats: List[OutlineBeat] = []
        for b in beats_new:
            if not isinstance(b, dict):
                continue
            bid = str(b.get("id") or "")
            if bid not in old:
                continue
            base = old[bid]
            try:
                out_beats.append(
                    OutlineBeat(
                        id=base.id,
                        act=base.act,
                        order=base.order,
                        title=str(b.get("title") or base.title),
                        summary=str(b.get("summary") or base.summary),
                        tension_level=str(b.get("tension_level") or base.tension_level),
                        purpose=str(b.get("purpose") or base.purpose),
                    )
                )
            except Exception:
                out_beats.append(base)

        if len(out_beats) != len(outline_tail.beats):
            return outline_tail

        out = StoryOutlineFull(theory=outline_tail.theory, beats=sorted(out_beats, key=lambda x: x.order))
        return self._normalize_outline_order(out, artifact_store=artifact_store)




    @staticmethod
    def _has_travel_glue(lines: List[SceneLine], max_lines: int = 8) -> bool:
        if not lines:
            return False
        head = " ".join((ln.text or "").lower() for ln in lines[:max_lines] if ln and ln.text)
        patterns = [
            "спустя", "через", "по дороге", "путь", "мы шли", "мы брели", "мы добира",
            "добрались", "когда пришли", "на подходе", "на подступах", "мы вышли", "мы поднял", "мы спустил"
        ]
        return any(p in head for p in patterns)

    @staticmethod
    def _extract_location_aff(loc_affordances: Dict[str, Dict[str, Any]], loc: str) -> Dict[str, Any]:
        if not loc_affordances:
            return {"kind": "mixed", "enterable": True, "scale": "area", "notes": ""}
        v = loc_affordances.get(loc)
        if isinstance(v, dict):
            return {
                "kind": str(v.get("kind") or "mixed"),
                "enterable": bool(v.get("enterable", True)),
                "scale": str(v.get("scale") or "area"),
                "notes": str(v.get("notes") or ""),
            }
        return {"kind": "mixed", "enterable": True, "scale": "area", "notes": ""}

    @staticmethod
    def _is_transition_required(prev_location: Optional[str], cur_location: Optional[str]) -> bool:
        if not prev_location or not cur_location:
            return False
        return str(prev_location) != str(cur_location)

    def _build_thread_agenda(
        self,
        story_state: StoryState,
        beat_id: str,
        branch_id: str,
        branch_order: int,
        total_branch_scenes: int,
        *,
        max_open_cards: int = 8,
    ) -> Dict[str, Any]:

        remaining = max(0, int(total_branch_scenes) - int(branch_order))
        must_touch: List[str] = []
        should_touch: List[str] = []
        must_resolve: List[str] = []
        open_cards: List[Dict[str, Any]] = []

        pt = getattr(story_state, "plot_threads", None) or {}
        if isinstance(pt, dict):
            items = list(pt.items())
        else:
            items = []

        for tid, raw in items:
            tid_s = str(tid)
            t = self._obj_to_dict(raw)
            status = str(t.get("status") or "open")
            if status in {"resolved", "dropped"}:
                continue

            scope = str(t.get("branch_scope") or "global")
            if scope == "branch":
                b_id = t.get("branch_id")
                if b_id is not None and str(b_id) != str(branch_id):
                    continue

            priority = str(t.get("priority") or "major")
            anchors = [str(x) for x in (t.get("anchors") or [])]
            gap = int(branch_order) - int(t.get("last_touched_order") or 0)
            anchored_now = beat_id in anchors

            open_cards.append(
                {
                    "id": tid_s,
                    "title": str(t.get("title") or tid_s),
                    "description": str(t.get("description") or ""),
                    "priority": priority,
                    "status": status,
                    "closure_signal": str(t.get("closure_signal") or ""),
                    "can_remain_open": bool(t.get("can_remain_open", False)),
                    "last_touched_order": int(t.get("last_touched_order") or 0),
                }
            )

            if remaining <= 1 and (not bool(t.get("can_remain_open", False))) and priority in {"critical", "major"}:
                must_resolve.append(tid_s)
                continue

            starved = (
                (priority == "critical" and gap >= 3) or
                (priority == "major" and gap >= 5) or
                (priority == "minor" and gap >= 8)
            )
            if anchored_now or starved or remaining <= 2:
                must_touch.append(tid_s)
            elif gap >= 2:
                should_touch.append(tid_s)

        def pr_key(card: Dict[str, Any]) -> Tuple[int, int]:
            p = str(card.get("priority") or "major")
            pri = 0 if p == "critical" else (1 if p == "major" else 2)
            lto = int(card.get("last_touched_order") or 0)
            return (pri, lto)

        open_cards_sorted = sorted(open_cards, key=pr_key)
        open_cards = open_cards_sorted[:max_open_cards]

        return {
            "branch_id": str(branch_id),
            "branch_order": int(branch_order),
            "total_branch_scenes": int(total_branch_scenes),
            "remaining_scenes": int(remaining),
            "must_touch": must_touch[:4],
            "should_touch": should_touch[:4],
            "must_resolve": must_resolve[:3],
            "open_cards": open_cards,
            "forbid_new_major_threads": bool(remaining <= 2),
        }

    def _min_lines(self, story_length: str) -> int:
        key = f"MIN_SCENE_LINES_{(story_length or 'medium').upper()}"
        env = os.getenv(key)
        if env:
            try:
                return int(env)
            except Exception:
                pass
        return {"short": 30, "medium": 45, "long": 70}.get(story_length or "medium", 45)

    def _min_lines_for_scene(self, outline: StoryOutlineFull, contract: SceneContract, story_length: str) -> int:
        base = self._min_lines(story_length)
        beat_map = {b.id: b for b in (outline.beats or [])}
        beat = beat_map.get(contract.beat_id)
        purpose = self._normalize_purpose_value((beat.purpose if beat else "") or "setup")

        multipliers = {
            "introduction": 0.90,
            "setup": 0.92,
            "reaction": 0.80,
            "sequel": 0.82,
            "travel": 0.82,
            "rising_action": 1.00,
            "conflict": 1.00,
            "turning_point": 1.05,
            "midpoint": 1.12,
            "crisis": 1.20,
            "climax": 1.30,
            "resolution": 0.92,
            "epilogue": 0.80,
            "revelation": 1.00,
        }

        out = int(base * float(multipliers.get(purpose, 1.0)))
        return max(18, out)

    def _writer_max_tokens(self, story_length: str) -> int:
        env = os.getenv("WRITER_MAX_TOKENS")
        if env:
            try:
                return int(env)
            except Exception:
                pass
        return {"short": 8000, "medium": 14000, "long": 18000}.get(story_length or "medium", 14000)

    def _build_scene_context(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contract: SceneContract,
        char_appearance_map: Dict[str, str],
        previous_summaries: List[str],
        previous_last_lines: Optional[List[str]],
        story_state: StoryState,
        loc_graph: Optional[Any],
        loc_canons: Dict[str, str],
        loc_affordances: Dict[str, Dict[str, Any]],
        prev_location: Optional[str] = None,
        *,
        next_contract: Optional[SceneContract] = None,
        thread_agenda: Optional[Dict[str, Any]] = None,
        char_graph: Optional[Any] = None,
        choice_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        prev_beat: Optional[OutlineBeat] = None
        if beat is not None:
            beats_sorted = sorted(outline.beats, key=lambda b: b.order)
            for idx, b_ in enumerate(beats_sorted):
                if b_.id == beat.id and idx > 0:
                    prev_beat = beats_sorted[idx - 1]
                    break

        lines: List[str] = []
        lines.append("=== WORLD / SETTING ===")
        if setting.setting:
            lines.append(setting.setting)
        if setting.world_rules:
            lines.append("\nWORLD RULES:")
            lines.append(setting.world_rules)

        if prev_beat is not None:
            lines.append("\n=== PREVIOUS BEAT (TEXT) ===")
            if prev_beat.title:
                lines.append(f"Title: {prev_beat.title}")
            if prev_beat.summary:
                lines.append(prev_beat.summary)

        if beat is not None:
            lines.append("\n=== CURRENT BEAT (TEXT) ===")
            if beat.title:
                lines.append(f"Title: {beat.title}")
            if beat.summary:
                lines.append(beat.summary)

        lines.append("\n=== CURRENT SCENE PLAN ===")
        lines.append(f"Location: {scene_contract.location}")
        lines.append(f"POV character: {scene_contract.pov_character}")
        lines.append(f"Present characters: {', '.join(scene_contract.present_characters)}")
        if scene_contract.summary:
            lines.append(f"Planned scene summary: {scene_contract.summary}")
        if scene_contract.scene_goal:
            lines.append(f"Scene goal: {scene_contract.scene_goal}")
        if scene_contract.scene_conflict:
            lines.append(f"Scene conflict: {scene_contract.scene_conflict}")
        if scene_contract.stakes:
            lines.append(f"Stakes: {scene_contract.stakes}")
        if scene_contract.reveal:
            lines.append(f"Reveal / shift: {scene_contract.reveal}")
        if scene_contract.emotional_beat:
            lines.append(f"Emotional beat: {scene_contract.emotional_beat}")
        if scene_contract.must_reference:
            lines.append("Must reference: " + "; ".join(scene_contract.must_reference[:8]))
        if scene_contract.entry_requirements:
            lines.append("Entry requirements: " + "; ".join(scene_contract.entry_requirements[:8]))
        if scene_contract.exit_targets:
            lines.append("Exit targets: " + "; ".join(scene_contract.exit_targets[:8]))
        if scene_contract.continuity_notes:
            lines.append("Continuity notes: " + "; ".join(scene_contract.continuity_notes[:8]))
        if scene_contract.thread_focus:
            lines.append("Thread focus: " + ", ".join(scene_contract.thread_focus[:6]))

        transition_required = self._is_transition_required(prev_location, scene_contract.location)
        if transition_required:
            lines.append("\n=== TRANSITION REQUIRED ===")
            lines.append(f"Previous location: {prev_location}")
            lines.append("You MUST include travel/arrival glue in the first lines. No teleportation.")
            path_hint = self._shortest_loc_path(loc_graph, prev_location, scene_contract.location, max_depth=5)
            if path_hint and len(path_hint) > 1:
                lines.append(f"Plausible travel path: {' -> '.join(path_hint)}")

        if next_contract is not None:
            lines.append("\n=== NEXT SCENE HANDOFF ===")
            lines.append(f"Next location: {next_contract.location}")
            lines.append(f"Next POV candidate: {next_contract.pov_character}")
            if next_contract.summary:
                lines.append(f"Next scene target: {next_contract.summary}")

        if thread_agenda is not None:
            lines.append("\n=== THREAD AGENDA ===")
            lines.append(json.dumps(thread_agenda, ensure_ascii=False))

        selected_branch_option = None
        if isinstance(story_state.world, dict):
            selected_branch_option = story_state.world.get("selected_branch_option")

        if selected_branch_option is not None:
            lines.append("\n=== BRANCH ENTRY CHOICE ===")
            lines.append(json.dumps(selected_branch_option, ensure_ascii=False))
            lines.append(
                "IMPORTANT: The early scenes of this branch must feel like the immediate consequence of this exact chosen option."
            )

        if story_state.characters:
            lines.append("\n=== STORY STATE (CHAR LOCATIONS / MOODS) ===")
            for name in scene_contract.present_characters:
                st = story_state.characters.get(name) or {}
                if isinstance(st, dict) and (st.get("location") or st.get("mood")):
                    lines.append(f"- {name}: location={st.get('location')}, mood={st.get('mood')}")

        if loc_graph is not None:
            neigh = self._neighbors_from_loc_graph(loc_graph, scene_contract.location, limit=12)
            if neigh:
                lines.append("\n=== LOCATION GRAPH HINT ===")
                lines.append(f"Adjacent plausible locations from '{scene_contract.location}': {', '.join(neigh)}")

        canon = (loc_canons or {}).get(scene_contract.location, "") or ""
        aff = self._extract_location_aff(loc_affordances, scene_contract.location)
        lines.append("\n=== CURRENT LOCATION CANON (BG) ===")
        lines.append(f"[{scene_contract.location}] {canon}".strip())
        lines.append("\n=== LOCATION AFFORDANCES (HARD CONSTRAINTS) ===")
        lines.append(json.dumps(aff, ensure_ascii=False))

        if not aff.get("enterable", True) or str(aff.get("scale")) == "object":
            lines.append(
                "IMPORTANT: This location is NOT a walkable interior. Scene must be outside; no rooms/walls/ceilings/entering inside."
            )
        if choice_context is not None:
            lines.append("\n=== CHOICE CONTEXT ===")
            lines.append(json.dumps(choice_context, ensure_ascii=False))

            role = str(choice_context.get("scene_role") or "")
            if role == "setup":
                lines.append(
                    "IMPORTANT: This scene should PREPARE an upcoming decision by making the future options feel concrete, costly, and mutually incompatible."
                )
            elif role == "decision":
                lines.append(
                    "IMPORTANT: This is a DECISION SCENE. By the end, the protagonist must face a clear, motivated dilemma; each option should feel distinct and earned."
                )
        relation_notes = self._present_relationship_notes(char_graph, scene_contract.present_characters)
        if relation_notes:
            lines.append("\n=== PRESENT CHARACTER RELATIONSHIPS (FROM CHAR_GRAPH) ===")
            for r in relation_notes[:12]:
                lines.append(f"- {r}")

        ct = None
        if isinstance(story_state.world, dict):
            ct = story_state.world.get("char_type")

        if isinstance(ct, dict) and ct:
            role_by_char: Dict[str, str] = {}
            for role, names in ct.items():
                if isinstance(names, list):
                    for n in names:
                        role_by_char[str(n)] = str(role)
                elif isinstance(names, str):
                    role_by_char[str(names)] = str(role)

            lines.append("\n=== CHARACTER ROLES (CHAR_TYPE) ===")
            for name in scene_contract.present_characters:
                r = role_by_char.get(name)
                if r:
                    lines.append(f"- {name}: {r}")

        lines.append("\n=== CHARACTER NOTES (PRESENT IN SCENE) ===")
        for name in scene_contract.present_characters:
            desc = char_appearance_map.get(name, "")
            lines.append(f"{name}: {desc or '(no detailed description provided)'}")

        if previous_summaries:
            lines.append("\n=== RECENT SCENE SUMMARIES ===")
            for s in previous_summaries[-10:]:
                text = s
                if ":" in text:
                    text = text.split(":", 1)[1].strip()
                lines.append(f"- {text}")

        if previous_last_lines:
            lines.append("\n=== IMMEDIATE CONTEXT (PREVIOUS SCENE END) ===")
            lines.append("The previous scene ended with these exact lines. Continue smoothly:")
            for l in previous_last_lines:
                lines.append(f"> {l}")

        return "\n".join(lines)

    async def _build_advanced_rag_context(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contract: SceneContract,
        char_appearance_map: Dict[str, str],
        previous_summaries: List[str],
        base_context_text: str,
        rag_bundle: RAGBundle,
        story_state: StoryState,
        artifact_store: Optional[ArtifactStore] = None,
        *,
        thread_agenda: Optional[Dict[str, Any]] = None,
        next_contract: Optional[SceneContract] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_name = self.router.get_model_for_agent("outline_agent")

        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        query_parts: List[str] = []
        if scene_contract.summary:
            query_parts.append(scene_contract.summary)
        if beat and beat.summary:
            query_parts.append(beat.summary)

        query_parts.append(f"Location: {scene_contract.location}")
        if scene_contract.present_characters:
            query_parts.append(f"Characters: {', '.join(scene_contract.present_characters)}")

        if thread_agenda is not None:
            cards = thread_agenda.get("open_cards") or []
            sel = set((thread_agenda.get("must_touch") or []) + (thread_agenda.get("must_resolve") or []))
            for c in cards:
                if isinstance(c, dict) and str(c.get("id") or "") in sel:
                    title = str(c.get("title") or "")
                    desc = str(c.get("description") or "")
                    if title or desc:
                        query_parts.append(f"THREAD: {title}. {desc}".strip())

        query_text = "\n\n".join([p for p in query_parts if (p or "").strip()]).strip()

        q_tokens = RAGIndex._tokenize(query_text)
        q_emb: Optional[List[float]] = None
        if rag_bundle.embed_model and query_text:
            q_emb = await self.client.generate_embedding(
                text=query_text,
                model=rag_bundle.embed_model,
                operation_name="embed_scene_query_once",
            )

        story_items = await rag_bundle.story.query(
            query_text=query_text,
            top_k=8,
            kinds=["scene", "beat", "story_checkpoint"],
            q_tokens=q_tokens,
            q_emb=q_emb,
        )
        world_items = await rag_bundle.world.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)
        char_items = await rag_bundle.characters.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)
        thread_items = await rag_bundle.threads.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)

        retrieved_items = {"story": story_items, "world": world_items, "characters": char_items, "threads": thread_items}

        payload = {
            "setting": setting.dict(),
            "current_beat": beat.dict() if beat else None,
            "scene_contract": scene_contract.dict(),
            "next_contract": next_contract.dict() if next_contract else None,
            "thread_agenda": thread_agenda,
            "char_appearance_map": char_appearance_map,
            "previous_summaries": previous_summaries[-10:],
            "base_context_text": base_context_text,
            "retrieved_items": retrieved_items,
            "story_state": story_state.dict(),
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=rag_context_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"rag_context_{scene_contract.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"global_facts":"...","current_beat_facts":"...","character_facts":{},"recent_events":"...","open_threads":[]}'
        ctx = await self._parse_json_with_repair(
            raw,
            model_name,
            f"rag_context_{scene_contract.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        return ctx, retrieved_items

    async def _plan_scene_microplan(
            self,
            setting: Setting,
            outline: StoryOutlineFull,
            scene_contract: SceneContract,
            story_state: StoryState,
            retrieved_items: Dict[str, Any],
            branch_context: Optional[Dict[str, Any]],
            choice_context: Optional[Dict[str, Any]],
            artifact_store: Optional[ArtifactStore] = None,
            *,
            thread_agenda: Optional[Dict[str, Any]] = None,
            next_contract: Optional[SceneContract] = None,
    ) -> Dict[str, Any]:
        model_name = self.router.get_model_for_agent("scene_microplanner_agent")
        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        payload = {
            "setting": setting.dict(),
            "current_beat": beat.dict() if beat else None,
            "scene_contract": scene_contract.dict(),
            "next_contract": next_contract.dict() if next_contract else None,
            "thread_agenda": thread_agenda,
            "story_state": story_state.dict(),
            "retrieved_items": retrieved_items,
            "branch_context": branch_context,
            "choice_context": choice_context,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=scene_microplanner_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"scene_microplan_{scene_contract.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"microbeats":["..."],"must_hold_true":["..."],"must_touch_threads":["thread_01"],"required_mentions":[],"forbidden":[]}'
        return await self._parse_json_with_repair(
            raw,
            model_name,
            f"scene_microplan_{scene_contract.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

    async def _critique_scene(
        self,
        knowledge_context: Dict[str, Any],
        scene_script: SceneScript,
        scene_contract: SceneContract,
        story_state: StoryState,
        branch_context: Optional[Dict[str, Any]],
        choice_context: Optional[Dict[str, Any]],
        loc_graph: Optional[Any],
        char_graph: Optional[Any],
        microplan: Optional[Dict[str, Any]],
        story_length: str,
        prev_location: Optional[str],
        transition_required: bool,
        location_canon: str,
        location_affordances: Dict[str, Any],
        loc_list: List[str],
        artifact_store: Optional[ArtifactStore] = None,
        *,
        thread_agenda: Optional[Dict[str, Any]] = None,
        next_contract: Optional[SceneContract] = None,
    ) -> Dict[str, Any]:
        model_name = self.router.get_model_for_agent("critic_agent")
        payload = {
            "knowledge_context": knowledge_context,
            "scene_contract": scene_contract.dict(),
            "next_contract": next_contract.dict() if next_contract else None,
            "thread_agenda": thread_agenda,
            "scene_script": scene_script.dict(),
            "story_state": story_state.dict(),
            "branch_context": branch_context,
            "choice_context": choice_context,
            "loc_graph": loc_graph,
            "char_graph": char_graph,
            "microplan": microplan,
            "story_length": story_length,
            "prev_location": prev_location,
            "transition_required": transition_required,
            "location_canon": location_canon,
            "location_affordances": location_affordances,
            "loc_list": loc_list,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.1,
            system_prompt=critic_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"critique_{scene_script.scene_id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"ok":true,"issues":[],"must_regenerate":false,"state_updates":{"world":{},"characters":{},"plot_threads":{}}}'
        return await self._parse_json_with_repair(
            raw,
            model_name,
            f"critique_{scene_script.scene_id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

    def _apply_state_updates(self, story_state: StoryState, state_updates: Dict[str, Any], *, branch_order: Optional[int] = None, scene_id: Optional[str] = None) -> None:
        if not state_updates:
            return

        world_updates = state_updates.get("world") or {}
        if isinstance(world_updates, dict):
            for k, v in world_updates.items():
                story_state.world[k] = v

        char_updates = state_updates.get("characters") or {}
        if isinstance(char_updates, dict):
            for name, delta in char_updates.items():
                base = story_state.characters.get(name, {})
                if not isinstance(base, dict):
                    base = {}
                if isinstance(delta, dict):
                    base.update(delta)
                    story_state.characters[name] = base
                elif isinstance(delta, str):
                    base["note"] = delta
                    story_state.characters[name] = base

        thread_updates = state_updates.get("plot_threads") or {}
        if isinstance(thread_updates, dict):
            for tid, delta in thread_updates.items():
                tid_s = str(tid)
                if tid_s not in story_state.plot_threads:
                    app_logger.warning(f"Ignoring unknown thread update: {tid_s}")
                    continue

                base_raw = story_state.plot_threads.get(tid_s, {"id": tid_s, "status": "open"})
                base = self._obj_to_dict(base_raw)
                base.setdefault("id", tid_s)

                touched = False
                new_status: Optional[str] = None

                if isinstance(delta, str):
                    new_status = delta
                elif isinstance(delta, dict):
                    if delta.get("touched") is True:
                        touched = True
                    if "status" in delta and delta.get("status") is not None:
                        new_status = str(delta.get("status"))
                    for k, v in delta.items():
                        if k == "status":
                            continue
                        if v is not None:
                            base[k] = v

                if new_status:
                    base["status"] = new_status
                    touched = True

                if touched and branch_order is not None:
                    base["last_touched_order"] = int(branch_order)
                if touched and scene_id:
                    base["last_touched_scene_id"] = str(scene_id)

                if new_status in {"resolved", "dropped"} and scene_id:
                    base["resolved_in_scene_id"] = str(scene_id)

                story_state.plot_threads[tid_s] = base

    def _soft_update_character_locations(self, story_state: StoryState, contract: SceneContract) -> None:

        present = set(contract.present_characters or [])

        for name, st in story_state.characters.items():
            if isinstance(st, dict):
                st["on_stage"] = name in present

        for name in present:
            st = story_state.characters.get(name)
            if not isinstance(st, dict):
                st = {}
            st["location"] = contract.location
            st["last_seen_scene_id"] = contract.id
            st["last_seen_branch_order"] = contract.branch_order
            st["on_stage"] = True
            story_state.characters[name] = st

    async def _populate_rag_indices(
        self,
        rag_bundle: RAGBundle,
        *,
        char_list: List[str],
        char_graph: Any,
        char_appearance: Any,
        loc_list: List[str],
        loc_graph: Any,
        loc_canons: Dict[str, str],
        loc_affordances: Dict[str, Dict[str, Any]],
    ) -> None:
        descs: List[str] = []
        if isinstance(char_appearance, dict):
            descs = char_appearance.get("descriptions") or []
        else:
            descs = getattr(char_appearance, "descriptions", None) or []

        for idx, name in enumerate(char_list):
            desc = descs[idx] if idx < len(descs) else ""
            rels = self._present_relationship_notes(char_graph, char_list, limit=40)
            text = "\n".join(
                [
                    f"Character: {name}",
                    f"Appearance: {desc}",
                    "Relations:",
                    *rels,
                ]
            ).strip()
            await rag_bundle.characters.upsert_item(f"char::{name}", "character", text)

        for loc in loc_list:
            aff = loc_affordances.get(loc) or {}
            neighbors = self._neighbors_from_loc_graph(loc_graph, loc, limit=12)
            text = "\n".join(
                [
                    f"Location: {loc}",
                    f"Canon: {loc_canons.get(loc, '')}",
                    f"Affordances: {json.dumps(aff, ensure_ascii=False)}",
                    f"Neighbors: {', '.join(neighbors)}",
                ]
            ).strip()
            await rag_bundle.world.upsert_item(f"loc::{loc}", "world_lore", text)

    async def _upsert_story_checkpoint(
        self,
        rag_bundle: RAGBundle,
        *,
        branch_id: str,
        previous_summaries: List[str],
    ) -> None:
        every = self._story_checkpoint_every()
        if not previous_summaries or (len(previous_summaries) % every != 0):
            return

        chunk = previous_summaries[-every:]
        text = "\n".join(chunk).strip()
        if not text:
            return

        await rag_bundle.story.upsert_item(
            f"{branch_id}::checkpoint::{len(previous_summaries):04d}",
            "story_checkpoint",
            text[:5000],
        )

    async def _update_thread_index_from_state(
        self,
        rag_bundle: RAGBundle,
        story_state: StoryState,
        *,
        changed_thread_ids: Optional[List[str]] = None,
    ) -> None:
        pt = getattr(story_state, "plot_threads", None) or {}
        if not isinstance(pt, dict):
            return

        changed = set(str(x) for x in (changed_thread_ids or []))

        for tid, raw in pt.items():
            tid_s = str(tid)
            if changed and tid_s not in changed:
                continue

            t = self._obj_to_dict(raw)
            title = str(t.get("title") or tid_s)
            desc = str(t.get("description") or "")
            status = str(t.get("status") or "open")
            priority = str(t.get("priority") or "major")

            text = "\n".join(
                [
                    f"Thread: {tid_s}",
                    f"Title: {title}",
                    f"Description: {desc}",
                    f"Status: {status}",
                    f"Priority: {priority}",
                    f"Closure signal: {t.get('closure_signal') or ''}",
                    f"Last touched scene: {t.get('last_touched_scene_id') or ''}",
                    f"Resolved in scene: {t.get('resolved_in_scene_id') or ''}",
                ]
            ).strip()

            await rag_bundle.threads.upsert_item(f"{tid_s}::state", "thread", text[:5000])

    def _scene_capsule_text(
        self,
        contract: SceneContract,
        script: SceneScript,
        story_state: StoryState,
        *,
        thread_agenda: Optional[Dict[str, Any]] = None,
    ) -> str:
        open_threads: List[str] = []
        pt = getattr(story_state, "plot_threads", None) or {}
        if isinstance(pt, dict):
            for tid, raw in pt.items():
                t = self._obj_to_dict(raw)
                st = str(t.get("status") or "open")
                if st in {"resolved", "dropped"}:
                    continue
                pri = str(t.get("priority") or "")
                open_threads.append(f"{tid}:{st}:{pri}".strip(":"))
                if len(open_threads) >= 12:
                    break

        parts = [
            f"Scene: {contract.id} ({contract.branch_id}#{contract.branch_order})",
            f"Beat: {contract.beat_id}",
            f"Location: {contract.location}",
            f"POV: {contract.pov_character}",
            f"Present: {', '.join(contract.present_characters)}",
            f"Plan: {contract.summary}",
            f"Goal: {contract.scene_goal}" if contract.scene_goal else "",
            f"Conflict: {contract.scene_conflict}" if contract.scene_conflict else "",
            f"Stakes: {contract.stakes}" if contract.stakes else "",
            f"Reveal: {contract.reveal}" if contract.reveal else "",
            ("ThreadFocus: " + ", ".join(contract.thread_focus[:6])) if contract.thread_focus else "",
            f"Summary: {script.summary}",
        ]
        if thread_agenda:
            parts.append(f"ThreadAgenda.must_touch: {', '.join(thread_agenda.get('must_touch') or [])}")
            parts.append(f"ThreadAgenda.must_resolve: {', '.join(thread_agenda.get('must_resolve') or [])}")

        if open_threads:
            parts.append("OpenThreads: " + ", ".join(open_threads))

        mem = getattr(script, "memory", None) or {}
        if isinstance(mem, dict) and mem.get("thread_updates"):
            parts.append("ThreadUpdates: " + json.dumps(mem.get("thread_updates"), ensure_ascii=False)[:1500])

        return "\n".join([p for p in parts if (p or "").strip()])[:6500]

    async def _edit_scene(
        self,
        setting: Setting,
        scene_contract: SceneContract,
        story_state: StoryState,
        microplan: Dict[str, Any],
        critic_issues: List[str],
        scene_script: SceneScript,
        location_canon: str,
        location_affordances: Dict[str, Any],
        prev_location: Optional[str],
        transition_required: bool,
        target_min_lines: Optional[int] = None,
        artifact_store: Optional[ArtifactStore] = None,
        *,
        thread_agenda: Optional[Dict[str, Any]] = None,
        next_contract: Optional[SceneContract] = None,
        choice_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SceneScript]:
        model_name = self.router.get_model_for_agent("scene_editor_agent")

        payload = {
            "setting": setting.dict(),
            "scene_contract": scene_contract.dict(),
            "next_contract": next_contract.dict() if next_contract else None,
            "thread_agenda": thread_agenda,
            "story_state": story_state.dict(),
            "microplan": microplan,
            "critic_issues": critic_issues,
            "scene_script": scene_script.dict(),
            "target_min_lines": target_min_lines,
            "location_canon": location_canon,
            "location_affordances": location_affordances,
            "prev_location": prev_location,
            "transition_required": transition_required,
            "choice_context": choice_context,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=scene_editor_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"edit_scene_{scene_contract.id}",
            max_tokens=15000,
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{ "scene_id":"...", "lines":[{"type":"narration","speaker":null,"text":"..."}], "summary":"..." }'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"edit_scene_{scene_contract.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        if not isinstance(data, dict) or not data:
            return None

        try:
            edited = SceneScript(**data)
        except Exception:
            return None

        edited.scene_id = scene_contract.id
        edited.branch_id = scene_contract.branch_id
        edited.branch_order = scene_contract.branch_order
        return edited

    async def _write_single_scene_with_retries(
        self,
        contract: SceneContract,
        setting: Setting,
        outline: StoryOutlineFull,
        char_appearance_map: Dict[str, str],
        rag_bundle: RAGBundle,
        story_state: StoryState,
        previous_summaries: List[str],
        story_length: str,
        branching: Optional[BranchingInfo] = None,
        previous_last_lines: Optional[List[str]] = None,
        loc_graph: Optional[Any] = None,
        char_graph: Optional[Any] = None,
        max_retries: int = 3,
        artifact_store: Optional[ArtifactStore] = None,
        prev_location: Optional[str] = None,
        loc_canons: Optional[Dict[str, str]] = None,
        loc_affordances: Optional[Dict[str, Dict[str, Any]]] = None,
        loc_list: Optional[List[str]] = None,
        name_canon: Optional[NameCanonicalizer] = None,
        strict_names: Optional[bool] = None,
        strict_locations: Optional[bool] = None,
        *,
        next_contract: Optional[SceneContract] = None,
        total_branch_scenes: int = 1,
        choice_context: Optional[Dict[str, Any]] = None,
    ) -> SceneScript:
        writer_model = self.router.get_model_for_agent("writer_agent")
        min_lines = self._min_lines_for_scene(outline, contract, story_length)

        loc_canons = loc_canons or {}
        loc_affordances = loc_affordances or {}
        loc_list = loc_list or []

        if strict_names is None:
            strict_names = self._strict_name_canon_enabled()
        if strict_locations is None:
            strict_locations = self._strict_location_gate_enabled()

        strict_thread_closure = self._strict_thread_closure_enabled()

        try:
            num_candidates = int(os.getenv("SCENE_NUM_CANDIDATES", "2"))
        except Exception:
            num_candidates = 2
        num_candidates = max(1, min(num_candidates, 4))

        branch_spec: Optional[BranchSpec] = None
        if branching:
            branch_spec = next((b for b in branching.branches if b.id == contract.branch_id), None)

        branch_context_payload: Optional[Dict[str, Any]] = None
        if branch_spec:
            branch_context_payload = {
                "id": branch_spec.id,
                "title": branch_spec.title,
                "description": branch_spec.description,
                "ending_tone": branch_spec.ending_tone,
                "kind": branch_spec.kind,
                "is_canonical": branch_spec.is_canonical,
            }

        schema_hint_scene = '{ "scene_id":"scene_001", "lines":[{"type":"dialogue","speaker":"Имя","text":"..."}], "summary":"..." }'

        location_patch_budget = 1

        thread_agenda = self._build_thread_agenda(
            story_state=story_state,
            beat_id=contract.beat_id,
            branch_id=contract.branch_id,
            branch_order=contract.branch_order,
            total_branch_scenes=total_branch_scenes,
        )
        due_threads = set(str(x) for x in (thread_agenda.get("must_resolve") or []))

        for _ctx_rebuild in range(0, 2):
            transition_required = self._is_transition_required(prev_location, contract.location)
            location_canon = (loc_canons.get(contract.location) or "").strip()
            location_aff = self._extract_location_aff(loc_affordances, contract.location)

            base_context_text = self._build_scene_context(
                setting=setting,
                outline=outline,
                scene_contract=contract,
                char_appearance_map=char_appearance_map,
                previous_summaries=previous_summaries,
                previous_last_lines=previous_last_lines,
                story_state=story_state,
                loc_graph=loc_graph,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                prev_location=prev_location,
                next_contract=next_contract,
                thread_agenda=thread_agenda,
                char_graph=char_graph,
                choice_context=choice_context,
            )

            rag_context, retrieved_items = await self._build_advanced_rag_context(
                setting=setting,
                outline=outline,
                scene_contract=contract,
                char_appearance_map=char_appearance_map,
                previous_summaries=previous_summaries,
                base_context_text=base_context_text,
                rag_bundle=rag_bundle,
                story_state=story_state,
                artifact_store=artifact_store,
                thread_agenda=thread_agenda,
                next_contract=next_contract,
            )

            microplan = await self._plan_scene_microplan(
                setting=setting,
                outline=outline,
                scene_contract=contract,
                story_state=story_state,
                retrieved_items=retrieved_items,
                branch_context=branch_context_payload,
                choice_context=choice_context,
                artifact_store=artifact_store,
                thread_agenda=thread_agenda,
                next_contract=next_contract,
            )

            char_type_ctx = story_state.world.get("char_type") if isinstance(story_state.world, dict) else None

            combined_context = {
                "base_context_text": base_context_text,
                "rag_context": rag_context,
                "microplan": microplan,
                "location_canon": location_canon,
                "location_affordances": location_aff,
                "thread_agenda": thread_agenda,
                "next_contract": next_contract.dict() if next_contract else None,
                "char_type": char_type_ctx,
                "present_char_roles": self._present_char_roles(char_type_ctx, contract.present_characters),
                "choice_context": choice_context,
            }

            if artifact_store is not None:
                artifact_store.save(
                    f"context/{contract.branch_id}/{contract.id}.json",
                    {
                        "scene_contract": contract.dict(),
                        "combined_context": combined_context,
                        "min_lines": min_lines,
                        "prev_location": prev_location,
                        "transition_required": transition_required,
                        "strict_names": bool(strict_names),
                        "strict_locations": bool(strict_locations),
                        "strict_thread_closure": bool(strict_thread_closure),
                    },
                )

            last_reason = ""
            last_details = ""
            last_short_script: Optional[SceneScript] = None

            for attempt in range(1, max_retries + 1):
                regen_info: Optional[Dict[str, Any]] = None
                if attempt > 1:
                    regen_info = {"attempt": attempt, "reason": last_reason or "unknown", "details": last_details[:900]}

                candidates: List[SceneScript] = []
                cand_unmatched_names: Dict[int, bool] = {}

                for cidx in range(1, num_candidates + 1):
                    payload: Dict[str, Any] = {
                        "context": combined_context,
                        "scene_contract": contract.dict(),
                        "story_length": story_length,
                        "min_lines": min_lines,
                        "prev_location": prev_location,
                        "transition_required": transition_required,
                    }
                    if branch_context_payload is not None:
                        payload["branch_context"] = branch_context_payload
                    if regen_info is not None:
                        payload["regen_info"] = {**regen_info, "candidate_idx": cidx}
                    if last_short_script is not None:
                        payload["previous_scene_script"] = last_short_script.dict()

                    resp = await self.client.generate_completion(
                        model=writer_model,
                        temperature=0.7 if attempt == 1 else 0.6,
                        system_prompt=writer_prompt,
                        prompt=json.dumps(payload, ensure_ascii=False),
                        response_format={"type": "json_object"},
                        operation_name=f"write_scene_{contract.id}_a{attempt}_c{cidx}",
                        max_tokens=self._writer_max_tokens(story_length),
                    )

                    raw = (resp["choices"][0]["message"]["content"] or "").strip()
                    data = await self._parse_json_with_repair(
                        raw=raw,
                        model_name=writer_model,
                        operation_name=f"write_scene_{contract.id}_parse",
                        schema_hint=schema_hint_scene,
                        artifact_store=artifact_store,
                    )
                    if not isinstance(data, dict) or not data:
                        continue

                    try:
                        script = SceneScript(**data)
                    except ValidationError:
                        continue

                    script.scene_id = contract.id
                    script.branch_id = contract.branch_id
                    script.branch_order = contract.branch_order

                    for line in script.lines:
                        if line.type == "narration":
                            line.speaker = None
                        if line.type == "thought" and not line.speaker:
                            line.speaker = contract.pov_character

                    had_unmatched = False
                    if name_canon is not None:
                        had_unmatched = self._canon_script_inplace(script, contract, name_canon, store=artifact_store)
                    cand_unmatched_names[id(script)] = had_unmatched

                    if not isinstance(script.memory, dict):
                        script.memory = {}
                    script.memory.setdefault("thread_agenda", thread_agenda)
                    script.memory.setdefault("next_contract", next_contract.dict() if next_contract else None)

                    candidates.append(script)

                if not candidates:
                    last_reason = "no_valid_candidates"
                    last_details = f"attempt={attempt}: all candidates invalid (json/validation)"
                    continue

                scored: List[Tuple[float, SceneScript, Dict[str, Any], bool]] = []

                for script in candidates:
                    cr = await self._critique_scene(
                        knowledge_context=rag_context,
                        scene_script=script,
                        scene_contract=contract,
                        story_state=story_state,
                        branch_context=branch_context_payload,
                        choice_context=choice_context,
                        loc_graph=loc_graph,
                        char_graph=char_graph,
                        microplan=microplan,
                        story_length=story_length,
                        prev_location=prev_location,
                        transition_required=transition_required,
                        location_canon=location_canon,
                        location_affordances=location_aff,
                        loc_list=loc_list,
                        artifact_store=artifact_store,
                        thread_agenda=thread_agenda,
                        next_contract=next_contract,
                    )

                    unmatched_names = bool(cand_unmatched_names.get(id(script), False))
                    must_regen_llm = bool(cr.get("must_regenerate", False))

                    issues = cr.get("issues") or []
                    issues_list = issues if isinstance(issues, list) else [str(issues)]
                    issues_count = len(issues_list)

                    hard_issues: List[str] = []
                    must_regen_rules = False

                    if strict_names and unmatched_names:
                        must_regen_rules = True
                        hard_issues.append("Non-canonical speaker name detected")

                    missing_travel = transition_required and (not self._has_travel_glue(script.lines, max_lines=8))
                    if missing_travel:
                        hard_issues.append("Missing travel glue at scene start while location changed (no teleportation).")
                        if strict_locations:
                            must_regen_rules = True

                    if (not location_aff.get("enterable", True)) or (str(location_aff.get("scale")) == "object"):
                        head_text = " ".join((ln.text or "") for ln in script.lines[:30])
                        if self._indoor_markers_found(head_text):
                            hard_issues.append(
                                "Location is not enterable / scale=object but text implies interior (walls/rooms/entering inside)."
                            )
                            if strict_locations:
                                must_regen_rules = True

                    loc_check = cr.get("location_check") or {}
                    if isinstance(loc_check, dict) and bool(loc_check.get("mismatch", False)):
                        rec = str(loc_check.get("recommended_action") or "edit_text")
                        sugg = loc_check.get("suggested_location")
                        if strict_locations and rec != "change_location":
                            must_regen_rules = True
                            hard_issues.append(
                                f"Critic reports location mismatch (recommended_action={rec}); strict location gate requires regeneration."
                            )
                        if strict_locations and rec == "change_location" and (not (isinstance(sugg, str) and sugg in loc_list)):
                            must_regen_rules = True
                            hard_issues.append(
                                "Critic suggests changing location but suggested_location is missing/invalid; strict location gate requires regeneration."
                            )

                    resolved_due = False
                    thread_updates = ((cr.get("state_updates") or {}).get("plot_threads") or {})
                    if isinstance(thread_updates, dict) and due_threads:
                        for tid, delta in thread_updates.items():
                            tid_s = str(tid)
                            if tid_s not in due_threads:
                                continue
                            status = None
                            if isinstance(delta, dict):
                                status = delta.get("status")
                            else:
                                status = delta
                            if str(status or "") in {"resolved", "dropped"}:
                                resolved_due = True
                                break

                    if strict_thread_closure and due_threads and (not resolved_due):
                        hard_issues.append(f"Due threads not resolved: {sorted(due_threads)[:6]}")
                        must_regen_rules = True

                    must_regen_effective = must_regen_llm or must_regen_rules

                    score = 100.0
                    if must_regen_effective:
                        score -= 60.0
                    score -= 6.0 * issues_count
                    score -= 18.0 * len(hard_issues)

                    score += min(20.0, len(script.lines) / 10.0)
                    if len(script.lines) < min_lines:
                        score -= min(30.0, float(min_lines - len(script.lines)) * 0.6)

                    if hard_issues:
                        cr["_hard_issues"] = hard_issues
                    cr["_must_regen_rules"] = must_regen_rules
                    cr["_must_regenerate_effective"] = must_regen_effective

                    scored.append((score, script, cr, must_regen_effective))

                scored.sort(key=lambda x: x[0], reverse=True)
                best_score, best_script, best_critic, best_must_regen = scored[0]

                if artifact_store is not None:
                    artifact_store.save(
                        f"critics/{contract.branch_id}/{contract.id}_a{attempt}.json",
                        {
                            "attempt": attempt,
                            "best_score": best_score,
                            "best_script_lines": len(best_script.lines),
                            "best_critic": best_critic,
                            "due_threads": sorted(list(due_threads))[:50],
                        },
                    )

                loc_check_best = best_critic.get("location_check") or {}
                if isinstance(loc_check_best, dict) and bool(loc_check_best.get("mismatch", False)):
                    rec = str(loc_check_best.get("recommended_action") or "edit_text")
                    sugg = loc_check_best.get("suggested_location")
                    if rec == "change_location" and location_patch_budget > 0 and isinstance(sugg, str) and sugg in loc_list:
                        if artifact_store is not None:
                            artifact_store.event(
                                "contract.location_patched",
                                {
                                    "scene_id": contract.id,
                                    "old_location": contract.location,
                                    "new_location": sugg,
                                    "reason": loc_check_best.get("details"),
                                },
                            )
                        contract.location = sugg
                        location_patch_budget -= 1
                        break

                if best_must_regen and attempt < max_retries:
                    last_reason = "critic_or_rules_feedback"
                    llm_issues = best_critic.get("issues") or []
                    llm_issues_list = llm_issues if isinstance(llm_issues, list) else [str(llm_issues)]
                    hard_issues = best_critic.get("_hard_issues") or []
                    hard_issues_list = hard_issues if isinstance(hard_issues, list) else [str(hard_issues)]
                    last_details = "; ".join((llm_issues_list + hard_issues_list))[:900]
                    continue

                final_script = best_script
                final_critic = best_critic

                if len(final_script.lines) < min_lines:
                    issues_list = final_critic.get("issues") or []
                    if not isinstance(issues_list, list):
                        issues_list = [str(issues_list)]
                    edited = await self._edit_scene(
                        setting=setting,
                        scene_contract=contract,
                        story_state=story_state,
                        microplan=microplan,
                        critic_issues=issues_list,
                        scene_script=final_script,
                        location_canon=location_canon,
                        location_affordances=location_aff,
                        prev_location=prev_location,
                        transition_required=transition_required,
                        target_min_lines=min_lines,
                        artifact_store=artifact_store,
                        thread_agenda=thread_agenda,
                        next_contract=next_contract,
                        choice_context=choice_context,
                    )
                    if edited is not None and len(edited.lines) >= len(final_script.lines):
                        if name_canon is not None:
                            self._canon_script_inplace(edited, contract, name_canon, store=artifact_store)
                        final_script = edited

                hard_issues2 = final_critic.get("_hard_issues") or []
                if hard_issues2:
                    edited = await self._edit_scene(
                        setting=setting,
                        scene_contract=contract,
                        story_state=story_state,
                        microplan=microplan,
                        critic_issues=list(hard_issues2) if isinstance(hard_issues2, list) else [str(hard_issues2)],
                        scene_script=final_script,
                        location_canon=location_canon,
                        location_affordances=location_aff,
                        prev_location=prev_location,
                        transition_required=transition_required,
                        target_min_lines=min_lines if len(final_script.lines) < min_lines else None,
                        artifact_store=artifact_store,
                        thread_agenda=thread_agenda,
                        next_contract=next_contract,
                        choice_context=choice_context,
                    )
                    if edited is not None:
                        if name_canon is not None:
                            self._canon_script_inplace(edited, contract, name_canon, store=artifact_store)
                        final_script = edited

                if len(final_script.lines) < min_lines and attempt < max_retries:
                    last_reason = "too_short"
                    last_details = f"lines={len(final_script.lines)} < min_lines={min_lines}"
                    last_short_script = final_script
                    continue

                cr2 = await self._critique_scene(
                    knowledge_context=rag_context,
                    scene_script=final_script,
                    scene_contract=contract,
                    story_state=story_state,
                    branch_context=branch_context_payload,
                    choice_context=choice_context,
                    loc_graph=loc_graph,
                    char_graph=char_graph,
                    microplan=microplan,
                    story_length=story_length,
                    prev_location=prev_location,
                    transition_required=transition_required,
                    location_canon=location_canon,
                    location_affordances=location_aff,
                    loc_list=loc_list,
                    artifact_store=artifact_store,
                    thread_agenda=thread_agenda,
                    next_contract=next_contract,
                )

                if bool(cr2.get("must_regenerate", False)) and attempt < max_retries:
                    issues3 = cr2.get("issues") or []
                    if not isinstance(issues3, list):
                        issues3 = [str(issues3)]
                    last_reason = "final_critique_failed"
                    last_details = "; ".join(issues3)[:900]
                    last_short_script = final_script
                    continue

                state_updates = cr2.get("state_updates") or {}
                thread_updates2 = (state_updates.get("plot_threads") or {}) if isinstance(state_updates, dict) else {}
                changed_thread_ids = list(thread_updates2.keys()) if isinstance(thread_updates2, dict) else []

                self._apply_state_updates(
                    story_state,
                    state_updates,
                    branch_order=contract.branch_order,
                    scene_id=contract.id,
                )
                self._soft_update_character_locations(story_state, contract)


                if not isinstance(final_script.memory, dict):
                    final_script.memory = {}
                final_script.memory["thread_agenda"] = thread_agenda
                final_script.memory["thread_updates"] = thread_updates2 if isinstance(thread_updates2, dict) else {}
                final_script.memory["state_updates"] = state_updates if isinstance(state_updates, dict) else {}
                final_script.memory["prev_location"] = prev_location
                final_script.memory["transition_required"] = transition_required
                final_script.memory["location"] = contract.location
                final_script.memory["next_location"] = next_contract.location if next_contract else None

                try:
                    await self._update_thread_index_from_state(rag_bundle, story_state, changed_thread_ids=changed_thread_ids)
                except Exception as e:
                    if artifact_store is not None:
                        artifact_store.event("threads.rag_update_failed", {"scene_id": contract.id, "error": str(e)})

                return final_script

            continue

        fb_text = "Техническая сцена-заглушка: генерация не смогла стабильно сформировать сцену."
        if artifact_store is not None:
            artifact_store.event(
                "scene.fallback",
                {"scene_id": contract.id, "branch_id": contract.branch_id, "reason": "exhausted_retries"},
            )

        return SceneScript(
            scene_id=contract.id,
            branch_id=contract.branch_id,
            branch_order=contract.branch_order,
            lines=[SceneLine(type="narration", speaker=None, text=fb_text)],
            summary=contract.summary or fb_text,
            memory={"thread_agenda": thread_agenda},
        )

    async def _write_scenes(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contracts: List[SceneContract],
        char_list: List[str],
        char_appearance: Dict[str, Any] | CharacterAppearance | None,
        rag_bundle: RAGBundle,
        story_state: StoryState,
        story_length: str,
        branching: Optional[BranchingInfo] = None,
        initial_previous_summaries: Optional[List[str]] = None,
        initial_previous_last_lines: Optional[List[str]] = None,
        initial_prev_location: Optional[str] = None,
        state_snapshots: Optional[Dict[str, StoryState]] = None,
        loc_graph: Optional[Any] = None,
        char_graph: Optional[Any] = None,
        loc_canons: Optional[Dict[str, str]] = None,
        loc_affordances: Optional[Dict[str, Dict[str, Any]]] = None,
        loc_list: Optional[List[str]] = None,
        artifact_store: Optional[ArtifactStore] = None,
        artifact_prefix: str = "main",
        name_canon: Optional[NameCanonicalizer] = None,
        strict_names: Optional[bool] = None,
        strict_locations: Optional[bool] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        generation_id: Optional[str] = None,
        progress_stage: str = "writing_scenes",
        progress_start: int = 0,
        progress_end: int = 0,
        choice_context_by_scene: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, SceneScript]:
        app_logger.info(f"Writing {len(scene_contracts)} scenes... story_length={story_length}")

        loc_canons = loc_canons or {}
        loc_affordances = loc_affordances or {}
        loc_list = loc_list or []

        if strict_names is None:
            strict_names = self._strict_name_canon_enabled()
        if strict_locations is None:
            strict_locations = self._strict_location_gate_enabled()

        if isinstance(char_appearance, CharacterAppearance):
            char_appearance_model = char_appearance
        elif isinstance(char_appearance, dict) and "descriptions" in char_appearance:
            char_appearance_model = CharacterAppearance(**char_appearance)
        else:
            char_appearance_model = CharacterAppearance(descriptions=[""] * len(char_list))

        char_appearance_map: Dict[str, str] = {}
        for name, desc in zip(char_list, char_appearance_model.descriptions):
            char_appearance_map[name] = desc

        scene_scripts: Dict[str, SceneScript] = {}
        previous_summaries: List[str] = []

        if initial_previous_summaries:
            history_embed_last = int(os.getenv("BRANCH_HISTORY_EMBED_LAST", "12") or "12")
            history_embed_last = max(4, min(history_embed_last, 40))

            for idx, entry in enumerate(initial_previous_summaries):
                previous_summaries.append(entry)
                if len(initial_previous_summaries) - idx <= history_embed_last:
                    await rag_bundle.story.add_item(_sha1(entry), "scene", entry[:2500])

            hist = "\n".join(initial_previous_summaries[-min(len(initial_previous_summaries), history_embed_last * 2):]).strip()
            if hist:
                await rag_bundle.story.upsert_item(f"{artifact_prefix}::history_bootstrap", "story_checkpoint", hist[:6000])

        last_scene_lines_buffer: List[str] = list(initial_previous_last_lines or [])
        prev_location: Optional[str] = initial_prev_location

        total_branch_scenes = len(scene_contracts)

        for idx, contract in enumerate(scene_contracts):
            next_contract = scene_contracts[idx + 1] if idx + 1 < len(scene_contracts) else None

            if name_canon is not None:
                self._canon_contract_inplace(contract, name_canon, store=artifact_store)

            if artifact_store is not None:
                artifact_store.event(
                    "scene.start",
                    {
                        "scene_id": contract.id,
                        "branch_id": contract.branch_id,
                        "order": contract.branch_order,
                        "summary_plan": contract.summary,
                        "location": contract.location,
                        "prev_location": prev_location,
                    },
                )
            scene_choice_context = (choice_context_by_scene or {}).get(contract.id)

            script = await self._write_single_scene_with_retries(
                contract=contract,
                setting=setting,
                outline=outline,
                char_appearance_map=char_appearance_map,
                rag_bundle=rag_bundle,
                story_state=story_state,
                previous_summaries=previous_summaries,
                story_length=story_length,
                branching=branching,
                previous_last_lines=last_scene_lines_buffer,
                loc_graph=loc_graph,
                char_graph=char_graph,
                max_retries=3,
                artifact_store=artifact_store,
                prev_location=prev_location,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_list=loc_list,
                name_canon=name_canon,
                strict_names=strict_names,
                strict_locations=strict_locations,
                next_contract=next_contract,
                total_branch_scenes=total_branch_scenes,
                choice_context=scene_choice_context,
            )

            scene_scripts[contract.id] = script

            thread_agenda = (script.memory or {}).get("thread_agenda") if isinstance(script.memory, dict) else None

            previous_summaries.append(f"{contract.id}: {script.summary}")

            capsule = self._scene_capsule_text(contract, script, story_state, thread_agenda=thread_agenda if isinstance(thread_agenda, dict) else None)
            await rag_bundle.story.upsert_item(contract.id, "scene", capsule)

            await self._upsert_story_checkpoint(rag_bundle, branch_id=contract.branch_id, previous_summaries=previous_summaries)

            if state_snapshots is not None:
                state_snapshots[contract.id] = StoryState(**story_state.dict())

            last_scene_lines_buffer = self._extract_last_lines(script, n=3)
            prev_location = contract.location

            app_logger.info(f"Scene {contract.id} written, lines={len(script.lines)}")
            if progress_callback is not None and generation_id is not None:
                if len(scene_contracts) > 0 and progress_end >= progress_start:
                    pct = progress_start + int(
                        ((idx + 1) / len(scene_contracts)) * (progress_end - progress_start)
                    )
                else:
                    pct = progress_end or progress_start

                await self._emit_progress(
                    progress_callback,
                    generation_id=generation_id,
                    stage=progress_stage,
                    message=f"{progress_stage}: scene {idx + 1}/{len(scene_contracts)} written",
                    percent=pct,
                    extra={
                        "scene_id": contract.id,
                        "scene_index": idx + 1,
                        "scene_total": len(scene_contracts),
                        "branch_id": contract.branch_id,
                    },
                )

            if artifact_store is not None:
                artifact_store.save(f"scenes/{artifact_prefix}/{contract.id}.json", script.dict())
                artifact_store.save(f"state/{artifact_prefix}/{contract.id}.json", story_state.dict())
                artifact_store.event(
                    "scene.done",
                    {"scene_id": contract.id, "branch_id": contract.branch_id, "lines": len(script.lines)},
                )

        return scene_scripts

    @staticmethod
    def _web_export_to_image_lists(
        web_export: Optional[Dict[str, Any]],
    ) -> Tuple[List[CharacterImage], List[LocationImage]]:
        if not isinstance(web_export, dict):
            return [], []

        out_dir_raw = str(web_export.get("out_dir") or "").strip()
        out_dir = Path(out_dir_raw) if out_dir_raw else None

        def pick_path(item: Dict[str, Any]) -> str:
            absolute_path = str(item.get("absolute_path") or "").strip()
            if absolute_path:
                return absolute_path

            web_path = str(item.get("web_path") or "").strip()
            if web_path and out_dir is not None:
                return str((out_dir / web_path).resolve())

            return web_path

        assets = web_export.get("assets") or {}
        char_assets = assets.get("characters") or {}
        loc_assets = assets.get("locations") or {}

        char_images: List[CharacterImage] = []
        for idx, (name, item) in enumerate(char_assets.items(), start=1):
            if not isinstance(item, dict):
                continue
            path = pick_path(item)
            if not path:
                continue
            char_images.append(
                CharacterImage(
                    id=f"char_{idx:03d}",
                    character=str(name),
                    path=path,
                    aspect_ratio="1:1",
                )
            )

        loc_images: List[LocationImage] = []
        for idx, (name, item) in enumerate(loc_assets.items(), start=1):
            if not isinstance(item, dict):
                continue
            path = pick_path(item)
            if not path:
                continue
            loc_images.append(
                LocationImage(
                    id=f"bg_{idx:03d}",
                    location=str(name),
                    path=path,
                    aspect_ratio="16:9",
                )
            )

        return char_images, loc_images

    async def generate_vn(
        self,
        user_prompt: str,
        story_length: str = "medium",
        char_list: list | None = None,
        loc_list: list | None = None,
        setting: str | None = "",
        max_branches: int | None = None,
        tone: str | None = None,
        artstyle: str | None = None,
        generate_images: bool = True,
        time_choice: Optional[str] = None,
        genre_choice: Optional[str] = None,
        tone_choice_ru: Optional[str] = None,
        mc_name: Optional[str] = None,
        mc_description: Optional[str] = None,
        extra_character_names: Optional[List[str]] = None,
        plot_prefs: Optional[PlotPreferences] = None,
        plot_freeform: Optional[str] = None,
        graphic_style_ru: Optional[str] = None,
        generation_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        generation_id = generation_id or f"vn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
        app_logger.info(f"Starting generation {generation_id}")
        await self._emit_progress(
            progress_callback,
            generation_id=generation_id,
            stage="start",
            message="Generation started",
            percent=1,
        )

        seed = int(hashlib.sha256(generation_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)

        output_root = Path(os.getenv("OUTPUT_DIR", "output")) / generation_id
        store = ArtifactStore(output_root)

        token = TRACE_HOOK.set(lambda evt: store.event(evt.get("type", "trace"), evt))

        try:
            def map_tone_ru_to_en(val: Optional[str]) -> Optional[str]:
                if not val:
                    return None
                m = {"веселый": "light", "грустный": "dark"}
                return m.get(val.strip().lower())

            def map_graphic_style(val: Optional[str]) -> Optional[str]:
                if not val:
                    return None
                m = {"аниме": "anime", "реализм": "realistic", "рисованная графика": "illustrative"}
                return m.get(val.strip().lower())

            explicit_tone = map_tone_ru_to_en(tone_choice_ru) or tone
            explicit_artstyle = map_graphic_style(graphic_style_ru) or artstyle

            store.checkpoint(
                "00_start_request",
                {
                    "generation_id": generation_id,
                    "seed": seed,
                    "user_prompt": user_prompt,
                    "story_length": story_length,
                    "max_branches": max_branches,
                    "generate_images": generate_images,
                    "strict_name_canon": self._strict_name_canon_enabled(),
                    "strict_location_gate": self._strict_location_gate_enabled(),
                    "strict_thread_closure": self._strict_thread_closure_enabled(),
                    "story_checkpoint_every": self._story_checkpoint_every(),
                },
            )

            user_request = await self._normalize_user_request(
                raw_prompt=user_prompt,
                explicit_length=story_length,
                explicit_tone=explicit_tone,
                explicit_artstyle=explicit_artstyle,
                explicit_max_branches=max_branches,
                artifact_store=store,
            )
            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="user_request_done",
                message="User request normalized",
                percent=5,
            )

            normalized_length = user_request.story_length or "medium"
            branches_max = user_request.max_branches or 1

            embed_model = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
            rag_main = RAGBundle(self.client, embed_model)
            story_state_main = StoryState()

            setting_obj = await self._generate_setting(
                user_prompt=user_request.user_prompt,
                setting_override=setting if setting else None,
                time_choice=time_choice,
                genre_choice=genre_choice,
                artifact_store=store,
            )

            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="setting_agent_done",
                message="Setting generated",
                percent=12,
            )

            world_text = setting_obj.setting or ""
            if setting_obj.world_rules:
                world_text += "\nWorld rules: " + setting_obj.world_rules
            await rag_main.world.upsert_item("setting_core", "world_lore", world_text[:8000])

            outline_obj = await self._generate_outline(
                user_prompt=user_request.user_prompt,
                story_length=normalized_length,
                setting=setting_obj,
                plot_prefs=plot_prefs,
                plot_freeform=plot_freeform,
                artifact_store=store,
            )
            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="outline_agent_done",
                message="Outline generated",
                percent=22,
            )


            threads_base_index: Optional[RAGIndex] = None
            threads: List[Dict[str, Any]] = []
            try:
                threads = await self._extract_plot_threads(
                    user_request.user_prompt,
                    setting_obj,
                    outline_obj,
                    artifact_store=store,
                )
                for t in threads:
                    tid = str(t.get("id"))
                    story_state_main.plot_threads[tid] = {
                        "id": tid,
                        "title": t.get("title") or tid,
                        "description": t.get("description") or "",
                        "status": t.get("status") or "open",
                        "priority": t.get("priority") or "major",
                        "anchors": t.get("anchors") or [],
                        "closure_signal": t.get("closure_signal") or "",
                        "can_remain_open": bool(t.get("can_remain_open", False)),
                        "branch_scope": t.get("branch_scope") or "global",
                        "branch_id": t.get("branch_id"),
                        "last_touched_scene_id": None,
                        "last_touched_order": 0,
                        "resolved_in_scene_id": None,
                    }
                    await rag_main.threads.upsert_item(
                        f"{tid}::core",
                        "thread",
                        f"{t.get('title','')}\n{t.get('description','')}\nanchors: {', '.join(t.get('anchors') or [])}",
                    )

                await self._update_thread_index_from_state(rag_main, story_state_main, changed_thread_ids=None)
                threads_base_index = rag_main.threads.clone()
                await self._emit_progress(
                    progress_callback,
                    generation_id=generation_id,
                    stage="plot_threads_done",
                    message="Plot threads extracted",
                    percent=28,
                )
            except Exception as e:
                app_logger.error(f"Thread extraction failed: {e}", exc_info=True)
                threads_base_index = rag_main.threads.clone()

            preferred_endings = plot_prefs.ending_types if plot_prefs and plot_prefs.ending_types else None
            branching_info = await self._plan_branches(
                outline=outline_obj,
                max_branches=branches_max,
                tone=user_request.tone,
                preferred_ending_types=preferred_endings,
                artifact_store=store,
            )

            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="branch_planner_done",
                message="Branch plan generated",
                percent=35,
            )

            seed_char_list: Optional[List[str]] = None
            if mc_name:
                seed_char_list = [mc_name]
                if extra_character_names:
                    seed_char_list.extend(extra_character_names)

            effective_char_list = char_list or seed_char_list
            char_list_final = await self._ensure_char_list(user_request.user_prompt, outline_obj, effective_char_list)

            if mc_name and mc_name in char_list_final:
                char_list_final = [mc_name] + [n for n in char_list_final if n != mc_name]

            loc_list_final = await self._ensure_loc_list(user_request.user_prompt, outline_obj, loc_list)

            limit_user = str(os.getenv("LIMIT_USER_LOC_LIST", "false")).strip().lower() in ("1", "true", "yes", "on")
            is_user_loc_list = loc_list is not None
            if (not is_user_loc_list) or limit_user:
                loc_list_final = self._limit_locations(
                    candidates=loc_list_final,
                    user_prompt=user_request.user_prompt,
                    setting=setting_obj,
                    outline=outline_obj,
                    story_length=normalized_length,
                    artifact_store=store,
                    reason="user_loc_list" if is_user_loc_list else "auto_loc_list",
                )

            name_canon = NameCanonicalizer(characters=char_list_final, locations=loc_list_final)
            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="names_ready",
                message="Character and location names prepared",
                percent=40,
            )

            hints: Dict[str, str] = {}
            if mc_name and mc_description:
                hints[mc_name] = mc_description

            char_result = await self._run_char_agent(char_list_final, setting_obj, hints=hints or None)
            char_results_map = char_result.get("results", {}) or {}
            char_graph = self._unwrap_last(char_results_map.get("char_graph"))
            char_appearance = self._unwrap_last(char_results_map.get("char_appearance"))
            char_type_raw = self._unwrap_last(char_results_map.get("char_type"))

            char_type = self._normalize_char_type(
                char_type_raw,
                char_list_final,
                canon=name_canon,
                protagonist=(
                    mc_name if mc_name in char_list_final else (char_list_final[0] if char_list_final else None)),
            )

            story_state_main.world["char_type"] = char_type
            char_appearance = await self._ensure_char_appearance_complete(
                char_list=char_list_final,
                char_appearance=char_appearance,
                setting=setting_obj,
                hints=hints or {},
                artifact_store=store,
            )

            if hasattr(char_graph, "model_dump"):
                char_graph = char_graph.model_dump()
            if hasattr(char_appearance, "model_dump"):
                char_appearance = char_appearance.model_dump()

            store.checkpoint(
                "06_characters",
                {
                    "char_list": char_list_final,
                    "char_graph": char_graph,
                    "char_appearance": char_appearance,
                    "char_type": char_type,
                },
            )
            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="char_agent_done",
                message="Characters generated",
                percent=50,
            )

            loc_result = await self._run_loc_agent(loc_list_final, setting_obj)
            loc_results_map = loc_result.get("results", {}) or {}
            loc_graph = self._unwrap_last(loc_results_map.get("loc_graph"))
            loc_description = self._unwrap_last(loc_results_map.get("loc_description"))

            loc_description = await self._ensure_loc_description_complete(
                loc_list=loc_list_final,
                loc_description=loc_description,
                setting=setting_obj,
                artifact_store=store,
            )

            if hasattr(loc_graph, "model_dump"):
                loc_graph = loc_graph.model_dump()

            store.checkpoint(
                "07_locations",
                {"loc_list": loc_list_final, "loc_graph": loc_graph, "loc_description": loc_description},
            )

            loc_canons = self._build_loc_desc_map(loc_list_final, loc_description)
            loc_affordances = await self._infer_location_affordances(setting_obj, loc_list_final, loc_canons, artifact_store=store)
            store.checkpoint("07c_location_kb", {"loc_canons": loc_canons, "loc_affordances": loc_affordances})
            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="loc_agent_done",
                message="Locations generated",
                percent=60,
            )

            try:
                await self._populate_rag_indices(
                    rag_main,
                    char_list=char_list_final,
                    char_graph=char_graph,
                    char_appearance=char_appearance,
                    loc_list=loc_list_final,
                    loc_graph=loc_graph,
                    loc_canons=loc_canons,
                    loc_affordances=loc_affordances,
                )
                store.event("rag.indices_populated", {"ok": True})
            except Exception as e:
                store.event("rag.indices_populated", {"ok": False, "error": str(e)})
                app_logger.error(f"RAG population failed: {e}", exc_info=True)

            char_images: List[CharacterImage] = []
            loc_images: List[LocationImage] = []
            web_export: Optional[Dict[str, Any]] = None

            outline_main_obj = self._beats_for_main_route(outline_obj, branching_info)
            for beat in outline_main_obj.beats:
                await rag_main.story.upsert_item(beat.id, "beat", (beat.summary or "")[:2500])

            main_contracts = await self._generate_scene_contracts_main(
                outline_main_obj,
                char_list_final,
                loc_list_final,
                normalized_length,
                char_type=char_type,
                branching=branching_info,
                plot_threads=threads,
                artifact_store=store,
            )
            main_contracts = self._sanitize_contracts_schema(
                main_contracts,
                char_list=char_list_final,
                loc_list=loc_list_final,
            )

            main_contracts = await self._patch_scene_contracts_with_location_critic(
                scene_contracts=main_contracts,
                loc_list=loc_list_final,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_graph=loc_graph,
                plot_threads=threads,
                artifact_store=store,
                artifact_name="main",
            )

            main_contracts = await self._patch_scene_contracts_with_consistency_critic(
                scene_contracts=main_contracts,
                setting=setting_obj,
                char_list=char_list_final,
                loc_list=loc_list_final,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_graph=loc_graph,
                plot_threads=threads,
                artifact_store=store,
                artifact_name="main",
                branch_context=None,
            )

            main_contracts = self._sanitize_contracts_schema(
                main_contracts,
                char_list=char_list_final,
                loc_list=loc_list_final,
            )
            for c in main_contracts:
                self._canon_contract_inplace(c, name_canon, store=store)

            main_contracts = await self._re_enrich_scene_contracts(
                setting=setting_obj,
                outline=outline_main_obj,
                scene_contracts=main_contracts,
                plot_threads=threads,
                artifact_store=store,
                artifact_name="main",
                branch_context=None,
            )

            choice_contracts: List[ChoiceContract] = []
            choice_context_by_scene: Dict[str, Dict[str, Any]] = {}
            planned_branch_outlines: Dict[str, StoryOutlineFull] = {}
            branch_contracts_by_branch: Dict[str, List[SceneContract]] = {}
            branch_contracts: List[SceneContract] = []

            beat_to_main_scene: Dict[str, str] = {sc.beat_id: sc.id for sc in main_contracts}
            main_contract_by_beat: Dict[str, SceneContract] = {sc.beat_id: sc for sc in main_contracts}

            for br in branching_info.branches:
                if br.id == "main":
                    continue
                if br.from_beat_id and br.from_beat_id in beat_to_main_scene:
                    br.from_scene_id = beat_to_main_scene[br.from_beat_id]

            if len(branching_info.branches) > 1:
                for br in branching_info.branches:
                    if br.id == "main" or not br.from_beat_id:
                        continue

                    divergence_contract = main_contract_by_beat.get(br.from_beat_id)

                    br_outline, br_contracts_for_branch = await self._generate_scene_contracts_for_branch(
                        outline_obj,
                        char_list_final,
                        loc_list_final,
                        normalized_length,
                        br,
                        setting_obj,
                        char_type=char_type,
                        artifact_store=store,
                        divergence_scene_contract=divergence_contract,
                        plot_threads=threads,
                    )
                    if not br_contracts_for_branch:
                        continue

                    br_contracts_for_branch = self._sanitize_contracts_schema(
                        br_contracts_for_branch,
                        char_list=char_list_final,
                        loc_list=loc_list_final,
                    )

                    br_contracts_for_branch = await self._patch_scene_contracts_with_location_critic(
                        scene_contracts=br_contracts_for_branch,
                        loc_list=loc_list_final,
                        loc_canons=loc_canons,
                        loc_affordances=loc_affordances,
                        loc_graph=loc_graph,
                        plot_threads=threads,
                        artifact_store=store,
                        artifact_name=br.id,
                    )

                    branch_ctx_payload = {
                        "branch_id": br.id,
                        "from_beat_id": br.from_beat_id,
                        "title": br.title,
                        "description": br.description,
                        "ending_tone": br.ending_tone,
                    }

                    br_contracts_for_branch = await self._patch_scene_contracts_with_consistency_critic(
                        scene_contracts=br_contracts_for_branch,
                        setting=setting_obj,
                        char_list=char_list_final,
                        loc_list=loc_list_final,
                        loc_canons=loc_canons,
                        loc_affordances=loc_affordances,
                        loc_graph=loc_graph,
                        plot_threads=threads,
                        artifact_store=store,
                        artifact_name=br.id,
                        branch_context=branch_ctx_payload,
                    )

                    br_contracts_for_branch = self._sanitize_contracts_schema(
                        br_contracts_for_branch,
                        char_list=char_list_final,
                        loc_list=loc_list_final,
                    )

                    for c in br_contracts_for_branch:
                        self._canon_contract_inplace(c, name_canon, store=store)

                    br_contracts_for_branch = await self._re_enrich_scene_contracts(
                        setting=setting_obj,
                        outline=br_outline,
                        scene_contracts=br_contracts_for_branch,
                        plot_threads=threads,
                        artifact_store=store,
                        artifact_name=br.id,
                        branch_context=branch_ctx_payload,
                    )

                    planned_branch_outlines[br.id] = br_outline
                    branch_contracts_by_branch[br.id] = br_contracts_for_branch
                    branch_contracts.extend(br_contracts_for_branch)

                choice_contracts = await self._plan_choice_contracts(
                    setting=setting_obj,
                    outline_main=outline_main_obj,
                    main_contracts=main_contracts,
                    branching=branching_info,
                    branch_contracts_by_branch=branch_contracts_by_branch,
                    artifact_store=store,
                )

                choice_context_by_scene = self._build_choice_context_by_scene(
                    main_contracts,
                    choice_contracts,
                )

            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="scene_contracts_done",
                message="Main scene contracts prepared",
                percent=68,
            )

            state_snapshots_main: Dict[str, StoryState] = {}
            main_scripts = await self._write_scenes(
                setting=setting_obj,
                outline=outline_main_obj,
                scene_contracts=main_contracts,
                char_list=char_list_final,
                char_appearance=char_appearance,
                rag_bundle=rag_main,
                story_state=story_state_main,
                story_length=normalized_length,
                branching=branching_info,
                initial_previous_summaries=None,
                initial_previous_last_lines=None,
                initial_prev_location=None,
                state_snapshots=state_snapshots_main,
                loc_graph=loc_graph,
                char_graph=char_graph,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_list=loc_list_final,
                artifact_store=store,
                artifact_prefix="main",
                name_canon=name_canon,
                strict_names=self._strict_name_canon_enabled(),
                strict_locations=self._strict_location_gate_enabled(),
                progress_callback=progress_callback,
                generation_id=generation_id,
                progress_stage="writing_main",
                progress_start=70,
                progress_end=88,
                choice_context_by_scene=choice_context_by_scene,
            )

            branch_scripts: Dict[str, SceneScript] = {}
            branch_states: Dict[str, StoryState] = {}

            if len(branching_info.branches) > 1:
                await self._emit_progress(
                    progress_callback,
                    generation_id=generation_id,
                    stage="branch_setup",
                    message="Preparing branch scenes",
                    percent=89,
                )

                for br in branching_info.branches:
                    if br.id == "main" or not br.from_beat_id:
                        continue

                    br_outline = planned_branch_outlines.get(br.id)
                    br_contracts_for_branch = branch_contracts_by_branch.get(br.id) or []
                    if br_outline is None or not br_contracts_for_branch:
                        continue

                    divergence_scene_id = br.from_scene_id or beat_to_main_scene.get(br.from_beat_id)
                    initial_prev: List[str] = []
                    initial_last_lines: List[str] = []
                    initial_prev_location: Optional[str] = None

                    if divergence_scene_id:
                        for sc in main_contracts:
                            initial_prev.append(f"{sc.id}: {main_scripts[sc.id].summary}")
                            if sc.id == divergence_scene_id:
                                initial_last_lines = self._extract_last_lines(main_scripts[divergence_scene_id], n=3)
                                initial_prev_location = sc.location
                                break

                    base_threads = threads_base_index.clone() if threads_base_index is not None else rag_main.threads.clone()

                    rag_branch = RAGBundle(
                        self.client,
                        embed_model,
                        world_index=rag_main.world,
                        char_index=rag_main.characters,
                        thread_index=base_threads,
                    )
                    for b in br_outline.beats:
                        await rag_branch.story.upsert_item(
                            f"{br.id}::beat::{b.id}",
                            "beat",
                            (b.summary or "")[:2500],
                        )

                    if divergence_scene_id and divergence_scene_id in state_snapshots_main:
                        branch_state = StoryState(**state_snapshots_main[divergence_scene_id].dict())
                    else:
                        store.event(
                            "branch.state_snapshot_missing",
                            {
                                "branch_id": br.id,
                                "from_beat_id": br.from_beat_id,
                                "divergence_scene_id": divergence_scene_id,
                            },
                        )
                        branch_state = StoryState()
                        branch_state.world["char_type"] = char_type
                    selected_option_ctx = None
                    for cc in choice_contracts:
                        for op in (cc.options or []):
                            if op.branch_id == br.id:
                                selected_option_ctx = {
                                    "choice_id": cc.id,
                                    "decision_scene_id": cc.decision_scene_id,
                                    "decision_question": cc.decision_question,
                                    "why_now": cc.why_now,
                                    "option": op.dict() if hasattr(op, "dict") else self._obj_to_dict(op),
                                }
                                break
                        if selected_option_ctx is not None:
                            break

                    if selected_option_ctx is not None:
                        branch_state.world["selected_branch_option"] = selected_option_ctx

                    try:
                        changed_tids: List[str] = []
                        for tid, raw in (branch_state.plot_threads or {}).items():
                            d = self._obj_to_dict(raw)
                            if str(d.get("status") or "open") != "open" or int(d.get("last_touched_order") or 0) > 0:
                                changed_tids.append(str(tid))
                        await self._update_thread_index_from_state(
                            rag_branch,
                            branch_state,
                            changed_thread_ids=changed_tids or None,
                        )
                    except Exception as e:
                        store.event("branch.thread_index_align_failed", {"branch_id": br.id, "error": str(e)})

                    br_scripts = await self._write_scenes(
                        setting=setting_obj,
                        outline=br_outline,
                        scene_contracts=br_contracts_for_branch,
                        char_list=char_list_final,
                        char_appearance=char_appearance,
                        rag_bundle=rag_branch,
                        story_state=branch_state,
                        story_length=normalized_length,
                        branching=branching_info,
                        initial_previous_summaries=initial_prev,
                        initial_previous_last_lines=initial_last_lines,
                        initial_prev_location=initial_prev_location,
                        state_snapshots=None,
                        loc_graph=loc_graph,
                        char_graph=char_graph,
                        loc_canons=loc_canons,
                        loc_affordances=loc_affordances,
                        loc_list=loc_list_final,
                        artifact_store=store,
                        artifact_prefix=br.id,
                        name_canon=name_canon,
                        strict_names=self._strict_name_canon_enabled(),
                        strict_locations=self._strict_location_gate_enabled(),
                    )

                    branch_scripts.update(br_scripts)
                    branch_states[br.id] = branch_state

                self._inject_branch_choices(main_scripts, choice_contracts)

            all_contracts = main_contracts + branch_contracts
            all_scripts: Dict[str, SceneScript] = {}
            all_scripts.update(main_scripts)
            all_scripts.update(branch_scripts)

            story_state_by_branch: Dict[str, Any] = {"main": story_state_main.dict()}
            for bid, st in branch_states.items():
                story_state_by_branch[bid] = st.dict()

            result: Dict[str, Any] = {
                "generation_id": generation_id,
                "status": "completed",
                "user_prompt": user_request.user_prompt,
                "story_length": normalized_length,
                "user_request": user_request.dict(),
                "setting": setting_obj.dict(),
                "outline": outline_obj.dict(),
                "char_list": char_list_final,
                "char_graph": char_graph,
                "char_appearance": char_appearance,
                "char_type": char_type,
                "loc_list": loc_list_final,
                "loc_graph": loc_graph,
                "loc_description": loc_description,
                "loc_canons": loc_canons,
                "loc_affordances": loc_affordances,
                "char_images": [img.dict() for img in char_images],
                "loc_images": [img.dict() for img in loc_images],
                "branching": branching_info.dict(),
                "choice_contracts": [cc.dict() for cc in choice_contracts],
                "scene_contracts": [sc.dict() for sc in all_contracts],
                "scenes": {sid: script.dict() for sid, script in all_scripts.items()},
                "story_state_main": story_state_main.dict(),
                "story_state_by_branch": story_state_by_branch,
            }
            #постпроцесс - легаси
            if str(os.getenv("POSTPROCESS_CHOICES", "true")).strip().lower() in ("1", "true", "yes", "on"):
                use_llm_pp = str(os.getenv("CHOICE_PATCH_USE_LLM", "true")).strip().lower() in ("1", "true", "yes", "on")
                choice_model = os.getenv("CHOICE_PATCH_MODEL") or self.router.get_model_for_agent("choice_patch_agent")

                try:
                    await self._emit_progress(
                        progress_callback,
                        generation_id=generation_id,
                        stage="postprocess_choices",
                        message="Postprocessing choices (no fake buttons, add POV/context)",
                        percent=96,
                    )

                    result = await patch_choices_payload(
                        result,
                        use_llm=use_llm_pp,
                        fake_every=0,  #удалено
                        client=self.client if use_llm_pp else None,
                        model=choice_model,
                        rng_seed=seed,
                    )

                    scenes_map = result.get("scenes") or {}
                    if isinstance(scenes_map, dict):
                        for sid, sc in scenes_map.items():
                            if not isinstance(sc, dict):
                                continue
                            bid = str(sc.get("branch_id") or "main")
                            prefix = "main" if bid == "main" else bid
                            store.save(f"scenes/{prefix}/{sid}.json", sc)

                    store.event("postprocess.choices_patched", result.get("postprocess") or {})

                    await self._emit_progress(
                        progress_callback,
                        generation_id=generation_id,
                        stage="postprocess_choices_done",
                        message="Choices postprocessed",
                        percent=97,
                    )
                except Exception as e:
                    app_logger.error(f"Choice postprocess failed: {e}", exc_info=True)
                    store.event("postprocess.choices_failed", {"error": str(e)})

            store.save("final.json", result)

            if generate_images:
                try:
                    await self._emit_progress(
                        progress_callback,
                        generation_id=generation_id,
                        stage="web_export",
                        message="Generating FLUX assets and HTML export",
                        percent=98,
                    )

                    web_export = await export_web_from_json(
                        store.root / "final.json",
                        out_dir=store.root / "web",
                        llm_client=self.client,
                        seed=seed,
                    )

                    char_images, loc_images = self._web_export_to_image_lists(web_export)

                    result["char_images"] = [img.dict() for img in char_images]
                    result["loc_images"] = [img.dict() for img in loc_images]
                    result["web_export"] = web_export

                except Exception as e:
                    app_logger.error(f"Web export failed: {e}", exc_info=True)
                    result["web_export"] = {"error": str(e)}
            else:
                app_logger.info("Image/web export disabled.")
                result["web_export"] = None

            store.save("final.json", result)
            store.checkpoint(
                "99_final_summary",
                {
                    "generation_id": generation_id,
                    "status": "completed",
                    "scenes_total": len(all_scripts),
                    "tokens_used": self.client.total_tokens_used,
                    "api_calls": self.client.call_count,
                },
            )
            await self._emit_progress(
                progress_callback,
                generation_id=generation_id,
                stage="finalizing",
                message="Finalizing result",
                percent=99,
            )

            return result

        finally:
            TRACE_HOOK.reset(token)