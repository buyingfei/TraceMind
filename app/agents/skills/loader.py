"""Agent Skills 规范加载器（目录/SKILL.md + YAML frontmatter）。

格式对齐 https://agentskills.io/specification 与 LangChain Deep Agents skills。
Deep Agent 模式：由 SkillsMiddleware 发现/加载；tool_loop 回退时仍可按意图注入正文。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

BizIntentLike = Literal["scan_tyre", "warranty_plate", "generic_biz", "none"]

# 意图 → skill 目录名（须与 SKILL.md name / 文件夹名一致）
_INTENT_SKILLS: dict[str, tuple[str, ...]] = {
    "scan_tyre": ("biz-sql-query", "biz-sharding-sql"),
    "warranty_plate": ("biz-sql-query",),
    "generic_biz": ("biz-sql-query", "biz-sharding-sql"),
}


@dataclass(frozen=True)
class SkillInfo:
    """单份 skill 的元数据 + 正文（不含 frontmatter）。"""

    name: str
    description: str
    dir_path: Path
    body: str
    metadata: dict[str, str]
    compatibility: str = ""


def _parse_simple_yaml(block: str) -> dict[str, Any]:
    """解析 skill frontmatter 常用子集（标量 / 折叠标量 / 一层 metadata map）。"""
    result: dict[str, Any] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()

        if key == "metadata" and not rest:
            meta: dict[str, str] = {}
            i += 1
            while i < len(lines):
                sub = lines[i]
                if not sub.startswith(" ") and not sub.startswith("\t"):
                    break
                if ":" not in sub:
                    i += 1
                    continue
                mk, _, mv = sub.partition(":")
                meta[mk.strip()] = mv.strip().strip("\"'")
                i += 1
            result["metadata"] = meta
            continue

        if rest in (">", ">-", "|", "|-"):
            folded: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith(" ") or nxt.startswith("\t"):
                    folded.append(nxt.strip())
                    i += 1
                    continue
                break
            result[key] = " ".join(folded).strip()
            continue

        result[key] = rest.strip("\"'")
        i += 1
    return result


def _split_skill_md(text: str) -> tuple[dict[str, Any], str]:
    stripped = text.strip()
    m = _FRONTMATTER_RE.match(stripped + "\n")
    if not m:
        return {}, stripped
    return _parse_simple_yaml(m.group(1)), (m.group(2) or "").strip()


@lru_cache(maxsize=32)
def load_skill(name: str) -> SkillInfo | None:
    """按目录名加载 skills/{name}/SKILL.md；校验 name 与目录一致。"""
    skill_dir = _SKILLS_DIR / name
    path = skill_dir / "SKILL.md"
    if not path.is_file():
        logger.warning("skill 不存在: %s", path)
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("读取 skill 失败 %s: %s", path, exc)
        return None

    meta, body = _split_skill_md(text)
    skill_name = str(meta.get("name") or "").strip()
    description = str(meta.get("description") or "").strip()
    if not skill_name or not _NAME_RE.match(skill_name):
        logger.warning("skill name 非法或不存在: %s (%s)", skill_name, path)
        return None
    if skill_name != name:
        logger.warning(
            "skill name 与目录不一致: frontmatter=%s dir=%s",
            skill_name,
            name,
        )
        return None
    if not description:
        logger.warning("skill 缺少 description: %s", path)
        return None
    if len(description) > 1024:
        logger.warning("skill description 超过 1024 字符: %s", path)

    raw_meta = meta.get("metadata") or {}
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    metadata = {str(k): str(v) for k, v in raw_meta.items()}

    return SkillInfo(
        name=skill_name,
        description=description,
        dir_path=skill_dir,
        body=body,
        metadata=metadata,
        compatibility=str(meta.get("compatibility") or "").strip(),
    )


def list_skills() -> list[SkillInfo]:
    """扫描 skills/*/SKILL.md。"""
    out: list[SkillInfo] = []
    if not _SKILLS_DIR.is_dir():
        return out
    for child in sorted(_SKILLS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        if child.name in {"__pycache__"}:
            continue
        if not (child / "SKILL.md").is_file():
            continue
        info = load_skill(child.name)
        if info:
            out.append(info)
    return out


def skill_summaries() -> list[dict[str, str]]:
    """仅 name + description，便于将来写入 system 摘要 / Deep Agents。"""
    return [{"name": s.name, "description": s.description} for s in list_skills()]


def skill_suffix_for_intent(intent: BizIntentLike | None) -> str:
    """按业务意图注入对应 skill 正文；none 不注入。

    Deep Agent 模式下不注入全文（由 SkillsMiddleware 按 description 渐进加载）。
    """
    if intent == "none":
        return ""
    try:
        from app.core.config import get_settings

        if get_settings().use_deep_agent:
            return ""
    except Exception:  # noqa: BLE001
        pass
    names = _INTENT_SKILLS.get(
        intent or "scan_tyre",
        ("biz-sql-query", "biz-sharding-sql"),
    )
    parts: list[str] = []
    for name in names:
        info = load_skill(name)
        if info and info.body:
            parts.append(info.body)
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)


def load_knowledge(name: str) -> str:
    """旧 API 兼容：'biz_sharding_sql' → 'biz-sharding-sql'。"""
    key = "biz-sharding-sql" if name == "biz_sharding_sql" else name
    info = load_skill(key)
    return info.body if info else ""


def knowledge_suffix_for_intent(intent: BizIntentLike | None) -> str:
    """旧 API 兼容。"""
    return skill_suffix_for_intent(intent)
