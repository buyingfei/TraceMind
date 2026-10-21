"""用 Tree-sitter 切割 Java 源码。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tree_sitter_java as tsjava
import yaml
from tree_sitter import Language, Node, Parser

_JAVA_LANGUAGE = Language(tsjava.language())
_PARSER = Parser(_JAVA_LANGUAGE)

DEFAULT_NODE_TYPES = (
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "method_declaration",
    "constructor_declaration",
    "static_initializer",
    "compact_constructor_declaration",
)


@dataclass
class CodeChunk:
    index: int
    node_type: str
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_code_index_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载索引切块 YAML；默认路径来自 Settings.code_index_config_path。"""
    if config_path is None:
        from app.core.config import get_settings

        path = get_settings().resolve_code_index_config_path()
    else:
        path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_preview_file(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_code_index_config()
    root = Path(cfg["repo"]["root"])
    rel = cfg["preview"]["file"]
    path = root / rel
    if not path.exists():
        raise FileNotFoundError(f"预览文件不存在: {path}")
    return path


def _node_name(node: Node, source: bytes) -> str:
    # 多数声明节点下有 name 字段
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="replace")
    if node.type == "static_initializer":
        return "<clinit>"
    return ""


def _slice_text(source: bytes, start_byte: int, end_byte: int) -> str:
    return source[start_byte:end_byte].decode("utf-8", errors="replace")


def _collect_nodes(root: Node, node_types: set[str]) -> list[Node]:
    hits: list[Node] = []

    def walk(node: Node) -> None:
        if node.type in node_types:
            hits.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    # 按出现顺序
    hits.sort(key=lambda n: (n.start_byte, n.end_byte))
    return hits


def _split_oversized(chunk: CodeChunk, max_chars: int) -> list[CodeChunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]

    lines = chunk.text.splitlines(keepends=True)
    parts: list[CodeChunk] = []
    buf: list[str] = []
    line_cursor = chunk.start_line
    part_start = chunk.start_line

    def flush(end_line: int) -> None:
        nonlocal buf, part_start
        if not buf:
            return
        text = "".join(buf)
        parts.append(
            CodeChunk(
                index=0,
                node_type=f"{chunk.node_type}#part",
                name=chunk.name,
                start_line=part_start,
                end_line=end_line,
                start_byte=chunk.start_byte,
                end_byte=chunk.end_byte,
                text=text,
            )
        )
        buf = []

    for i, line in enumerate(lines):
        current_line = chunk.start_line + i
        if buf and sum(len(x) for x in buf) + len(line) > max_chars:
            flush(current_line - 1)
            part_start = current_line
        buf.append(line)
        line_cursor = current_line
    flush(line_cursor)
    return parts


def split_java_source(
    source_text: str,
    *,
    node_types: list[str] | None = None,
    include_preamble: bool = True,
    max_chars: int = 4000,
) -> list[CodeChunk]:
    source = source_text.encode("utf-8")
    tree = _PARSER.parse(source)
    types = set(node_types or DEFAULT_NODE_TYPES)
    nodes = _collect_nodes(tree.root_node, types)

    raw_chunks: list[CodeChunk] = []

    if include_preamble:
        first_decl = min((n.start_byte for n in nodes), default=None)
        if first_decl is None:
            preamble_end = len(source)
        else:
            preamble_end = first_decl
        preamble = _slice_text(source, 0, preamble_end).strip()
        if preamble:
            # 行号：前言到第一个声明前一行
            end_line = nodes[0].start_point[0] if nodes else source_text.count("\n")
            # start_point 是 0-based；前言结束行取声明起始行（若无声明则为末行）
            raw_chunks.append(
                CodeChunk(
                    index=0,
                    node_type="preamble",
                    name="<package+imports>",
                    start_line=1,
                    end_line=max(1, end_line),
                    start_byte=0,
                    end_byte=preamble_end,
                    text=preamble,
                )
            )

    for node in nodes:
        text = _slice_text(source, node.start_byte, node.end_byte)
        raw_chunks.append(
            CodeChunk(
                index=0,
                node_type=node.type,
                name=_node_name(node, source),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                text=text,
            )
        )

    # 超长切分 + 重新编号
    final: list[CodeChunk] = []
    for chunk in raw_chunks:
        final.extend(_split_oversized(chunk, max_chars))
    for i, chunk in enumerate(final):
        chunk.index = i
    return final


def split_java_file(path: str | Path, **kwargs: Any) -> list[CodeChunk]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return split_java_source(text, **kwargs)


def split_preview_from_config(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_code_index_config(config_path)
    file_path = resolve_preview_file(cfg)
    splitter_cfg = cfg.get("splitter") or {}
    chunks = split_java_file(
        file_path,
        node_types=splitter_cfg.get("node_types"),
        include_preamble=bool(splitter_cfg.get("include_preamble", True)),
        max_chars=int(splitter_cfg.get("max_chars", 4000)),
    )
    return {
        "repo": cfg.get("repo", {}),
        "file": str(file_path),
        "relative_file": cfg.get("preview", {}).get("file"),
        "splitter": {
            "engine": splitter_cfg.get("engine", "tree-sitter"),
            "language": splitter_cfg.get("language", "java"),
            "node_types": splitter_cfg.get("node_types", list(DEFAULT_NODE_TYPES)),
            "include_preamble": splitter_cfg.get("include_preamble", True),
            "max_chars": splitter_cfg.get("max_chars", 4000),
        },
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
