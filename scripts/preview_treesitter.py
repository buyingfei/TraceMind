import json
import os
from pathlib import Path

os.environ.pop("MILVUS_URI", None)

from app.services.treesitter_java_splitter import split_preview_from_config

result = split_preview_from_config()
print("file=", result["file"])
print("chunk_count=", result["chunk_count"])
for c in result["chunks"]:
    print(
        f"#{c['index']} type={c['node_type']} name={c['name']} "
        f"lines={c['start_line']}-{c['end_line']} chars={len(c['text'])}"
    )

out = Path("output/treesitter_preview.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
