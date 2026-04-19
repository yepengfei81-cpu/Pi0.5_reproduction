"""直接用 pyarrow 修复 parquet 文件中 'List' -> 'Sequence' 的兼容性问题"""
import json
from pathlib import Path
import pyarrow.parquet as pq

DATA_DIR = Path("/root/autodl-tmp/data/airbot_play_data")

parquet_files = sorted(DATA_DIR.rglob("*.parquet"))
print(f"找到 {len(parquet_files)} 个 parquet 文件")

fixed_count = 0
for pf in parquet_files:
    table = pq.read_table(pf)
    metadata = table.schema.metadata
    if metadata is None:
        print(f"  跳过 (无 metadata): {pf}")
        continue

    new_metadata = {}
    changed = False
    for k, v in metadata.items():
        k_str = k.decode("utf-8") if isinstance(k, bytes) else k
        v_str = v.decode("utf-8") if isinstance(v, bytes) else v

        if '"_type": "List"' in v_str:
            v_str = v_str.replace('"_type": "List"', '"_type": "Sequence"')
            changed = True

        new_metadata[k if isinstance(k, bytes) else k.encode("utf-8")] = (
            v_str.encode("utf-8") if isinstance(v, bytes) else v_str
        )

    if changed:
        new_schema = table.schema.with_metadata(new_metadata)
        new_table = table.replace_schema_metadata(new_metadata)
        pq.write_table(new_table, pf)
        fixed_count += 1
        print(f"  已修复: {pf.relative_to(DATA_DIR)}")
    else:
        print(f"  无需修改: {pf.relative_to(DATA_DIR)}")

print(f"\n完成！共修复 {fixed_count} 个文件")