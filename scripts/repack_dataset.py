"""直接用 pyarrow 修复 parquet 文件中 'List' -> 'Sequence' 的兼容性问题"""
import argparse
import json
import os
from pathlib import Path
import pyarrow.parquet as pq

# 数据集根目录取自 HF_LEROBOT_HOME（与 collect_data.py / 训练保持一致），
# 未设置时回退到 lerobot 默认缓存目录。可用 --repo-id / --data-dir 覆盖。
parser = argparse.ArgumentParser(description="修复 LeRobot parquet 元数据兼容性")
parser.add_argument("--repo-id", type=str, default="airbot_play_data",
                    help="数据集名称（HF_LEROBOT_HOME 下的子目录）")
parser.add_argument("--data-dir", type=str, default=None,
                    help="直接指定数据集目录，优先级高于 HF_LEROBOT_HOME/repo-id")
args = parser.parse_args()

if args.data_dir:
    DATA_DIR = Path(args.data_dir)
else:
    lerobot_home = os.environ.get(
        "HF_LEROBOT_HOME", str(Path.home() / ".cache" / "huggingface" / "lerobot")
    )
    DATA_DIR = Path(lerobot_home) / args.repo_id

print(f"数据集目录: {DATA_DIR}")

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