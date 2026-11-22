# 数据（不入库）

仓库约定：数据集留在本地。这里的文件由 `src/muonint/data.py` 生成或从公开数据获得。

- `climbmix_400m.u16` — 预训练 token 缓存（uint16，约 800 MB）。由
  `muonint.data.build_token_cache(path, 400_000_000)` 从集群共享的
  `/mnt/oss/datasets/climbmix/part_0.tokenized.jsonl` 流式生成（GPT-2 tokenizer，已预先分词）。
- `gsm8k_train.jsonl` / `gsm8k_test.jsonl` — GSM8K 数据集，供 `muonint.rl` 读取。

重新生成缓存约需 30 秒（读一次共享文件）。
