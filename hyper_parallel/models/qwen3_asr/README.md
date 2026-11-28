# Qwen3-ASR（1.7B）接入

Qwen3-ASR 是语音识别模型：24 层卷积下采样 + 语音编码器（`Qwen3ASREncoder`）输出经
`multi_modal_projector` 投影后，替换文本序列中的音频占位符，再由 28 层 Qwen3 文本解码器
生成转写。本目录把它接入 HyperParallel 的 Omni 数据生命周期
（`scripts/train_vl.py` + `VLMTrainer`）。

## 文件

```text
models/qwen3_asr/
├── modeling_qwen3_asr.py          # 训练用模型类（补 from_config 入口）
├── adapter/
│   ├── registration.py            # 架构注册 + ModelAdapterSpec（目录约定自动发现）
│   ├── materialization.py         # 语音塔正弦位置表 + 文本塔 RoPE 的重建
│   ├── policies/sharding.py       # 语音塔/投影器的 TP 角色
│   └── data/transform_fn.py       # 样本 → 模型输入与标签
└── recipes/
    ├── train.yaml                 # 训练配方
    ├── prepare_asr_data.py        # 合成演示数据生成
    └── run_qwen3_asr_sft.sh       # 启动脚本
```

`models/registry.py` 按目录约定（`models/<family>/adapter/registration.py`）自动发现
本家族，无需改动任何共享注册代码。

## 数据格式

两种 JSONL 写法都接受。**官方格式**与 `QwenLM/Qwen3-ASR` 的 `finetuning/` 目录一致：

```jsonl
{"audio": "wavs/utt0000.wav", "text": "language English<asr_text>the quick brown fox"}
{"audio": "wavs/utt0001.wav", "text": "language Chinese<asr_text>这是一个测试"}
```

**Omni chat 格式**（与其他多模态家族一致，演示脚本产出的就是这种）：

```jsonl
{"messages": [
  {"role": "user", "content": [{"type": "audio", "audio": "wavs/utt0000.wav"}]},
  {"role": "assistant", "content": "language English<asr_text>the quick brown fox"}]}
```

- `audio`：WAV 路径。绝对路径或 http(s) 直接使用；相对路径按 JSONL 文件所在目录解析
  （由 `OmniDataTransform.prepare_messages` 统一处理）。
- 转写文本带语言信息时用 `language <NAME><asr_text>...`，语言未知时用
  `language None<asr_text>...`（官方约定）。
- 采样率不是 16 kHz 时会自动重采样到 16 kHz。
- 每条记录只支持一个音频片段；多片段样本会显式报错。

标签只监督 `<asr_text>` 之后的转写与结尾 `<|im_end|>`，chat 模板头、音频占位符和
assistant 头全部置为 `-100`。

## 权重

需要 **HF 格式**的 checkpoint（`config.json` 里是 `Qwen3ASRConfig`、`model.safetensors`
按 `model.audio_tower.*` / `model.language_model.*` / `model.multi_modal_projector.*` 命名）：

```bash
modelscope download --model Qwen/Qwen3-ASR-1.7B-hf --local_dir /path/to/Qwen3-ASR-1.7B-hf
```

原始 `Qwen/Qwen3-ASR-1.7B`（`thinker.*` 前缀）不能直接加载，需先转成 HF 格式
（`transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py`）。

## 运行

```bash
# 1) 生成演示数据（合成语音，仅用于打通链路）
python hyper_parallel/models/qwen3_asr/recipes/prepare_asr_data.py \
  --output-dir ./qwen3_asr_demo_data --num-samples 8

# 2) 单机 8 卡 SFT
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash hyper_parallel/models/qwen3_asr/recipes/run_qwen3_asr_sft.sh \
  /path/to/Qwen3-ASR-1.7B-hf ./qwen3_asr_demo_data/train.jsonl
```

## 并行说明

- 语音编码器（`model.audio_tower`）与音频投影器（`model.multi_modal_projector`）**整棵子树
  复制**：编码器注意力在完整 hidden 维上 reshape heads，卷积栈也无法按 head 切分。该布局在
  `recipes/train.yaml` 的 `plan_overrides` 里逐模块声明（这些子树不匹配内置边界模板）。
- 文本塔（`model.language_model`）走通用 colwise/rowwise 规则，可 TP。
- CP 未接入：音频特征在文本位置上是离散替换，需要专门的 CP 包装器。
- `micro_batch_size > 1` 时必须同时给 `data_transform.audio_pad_frames` 和
  `data_transform.pad_to_length` 一个固定值，否则不同长度的语谱图 / token 序列无法堆叠
  （Omni collator 只做批次尾部对齐补齐，不会逐样本补齐）。

## 精度

HP 与 HF 原生实现在同一份输入上的 loss 完全一致（fp32，真实 1.7B 权重：1.9192757606506348
vs 1.9192757606506348）。改动前 HP 偏 0.77 的 logits、loss 从 11.97 漂到 16.21，原因是
`init_device: meta` 构造后非持久 buffer 只剩 meta 值，见 `adapter/materialization.py`：
语音塔的正弦位置表按模块几何重建，文本塔的 RoPE 频率（`inv_freq` / `original_inv_freq`）
从 CPU 参考实例重建后再拷回目标设备（NPU pow 的 FP32 ULP 差异会翻转 BF16 的 cos/sin）。
