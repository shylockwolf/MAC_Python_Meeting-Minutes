# MAC Python Meeting Minutes

macOS 平台的会议纪要自动生成工具，支持 **mlx-whisper / Qwen3-ASR / Audio8-ASR** 三引擎语音转录 + DeepSeek 大模型智能总结，一键完成从音频到 PDF 会议纪要的全流程。

## 功能特性

- **三引擎语音转录**：支持 mlx-whisper、Qwen3-ASR-0.6B、Audio8-ASR-0.1B 三种本地 ASR 模型，转录为带精确时间戳的文本（输出 `.txt`）
- **智能成文**：调用 DeepSeek 大模型对转录文本进行段落合并、错别字修正、格式优化、ASR 噪声清洗，按时间片段或语义自动分块处理
- **会议纪要**：基于成文内容生成结构化会议纪要，支持超长文本（10万+字符）两阶段处理（分段摘要 → 全局整合），输出 `.pdf`
- **播客脚本**：将会议内容转化为双人对话形式的播客脚本（输出 `.txt`）
- **一键模式**：选择音频 → 一键生成，自动完成 转录 → 成文 → 纪要 PDF 全流程
- **多文件支持**：可同时选择多个文本文件进行批量成文处理
- **ASR 噪声清洗**：自动消除 ASR 产生的连续重复字（如"对对对""嗯嗯嗯""好好好"）和逗号分隔的重复模式（如"，对，对，对"）
- **智能分块**：
  - 按 ASR 时间片段（`[MM:SS-MM:SS]`）精确分割，逐段调用 AI 处理
  - 密集时间片段（如 whisper 逐句输出）自动按时长合并为 ~12 分钟大块，保持上下文连贯
  - 单个文本块超过 15,000 字符时自动递归拆分为子段，防止 API 输出截断
  - 自动识别并跳过填充词密集段（嗯/啊/哦等占比 > 60%），节省 API 调用
- **PDF 生成**：内置 reportlab / cupsfilter / textutil 三条 PDF 生成路径，自动降级确保输出成功

## 环境要求

- macOS（Apple Silicon 推荐，利用 MLX 加速）
- Python 3.10+
- ffmpeg（用于音频分割与格式转换）

## 安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 ffmpeg（如未安装）
brew install ffmpeg
```

### 模型准备

**mlx-whisper 模型**（默认 Apple Silicon）：
```bash
# 模型默认路径为 ../../myMLX/whisper-small-mlx
```

**Qwen3-ASR 模型**（Apple Silicon 推荐）：
```bash
# 需安装 mlx-audio 包（已包含在 requirements.txt 中）
# 模型默认路径为 ../../myMLX/Qwen3-ASR-0.6B-8bit
```

**Audio8-ASR 模型**（跨平台，Intel Mac 备选）：
```bash
# 需额外安装：pip install numpy onnxruntime soundfile librosa psutil tokenizers transformers
# 模型默认路径为 ../../myMLX/Audio8-ASR-0.1B-onnx-runtime
```

## 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 使用

```bash
python meeting_minutes.py
```

### 操作流程

1. **选择音频文件**：点击"打开"按钮选择音频文件（支持 mp3/wav/m4a/flac/ogg）
2. **选择 ASR 模型**：在下拉框中选择 whisper-small、Qwen3-ASR-0.6B-8bit 或 Audio8-ASR-0.1B-onnx
3. **一键生成**：点击"一键生成"按钮，自动完成转录 → 成文 → 纪要 PDF
4. **分步操作**：
   - "开始转录"：仅执行语音转文字
   - "开始成文"：对已转录或已载入的文本进行格式化
   - "生成会议纪要"：基于文本生成结构化纪要
   - "生成播客"：生成双人对话播客脚本

### 文本文件独立处理

无需音频文件，可直接打开 `.txt` 文本进行成文、纪要、播客等处理：
- 支持多文件同时选择，批量成文处理
- 适用于已有转录文本的二次处理场景

### 输出文件

| 功能 | 输出文件 | 说明 |
|------|---------|------|
| 语音转录 | `文件名.txt` | 带 `[MM:SS-MM:SS]` 时间戳的原始转录文本 |
| 成文处理 | `文件名_成文.txt` | 段落合并、格式优化后，保留时间戳 |
| 会议纪要 | `文件名_纪要.pdf` | 结构化会议纪要（PDF格式，含中文排版） |
| 播客脚本 | `文件名_播客.txt` | 双人对话形式播客脚本 |

## 模型与参数配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| ASR 模型（Apple Silicon 首选） | `whisper-small (mlx)` | 约 10x 实时速度，MLX 加速 |
| ASR 模型（备选） | `Qwen3-ASR-0.6B-8bit` | 约 7x 实时速度，含模型加载开销 |
| ASR 模型（跨平台） | `Audio8-ASR-0.1B-onnx` | 约 14x 实时速度，ONNX INT8，支持 Intel Mac |
| 成文 LLM | `deepseek-v4-flash` | 从 `.env` 中 `DEEPSEEK_MODEL` 读取 |
| 纪要 LLM | `deepseek-v4-flash` | 同上 |
| 播客 LLM | `deepseek-v4-pro` | 固定使用 Pro 模型以保证生成质量 |
| 最大输出 token | 384,000 | 单次 API 调用最大输出 |
| API 超时 | 300s（连接 30s） | 通过 httpx.Timeout 在客户端层统一控制 |
| 文本块安全上限 | 15,000 字符 | 超限自动递归拆分，防止输出截断 |
| 成文分块大小 | 30,000 字符 | 无时间戳时按此大小分段 |
| 纪要分块大小 | 100,000 字符 | 超过则启用两阶段处理 |
| 密集片段阈值 | 50 段 | 超过则按时长合并为 ~12 分钟大块处理 |
| 填充词过滤阈值 | 60% | 非空白字符中填充词（嗯啊哦等）占比超此值则跳过 |

### 三种 ASR 模型对比

| 模型 | 引擎 | 速度 | 精度 | 平台 | 适用场景 |
|------|------|------|------|------|---------|
| whisper-small (mlx) | MLX | 10x | ⭐⭐⭐ | Apple Silicon | 默认选择，速度快 |
| Qwen3-ASR-0.6B-8bit | MLX | 7x | ⭐⭐⭐⭐ | Apple Silicon | 中文识别优秀 |
| Audio8-ASR-0.1B-onnx | ONNX | 14x | ⭐⭐⭐ | 跨平台 | Intel Mac / Linux / Windows 备选 |

**Audio8 模型限制：**
- ONNX max_total_len=512，prompt 占 385 tokens，最多只能输出 127 tokens
- 因此自动按 **30 秒/段** 切分音频（其他模型按 15 分钟/段）
- 17 分钟音频会被切成约 36 段

## 处理策略

### 成文阶段

1. **时间片段优先**：如文本含 `[MM:SS-MM:SS]` 标记，优先按时间片段逐段调用 AI
2. **密集片段合并**：片段数 > 50 时（如 whisper 逐句输出），按累计时长合并为 ~12 分钟大块，保持上下文连贯
3. **填充词过滤**：每段处理前检测填充词占比（嗯/啊/哦/额等），超过 60% 则跳过 API 调用，直接标记为无效片段
4. **安全拆分**：单段超过 15,000 字符时，递归拆分为子段分别处理
5. **时间范围透传**：每段携带原始时间范围，AI 根据文本相对位置估算子段落时间戳
6. **ASR 噪声清洗**：完成后自动消除连续重复字（如"对对对""嗯嗯嗯"）、逗号分隔重复模式（如"，对，对"）、连续重复标点

### 纪要阶段

1. **短文本**（≤ 100,000 字符）：直接调用 AI 生成完整纪要
2. **长文本**（> 100,000 字符）：
   - 第一阶段：按 100,000 字符分段，每段生成结构化摘要（议题/结论/行动项/未解决问题）
   - 第二阶段：全局整合所有摘要，去重合并，生成最终纪要

### PDF 生成

按优先级尝试三种方式：
1. `reportlab`（Python 库，支持中文字体、标题层级、列表样式）
2. `cupsfilter`（macOS 系统自带）
3. `textutil`（macOS 系统自带，HTML → RTF → PDF）

## 提示词模板

- `会议纪要提示词.txt`：会议纪要生成的提示词模板
- `播客提示词.txt`：播客脚本生成的提示词模板

可根据需要自定义修改。

## 项目结构

```
.
├── meeting_minutes.py      # 主程序（GUI）
├── requirements.txt        # Python 依赖
├── 会议纪要提示词.txt        # 会议纪要提示词模板
├── 播客提示词.txt           # 播客脚本提示词模板
├── .env                    # API 配置（需自行创建）
├── .gitignore              # Git 忽略配置
└── README.md
```

## 更新日志

### v1.5+
- ✨ 新增 **Audio8-ASR-0.1B-onnx** 第三种 ASR 模型，支持跨平台（Intel Mac / Linux / Windows）
- ✨ 新增 **ASR 噪声清洗** 功能，自动消除连续重复字、逗号分隔重复模式
- ✨ 优化成文提示词，要求按 2-5 句话适当分段，提升可读性
- 🐛 修复 DeepSeek API Key 误提交问题（添加 `.gitignore`）

## License

MIT
