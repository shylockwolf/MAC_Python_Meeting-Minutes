# MAC Python Meeting Minutes

macOS 平台的会议纪要自动生成工具，基于 mlx-whisper 语音转录 + DeepSeek 大模型智能总结，一键完成从音频到 PDF 会议纪要的全流程。

## 功能特性

- **语音转录**：使用 mlx-whisper 将音频文件转录为带时间戳的文本（输出 `.txt`）
- **智能成文**：调用 DeepSeek 大模型对转录文本进行段落合并、错别字修正、格式优化
- **会议纪要**：基于成文内容生成结构化会议纪要（输出 `.pdf`）
- **播客脚本**：将会议内容转化为双人对话形式的播客脚本（输出 `.txt`）
- **一键模式**：转录 → 成文 → 纪要 PDF，一键完成全流程

## 环境要求

- macOS（Apple Silicon 推荐）
- Python 3.10+
- ffmpeg（用于音频处理）
- mlx-whisper 模型文件

## 安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 ffmpeg（如未安装）
brew install ffmpeg
```

## 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
```

将 mlx-whisper 模型放置到指定路径，默认为 `../../myMLX/whisper-small-mlx`。

## 使用

```bash
python meeting_minutes.py
```

### 操作流程

1. **选择音频文件**：点击"打开"按钮选择音频文件
2. **一键生成**：点击"一键生成"按钮，自动完成转录 → 成文 → 纪要 PDF
3. **分步操作**：也可单独使用各功能按钮

### 输出文件

| 功能 | 输出文件 | 说明 |
|------|---------|------|
| 语音转录 | `文件名.txt` | 带时间戳的原始转录文本 |
| 成文处理 | `文件名_成文.txt` | 段落合并、格式优化后的文本 |
| 会议纪要 | `文件名_纪要.pdf` | 结构化会议纪要（PDF格式） |
| 播客脚本 | `文件名_播客.txt` | 双人对话形式播客脚本 |

## 模型配置

- **转录模型**：mlx-whisper（本地运行）
- **默认 LLM**：`deepseek-v4-flash`
- **播客生成**：`deepseek-v4-pro`
- **最大输出 token**：384,000
- **文本分块大小**：100,000 字符/段

## 提示词模板

- `会议纪要提示词.txt`：会议纪要生成的提示词模板
- `播客提示词.txt`：播客脚本生成的提示词模板

可根据需要自定义修改。

## 项目结构

```
.
├── meeting_minutes.py      # 主程序（GUI）
├── md_to_pdf_gui.py        # MD转PDF独立工具（GUI）
├── md_to_pdf.py            # MD转PDF独立工具（CLI）
├── requirements.txt        # Python 依赖
├── 会议纪要提示词.txt        # 会议纪要提示词模板
├── 播客提示词.txt           # 播客脚本提示词模板
└── .env                    # API 配置（需自行创建）
```

## License

MIT
