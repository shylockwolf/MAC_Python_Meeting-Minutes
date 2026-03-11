import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import os
import time
import subprocess
import tempfile
import shutil
import multiprocessing as mp
from multiprocessing import Queue, Process
import mlx_whisper
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

def get_audio_duration(audio_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
    except Exception as e:
        pass
    return None

def split_audio_with_progress(audio_path, log_callback=None, segment_duration=900):
    """将音频文件分割成指定时长的片段，默认15分钟(900秒)，带进度显示"""
    if log_callback:
        log_callback("正在获取音频时长...")
    
    duration = get_audio_duration(audio_path)
    if duration is None:
        if log_callback:
            log_callback("错误: 无法获取音频时长")
        return None
    
    if log_callback:
        log_callback(f"音频时长: {duration:.2f}秒")
    
    if duration <= segment_duration:
        if log_callback:
            log_callback(f"音频时长较短({duration:.0f}秒 <= {segment_duration}秒)，无需分割")
        return [(audio_path, 0, duration)]
    
    temp_dir = tempfile.mkdtemp()
    segments = []
    num_segments = int(duration // segment_duration) + (1 if duration % segment_duration > 0 else 0)
    
    if log_callback:
        log_callback(f"开始分割音频，共 {num_segments} 个片段...")
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration)
        segment_duration_actual = end_time - start_time
        
        segment_path = os.path.join(temp_dir, f"segment_{i:03d}.wav")
        
        if log_callback:
            log_callback(f"  正在生成片段 {i+1}/{num_segments} ({start_time:.0f}s - {end_time:.0f}s)...")
        
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(segment_duration_actual),
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
            segment_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and os.path.exists(segment_path):
            segments.append((segment_path, start_time, end_time))
            if log_callback:
                log_callback(f"  ✓ 片段 {i+1}/{num_segments} 完成")
        else:
            if log_callback:
                log_callback(f"  ✗ 片段 {i+1}/{num_segments} 失败")
                if result.stderr:
                    log_callback(f"    错误: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return None
    
    if log_callback:
        log_callback(f"音频分割完成，共 {len(segments)} 个片段")
    
    return segments

def transcribe_worker(audio_path, model_path, result_queue, log_queue, segment_index=None, time_offset=0):
    try:
        prefix = f"[片段{segment_index+1}] " if segment_index is not None else ""
        log_queue.put(f"{prefix}调用 mlx_whisper.transcribe()...")
        
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            language="zh",
            word_timestamps=True,
            fp16=True
        )
        
        # 调整时间戳
        if time_offset > 0 and 'segments' in result:
            for segment in result['segments']:
                segment['start'] += time_offset
                segment['end'] += time_offset
        
        log_queue.put(f"{prefix}转录函数调用完成，正在处理结果...")
        result_queue.put(("success", result, segment_index))
        
    except Exception as e:
        log_queue.put(f"{prefix}转录过程出错: {e}")
        result_queue.put(("error", str(e), segment_index))

class MeetingMinutesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("会议纪要 - 语音转文字")
        self.root.geometry("800x600")
        
        self.transcribing = False
        self.current_audio_path = None
        self.model_path = "/Users/noone/Movies/myMLX/whisper-small-mlx"
        self.transcribe_process = None
        self.log_queue = None
        self.result_queue = None
        self.start_time = None
        self.audio_duration = None
        self.last_log_time = None
        self.segments = []
        self.current_segment = 0
        self.total_segments = 0
        self.all_results = []
        self.temp_dir = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # 第一行按钮 - 音频处理
        audio_frame = tk.Frame(self.root)
        audio_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.open_button = tk.Button(audio_frame, text="打开音频文件", command=self.open_file, width=15)
        self.open_button.pack(side=tk.LEFT, padx=5)

        self.start_button = tk.Button(audio_frame, text="开始转录", command=self.start_transcription, width=15, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(audio_frame, text="中断", command=self.stop_transcription, width=15, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.one_click_button = tk.Button(audio_frame, text="一键生成会议纪要", command=self.one_click_generate, width=15, state=tk.DISABLED)
        self.one_click_button.pack(side=tk.LEFT, padx=5)
        
        # 第二行按钮 - 文本处理
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.open_text_button = tk.Button(text_frame, text="打开文本文件", command=self.open_text_file, width=15)
        self.open_text_button.pack(side=tk.LEFT, padx=5)
        
        self.format_button = tk.Button(text_frame, text="开始成文", command=self.start_formatting, width=15, state=tk.DISABLED)
        self.format_button.pack(side=tk.LEFT, padx=5)
        
        self.minutes_button = tk.Button(text_frame, text="生成会议纪要", command=self.generate_minutes, width=15, state=tk.DISABLED)
        self.minutes_button.pack(side=tk.LEFT, padx=5)

        self.podcast_button = tk.Button(text_frame, text="生成播客", command=self.generate_podcast, width=15, state=tk.DISABLED)
        self.podcast_button.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="请选择音频文件或文本文件", pady=5)
        self.status_label.pack()
        
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Arial", 10))
        self.log_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def open_file(self):
        if self.transcribing:
            self.log("警告: 正在处理中，请先中断当前操作")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.current_audio_path = file_path
            self.status_label.config(text=f"已选择: {os.path.basename(file_path)}")
            self.log(f"已选择音频文件: {file_path}")
            
            duration = get_audio_duration(file_path)
            if duration:
                self.audio_duration = duration
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self.log(f"音频时长: {minutes}分{seconds}秒 ({duration:.2f}秒)")
                
                estimated_time = duration / 10
                est_minutes = int(estimated_time // 60)
                est_seconds = int(estimated_time % 60)
                self.log(f"预估处理时间: 约{est_minutes}分{est_seconds}秒 (基于10x实时速度)")
            else:
                self.audio_duration = None
                self.log("无法获取音频时长信息")
            
            self.start_button.config(state=tk.NORMAL)
            self.one_click_button.config(state=tk.NORMAL)

    def one_click_generate(self):
        """一键生成：转录 -> 成文 -> 生成会议纪要"""
        if not self.current_audio_path:
            self.log("警告: 请先选择音频文件")
            return

        # 获取 API 密钥
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key or api_key == 'your_deepseek_api_key_here':
            self.log("错误: 请先配置 DeepSeek API 密钥到 .env 文件")
            return

        # 加载提示词文件
        prompt_file = Path(__file__).parent / "会议纪要提示词.txt"
        if not prompt_file.exists():
            self.log(f"错误: 找不到提示词文件: {prompt_file}")
            return

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.minutes_prompt_template = f.read()
        except Exception as e:
            self.log(f"错误: 读取提示词文件失败: {e}")
            return

        self.log(f"\n{'='*50}")
        self.log("开始一键生成流程...")
        self.log("步骤: 1.转录 -> 2.成文 -> 3.生成会议纪要")
        self.log(f"{'='*50}\n")

        # 禁用按钮
        self.one_click_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.open_button.config(state=tk.DISABLED)
        self.open_text_button.config(state=tk.DISABLED)
        self.format_button.config(state=tk.DISABLED)
        self.minutes_button.config(state=tk.DISABLED)

        # 设置标志
        self.one_click_mode = True

        # 开始转录
        self.start_transcription()

    def _on_transcription_complete_for_one_click(self):
        """转录完成后自动开始成文"""
        if hasattr(self, 'one_click_mode') and self.one_click_mode:
            self.log(f"\n{'='*50}")
            self.log("步骤 1/3 完成: 转录完成")
            self.log("开始步骤 2/3: 成文处理...")
            self.log(f"{'='*50}\n")

            # 将转录结果保存为当前文本内容
            if hasattr(self, 'transcription_result'):
                self.current_text_content = self.transcription_result
                self.formatted_text_content = None  # 重置成文内容

                # 开始成文
                self._call_deepseek_for_formatting_one_click()
            else:
                self.log("错误: 转录结果为空")
                self._reset_one_click_mode()

    def _call_deepseek_for_formatting_one_click(self):
        """一键模式下的成文处理"""
        thread = threading.Thread(target=self._formatting_worker_one_click)
        thread.daemon = True
        thread.start()

    def _formatting_worker_one_click(self):
        """成文工作线程（一键模式）"""
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

            self.log("正在连接 DeepSeek API...")

            client = OpenAI(
                api_key=api_key,
                base_url=api_url
            )

            # 估算每段最大字符数
            MAX_CHARS_PER_CHUNK = 15000

            content = self.current_text_content
            content_length = len(content)

            self.log(f"原始文本长度: {content_length} 字符")

            # 如果文本较短，直接处理
            if content_length <= MAX_CHARS_PER_CHUNK:
                self.log("文本较短，直接处理...")
                formatted_text = self._process_formatting_chunk(client, model, content, 1, 1)
            else:
                # 分段处理
                chunks = self._split_text_into_chunks(content, MAX_CHARS_PER_CHUNK)
                total_chunks = len(chunks)
                self.log(f"文本较长，分成 {total_chunks} 段处理...")

                formatted_chunks = []
                for i, chunk in enumerate(chunks, 1):
                    self.log(f"\n处理第 {i}/{total_chunks} 段...")
                    formatted_chunk = self._process_formatting_chunk(client, model, chunk, i, total_chunks)
                    formatted_chunks.append(formatted_chunk)

                # 合并所有段
                self.log(f"\n合并 {total_chunks} 段结果...")
                formatted_text = '\n\n'.join(formatted_chunks)

            self.log(f"✓ 成文处理完成，总长度: {len(formatted_text)} 字符")

            # 保存成文后的内容
            self.formatted_text_content = formatted_text

            # 保存成文文件
            self._save_formatted_text_one_click(formatted_text)

            # 继续生成会议纪要
            self.root.after(0, self._on_formatting_complete_for_one_click)

        except Exception as e:
            self.log(f"成文处理失败: {e}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            self.log(f"错误: 成文处理失败: {e}")
            self.root.after(0, self._reset_one_click_mode)

    def _process_formatting_chunk(self, client, model, chunk, chunk_index, total_chunks):
        """处理单个文本块（成文）"""
        prompt = """以下是音频转文字的原始文件，请做一下处理：
1、保留时间戳，格式必须统一为 [MM:SS-MM:SS]，如 [15:09-16:34]
2、每个时间戳前面加4个空格，时间戳与文本之间不加粗、直接连接，如：    [15:09-16:34] 他说就是我现在找的 Sales Leads
3、按语义拼接成合理的句子和段落，并增加标点符号
4、按简体中文输出
5、保持段落结构清晰

原始文本：
"""

        full_prompt = prompt + chunk

        self.log(f"  发送请求 (长度: {len(full_prompt)} 字符)...")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本编辑助手，擅长整理和优化语音转文字的文本。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=16000,
                timeout=180
            )

            formatted_text = response.choices[0].message.content

            if not formatted_text or len(formatted_text.strip()) == 0:
                self.log("  警告: API 返回内容为空")
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    formatted_text = response.choices[0].message.reasoning_content

            self.log(f"  ✓ 完成，返回长度: {len(formatted_text)} 字符")
            return formatted_text

        except Exception as e:
            self.log(f"  ✗ 调用失败: {e}")
            # 尝试使用备用模型
            if model != 'deepseek-chat':
                self.log("  尝试使用 deepseek-chat 模型...")
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[
                        {"role": "system", "content": "你是一个专业的文本编辑助手，擅长整理和优化语音转文字的文本。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=16000,
                    timeout=180
                )
                formatted_text = response.choices[0].message.content
                self.log(f"  ✓ 备用模型完成，返回长度: {len(formatted_text)} 字符")
                return formatted_text
            else:
                raise

    def _save_formatted_text_one_click(self, formatted_text):
        """一键模式下保存成文文件"""
        try:
            if self.current_audio_path:
                audio_path = Path(self.current_audio_path)
                output_path = audio_path.parent / f"{audio_path.stem}_成文.txt"

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)

                self.log(f"\n✓ 成文结果已保存到: {output_path}")

                # 设置当前文本路径（用于保存会议纪要）
                self.current_text_path = str(audio_path.parent / f"{audio_path.stem}.txt")

        except Exception as e:
            self.log(f"保存成文文件失败: {e}")
            # 不中断流程，继续生成会议纪要

    def _on_formatting_complete_for_one_click(self):
        """成文完成后自动生成会议纪要"""
        if hasattr(self, 'one_click_mode') and self.one_click_mode:
            self.log(f"\n{'='*50}")
            self.log("步骤 2/3 完成: 成文处理完成")
            self.log("开始步骤 3/3: 生成会议纪要...")
            self.log(f"{'='*50}\n")

            # 生成会议纪要
            self._call_deepseek_for_minutes_one_click()

    def _call_deepseek_for_minutes_one_click(self):
        """一键模式下的会议纪要生成"""
        thread = threading.Thread(target=self._minutes_worker_one_click)
        thread.daemon = True
        thread.start()

    def _minutes_worker_one_click(self):
        """会议纪要生成工作线程（一键模式）"""
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

            self.log("正在连接 DeepSeek API...")

            client = OpenAI(
                api_key=api_key,
                base_url=api_url
            )

            # 使用成文后的内容
            content = self.formatted_text_content
            content_length = len(content)

            self.log(f"文本长度: {content_length} 字符")

            # 64K tokens 约等于 48K 汉字
            MAX_CHARS_THRESHOLD = 45000

            # 如果文本较短，直接处理
            if content_length <= MAX_CHARS_THRESHOLD:
                self.log("文本较短，直接生成会议纪要...")
                minutes_text = self._process_minutes_full(client, model, self.minutes_prompt_template, content)
                self._save_minutes(minutes_text)
            else:
                # 两阶段处理
                self.log(f"文本较长({content_length}字符)，使用两阶段处理...")

                # 第一阶段：分段生成结构摘要
                chunk_size = 15000
                chunks = self._split_text_into_chunks(content, chunk_size)
                total_chunks = len(chunks)

                self.log(f"\n第一阶段：生成段落摘要，共 {total_chunks} 段...")

                summaries = []

                for i, chunk in enumerate(chunks, 1):
                    self.log(f"\n  处理第 {i}/{total_chunks} 段...")
                    summary = self._generate_chunk_summary(client, model, chunk, i)
                    summaries.append(summary)

                # 第二阶段：全局整合
                self.log(f"\n第二阶段：全局整合 {total_chunks} 段摘要...")
                all_summaries = "\n\n".join([f"=== 第 {i+1} 段摘要 ===\n{s}" for i, s in enumerate(summaries)])
                minutes_text = self._integrate_summaries(client, model, self.minutes_prompt_template, all_summaries)

                self.log(f"✓ 会议纪要生成完成，总长度: {len(minutes_text)} 字符")

                # 保存会议纪要
                self._save_minutes(minutes_text)

            self.log(f"\n{'='*50}")
            self.log("✓✓✓ 一键生成流程全部完成！✓✓✓")
            self.log(f"{'='*50}\n")

            # 重置一键模式
            self.root.after(0, self._reset_one_click_mode)

        except Exception as e:
            self.log(f"生成会议纪要失败: {e}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            self.log(f"错误: 生成会议纪要失败: {e}")
            self.root.after(0, self._reset_one_click_mode)

    def _reset_one_click_mode(self):
        """重置一键模式状态"""
        self.one_click_mode = False
        self.open_button.config(state=tk.NORMAL)
        self.open_text_button.config(state=tk.NORMAL)
        if self.current_audio_path:
            self.start_button.config(state=tk.NORMAL)
            self.one_click_button.config(state=tk.NORMAL)
        if hasattr(self, 'current_text_content'):
            self.format_button.config(state=tk.NORMAL)
            self.minutes_button.config(state=tk.NORMAL)

    def open_text_file(self):
        if self.transcribing:
            self.log("警告: 正在处理中，请先中断当前操作")
            return

        file_path = filedialog.askopenfilename(
            title="选择文本文件",
            filetypes=[
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )

        if file_path:
            self.current_text_path = file_path
            self.status_label.config(text=f"已选择文本: {os.path.basename(file_path)}")
            self.log(f"已选择文本文件: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.current_text_content = content
                    lines = content.count('\n') + 1
                    chars = len(content)
                    self.log(f"文件行数: {lines}, 字符数: {chars}")
                    self.format_button.config(state=tk.NORMAL)
                    self.minutes_button.config(state=tk.NORMAL)
                    self.podcast_button.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"读取文件失败: {e}")
                self.log(f"错误: 读取文件失败: {e}")
                
    def start_formatting(self):
        if not hasattr(self, 'current_text_content'):
            self.log("警告: 请先选择文本文件")
            return

        # 获取 API 密钥
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key or api_key == 'your_deepseek_api_key_here':
            self.log("错误: 请先配置 DeepSeek API 密钥到 .env 文件")
            return
        
        self.log(f"\n{'='*50}")
        self.log("开始成文处理...")
        self.log(f"{'='*50}\n")
        
        # 禁用按钮
        self.format_button.config(state=tk.DISABLED)
        self.open_text_button.config(state=tk.DISABLED)
        
        # 在新线程中调用 API
        thread = threading.Thread(target=self._call_deepseek_for_formatting)
        thread.daemon = True
        thread.start()
    
    def _call_deepseek_for_formatting(self):
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
            
            self.log("正在连接 DeepSeek API...")
            
            client = OpenAI(
                api_key=api_key,
                base_url=api_url
            )
            
            # 估算每段最大字符数（预留空间给提示词和输出）
            # DeepSeek 最大 32K tokens，约 24K 汉字
            # 预留 4K 给提示词和输出，每段约 20K 字符
            MAX_CHARS_PER_CHUNK = 15000  # 保守估计，每段1.5万字符
            
            content = self.current_text_content
            content_length = len(content)
            
            self.log(f"原始文本长度: {content_length} 字符")
            
            # 如果文本较短，直接处理
            if content_length <= MAX_CHARS_PER_CHUNK:
                self.log("文本较短，直接处理...")
                formatted_text = self._process_text_chunk(client, model, content, 1, 1)
            else:
                # 分段处理
                chunks = self._split_text_into_chunks(content, MAX_CHARS_PER_CHUNK)
                total_chunks = len(chunks)
                self.log(f"文本较长，分成 {total_chunks} 段处理...")
                
                formatted_chunks = []
                for i, chunk in enumerate(chunks, 1):
                    self.log(f"\n处理第 {i}/{total_chunks} 段...")
                    formatted_chunk = self._process_text_chunk(client, model, chunk, i, total_chunks)
                    formatted_chunks.append(formatted_chunk)
                
                # 合并所有段
                self.log(f"\n合并 {total_chunks} 段结果...")
                formatted_text = "\n\n".join(formatted_chunks)
            
            self.log(f"✓ 成文处理完成，总长度: {len(formatted_text)} 字符")
            
            # 保存成文后的文件
            self._save_formatted_text(formatted_text)
            
            # 更新状态
            self.root.after(0, lambda: self.minutes_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.open_text_button.config(state=tk.NORMAL))
            
        except Exception as e:
            self.log(f"成文处理失败: {e}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            self.log(f"错误: 成文处理失败: {e}")
            self.root.after(0, lambda: self.format_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.open_text_button.config(state=tk.NORMAL))
    
    def _split_text_into_chunks(self, text, max_chars):
        """将文本分割成多个块，尽量在句子边界处分割"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line)
            
            # 如果当前行加上去会超过限制，先保存当前块
            if current_length + line_length > max_chars and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(line)
            current_length += line_length + 1  # +1 for newline
        
        # 添加最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _process_text_chunk(self, client, model, chunk, chunk_index, total_chunks):
        """处理单个文本块"""
        prompt = """以下是音频转文字的原始文件，请做一下处理：
1、保留时间戳，格式必须统一为 [MM:SS-MM:SS]，如 [15:09-16:34]
2、每个时间戳前面加4个空格，时间戳与文本之间不加粗、直接连接，如：    [15:09-16:34] 他说就是我现在找的 Sales Leads
3、按语义拼接成合理的句子和段落，并增加标点符号
4、按简体中文输出
5、保持段落结构清晰

原始文本：
"""
        
        full_prompt = prompt + chunk
        
        self.log(f"  发送请求 (长度: {len(full_prompt)} 字符)...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本编辑助手，擅长整理和优化语音转文字的文本。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=16000,  # 增加到最大限制
                timeout=180  # 增加到3分钟
            )
            
            formatted_text = response.choices[0].message.content
            
            if not formatted_text or len(formatted_text.strip()) == 0:
                self.log("  警告: API 返回内容为空")
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    formatted_text = response.choices[0].message.reasoning_content
            
            self.log(f"  ✓ 完成，返回长度: {len(formatted_text)} 字符")
            return formatted_text
            
        except Exception as e:
            self.log(f"  ✗ 调用失败: {e}")
            # 尝试使用备用模型
            if model != 'deepseek-chat':
                self.log("  尝试使用 deepseek-chat 模型...")
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[
                        {"role": "system", "content": "你是一个专业的文本编辑助手，擅长整理和优化语音转文字的文本。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=16000,
                    timeout=180
                )
                formatted_text = response.choices[0].message.content
                self.log(f"  ✓ 备用模型完成，返回长度: {len(formatted_text)} 字符")
                return formatted_text
            else:
                raise
    
    def _save_formatted_text(self, formatted_text):
        try:
            # 生成输出文件名：源文件名字_成文.txt
            input_path = Path(self.current_text_path)
            output_path = input_path.parent / f"{input_path.stem}_成文.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            self.log(f"\n✓ 成文结果已保存到: {output_path}")
            
            # 保存成文后的内容，供生成会议纪要使用
            self.formatted_text_content = formatted_text
            self.formatted_text_path = str(output_path)
            
        except Exception as e:
            self.log(f"保存成文文件失败: {e}")
            raise
        
    def generate_minutes(self):
        if not hasattr(self, 'current_text_content'):
            self.log("警告: 请先选择文本文件")
            return

        # 获取 API 密钥
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key or api_key == 'your_deepseek_api_key_here':
            self.log("错误: 请先配置 DeepSeek API 密钥到 .env 文件")
            return

        # 加载提示词文件
        prompt_file = Path(__file__).parent / "会议纪要提示词.txt"
        if not prompt_file.exists():
            self.log(f"错误: 找不到提示词文件: {prompt_file}")
            return

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            self.log(f"错误: 读取提示词文件失败: {e}")
            return
        
        self.log(f"\n{'='*50}")
        self.log("生成会议纪要...")
        self.log(f"{'='*50}\n")
        self.log(f"已加载提示词模板: {prompt_file}")
        
        # 禁用按钮
        self.minutes_button.config(state=tk.DISABLED)
        self.open_text_button.config(state=tk.DISABLED)
        self.format_button.config(state=tk.DISABLED)
        
        # 在新线程中调用 API
        thread = threading.Thread(target=self._call_deepseek_for_minutes, args=(prompt_template,))
        thread.daemon = True
        thread.start()
    
    def _call_deepseek_for_minutes(self, prompt_template):
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')
            model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
            
            self.log("正在连接 DeepSeek API...")
            
            client = OpenAI(
                api_key=api_key,
                base_url=api_url
            )
            
            # 使用成文后的内容，如果没有则使用原始内容
            content = getattr(self, 'formatted_text_content', self.current_text_content)
            content_length = len(content)
            
            self.log(f"文本长度: {content_length} 字符")
            
            # 64K tokens 约等于 48K 汉字（1 token ≈ 0.75 汉字）
            # 预留空间给输出，设置阈值为 45000 字符
            MAX_CHARS_THRESHOLD = 45000
            
            # 如果文本较短，直接处理
            if content_length <= MAX_CHARS_THRESHOLD:
                self.log("文本较短，直接生成会议纪要...")
                minutes_text = self._process_minutes_full(client, model, prompt_template, content)
                self._save_minutes(minutes_text)
            else:
                # 两阶段处理：先生成摘要，再整合
                self.log(f"文本较长({content_length}字符)，使用两阶段处理...")
                
                # 第一阶段：分段生成结构摘要
                chunk_size = 15000  # 每段约15000字符
                chunks = self._split_text_into_chunks(content, chunk_size)
                total_chunks = len(chunks)
                
                self.log(f"\n第一阶段：生成段落摘要，共 {total_chunks} 段...")
                
                summaries = []
                
                for i, chunk in enumerate(chunks, 1):
                    self.log(f"\n  处理第 {i}/{total_chunks} 段...")
                    summary = self._generate_chunk_summary(client, model, chunk, i)
                    summaries.append(summary)
                
                # 第二阶段：全局整合
                self.log(f"\n第二阶段：全局整合 {total_chunks} 段摘要...")
                all_summaries = "\n\n".join([f"=== 第 {i+1} 段摘要 ===\n{s}" for i, s in enumerate(summaries)])
                minutes_text = self._integrate_summaries(client, model, prompt_template, all_summaries)
                
                self.log(f"✓ 会议纪要生成完成，总长度: {len(minutes_text)} 字符")
                
                # 保存会议纪要
                self._save_minutes(minutes_text)
            
            # 更新状态
            self.root.after(0, lambda: self.open_text_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.format_button.config(state=tk.NORMAL))

        except Exception as e:
            self.log(f"生成会议纪要失败: {e}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            self.log(f"错误: 生成会议纪要失败: {e}")
            self.root.after(0, lambda: self.open_text_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.format_button.config(state=tk.NORMAL))

    def generate_podcast(self):
        if not hasattr(self, 'current_text_content'):
            self.log("警告: 请先选择文本文件")
            return

        # 获取 API 密钥
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key or api_key == 'your_deepseek_api_key_here':
            self.log("错误: 请先配置 DeepSeek API 密钥到 .env 文件")
            return

        # 加载提示词文件
        prompt_file = Path(__file__).parent / "播客提示词.txt"
        if not prompt_file.exists():
            self.log(f"错误: 找不到提示词文件: {prompt_file}")
            return

        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except Exception as e:
            self.log(f"错误: 读取提示词文件失败: {e}")
            return

        self.log(f"\n{'='*50}")
        self.log("生成播客脚本...")
        self.log(f"{'='*50}\n")
        self.log(f"已加载提示词模板: {prompt_file}")

        # 禁用按钮
        self.podcast_button.config(state=tk.DISABLED)
        self.open_text_button.config(state=tk.DISABLED)
        self.format_button.config(state=tk.DISABLED)
        self.minutes_button.config(state=tk.DISABLED)

        # 在新线程中调用 API
        thread = threading.Thread(target=self._call_deepseek_for_podcast, args=(prompt_template,))
        thread.daemon = True
        thread.start()

    def _call_deepseek_for_podcast(self, prompt_template):
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            api_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')
            # 固定使用 deepseek-reasoner 模型
            model = 'deepseek-reasoner'

            self.log("正在连接 DeepSeek API...")

            client = OpenAI(
                api_key=api_key,
                base_url=api_url
            )

            # 使用成文后的内容，如果没有则使用原始内容
            content = getattr(self, 'formatted_text_content', self.current_text_content)
            content_length = len(content)

            self.log(f"文本长度: {content_length} 字符")

            # 16K tokens 约等于 12K 汉字（1 token ≈ 0.75 汉字）
            # 预留提示词空间，设置阈值为 10000 字符
            MAX_CHARS_PER_CHUNK = 10000

            if content_length <= MAX_CHARS_PER_CHUNK:
                # 文本较短，直接处理
                self.log("文本较短，直接生成播客脚本...")
                podcast_text = self._process_podcast_chunk(client, model, prompt_template, content, 1, 1)
            else:
                # 按段落分割处理
                self.log(f"文本较长({content_length}字符)，分段处理...")
                chunks = self._split_text_by_paragraphs(content, MAX_CHARS_PER_CHUNK)
                total_chunks = len(chunks)
                self.log(f"分成 {total_chunks} 段处理...")

                podcast_parts = []
                for i, chunk in enumerate(chunks, 1):
                    self.log(f"\n处理第 {i}/{total_chunks} 段...")
                    part = self._process_podcast_chunk(client, model, prompt_template, chunk, i, total_chunks)
                    podcast_parts.append(part)
                    if i < total_chunks:
                        self.log(f"  等待 1 秒避免请求过快...")
                        time.sleep(1)

                # 拼接所有部分
                self.log(f"\n拼接 {total_chunks} 段结果...")
                podcast_text = '\n\n'.join(podcast_parts)

            self.log(f"✓ 播客脚本生成完成，总长度: {len(podcast_text)} 字符")

            # 保存播客脚本
            self._save_podcast(podcast_text)

            # 更新状态
            self.root.after(0, lambda: self.open_text_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.format_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.minutes_button.config(state=tk.NORMAL))

        except Exception as e:
            self.log(f"生成播客失败: {e}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            self.log(f"错误: 生成播客失败: {e}")
            self.root.after(0, lambda: self.open_text_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.format_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.minutes_button.config(state=tk.NORMAL))

    def _split_text_by_paragraphs(self, text, max_chars):
        """按段落分割文本，确保每段不超过最大字符数"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前段落本身超过限制，需要进一步分割
            if len(para) > max_chars:
                # 先保存当前积累的块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 按句子分割长段落
                sentences = para.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
                temp_chunk = ""

                for sent in sentences:
                    if len(temp_chunk) + len(sent) > max_chars:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sent
                    else:
                        temp_chunk += sent

                if temp_chunk:
                    if current_chunk:
                        current_chunk += "\n\n" + temp_chunk
                    else:
                        current_chunk = temp_chunk
            else:
                # 检查加入当前段落后是否超过限制
                if len(current_chunk) + len(para) + 2 > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para

        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _process_podcast_chunk(self, client, model, prompt_template, content, chunk_index, total_chunks):
        """处理单个段落生成播客脚本"""
        full_prompt = prompt_template + content

        self.log(f"  发送请求 (长度: {len(full_prompt)} 字符)...")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一位资深的播客内容策划和编辑专家，擅长将会议或访谈内容转化为精彩的播客节目。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.8,
                max_tokens=16000,
                timeout=180
            )

            podcast_text = response.choices[0].message.content

            if not podcast_text or len(podcast_text.strip()) == 0:
                self.log("  警告: API 返回内容为空")
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    podcast_text = response.choices[0].message.reasoning_content

            self.log(f"  ✓ 第 {chunk_index}/{total_chunks} 段完成，返回长度: {len(podcast_text)} 字符")
            return podcast_text

        except Exception as e:
            self.log(f"  ✗ 第 {chunk_index}/{total_chunks} 段调用失败: {e}")
            raise

    def _save_podcast(self, podcast_text):
        """保存播客脚本到文本文件"""
        try:
            # 生成输出文件名：源文件名字_播客.txt
            input_path = Path(self.current_text_path)
            output_path = input_path.parent / f"{input_path.stem}_播客.txt"

            # 直接覆盖保存
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(podcast_text)

            self.log(f"\n✓ 播客脚本已保存到: {output_path}")

        except Exception as e:
            self.log(f"保存播客脚本失败: {e}")
            raise

    def _generate_chunk_summary(self, client, model, chunk, chunk_index):
        """生成段落结构摘要"""
        summary_prompt = """请总结本段会议内容，按以下格式输出：

- 本段议题：[简要描述本段讨论的议题]
- 本段形成的结论：[列出本段达成的结论，如无则写"无"]
- 本段产生的行动项：[列出具体的行动项，格式：任务 - 负责人 - 截止时间，如无则写"无"]
- 未解决问题：[列出本段提出但未解决的问题，如无则写"无"]

会议内容：
"""
        
        full_prompt = summary_prompt + chunk
        
        self.log(f"    发送摘要请求 (长度: {len(full_prompt)} 字符)...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的会议分析助手，擅长提取会议关键信息。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=8000,
                timeout=120
            )
            
            summary = response.choices[0].message.content
            
            if not summary or len(summary.strip()) == 0:
                self.log("    警告: API 返回内容为空")
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    summary = response.choices[0].message.reasoning_content
            
            self.log(f"    ✓ 摘要完成，长度: {len(summary)} 字符")
            return summary
            
        except Exception as e:
            self.log(f"    ✗ 摘要生成失败: {e}")
            # 尝试使用备用模型
            if model != 'deepseek-chat':
                self.log("    尝试使用 deepseek-chat 模型...")
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[
                        {"role": "system", "content": "你是一个专业的会议分析助手，擅长提取会议关键信息。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                    timeout=120
                )
                summary = response.choices[0].message.content
                self.log(f"    ✓ 备用模型完成，长度: {len(summary)} 字符")
                return summary
            else:
                raise
    
    def _integrate_summaries(self, client, model, prompt_template, all_summaries):
        """整合所有摘要生成完整会议纪要"""
        integrate_prompt = """以下是会议的分段摘要，请合并为一份完整的会议纪要：

要求：
1. 去重行动项（相同或相似的任务只保留一条）
2. 合并重复议题（将相同主题的讨论合并）
3. 保留所有决策（不要遗漏任何决定）
4. 按逻辑重组内容，确保条理清晰
5. 使用正式的会议纪要格式

分段摘要：
"""
        
        full_prompt = integrate_prompt + all_summaries + "\n\n" + prompt_template
        
        self.log(f"  发送整合请求 (摘要总长度: {len(all_summaries)} 字符)...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的会议记录员，擅长整合信息并生成结构化的会议纪要。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=16000,
                timeout=180
            )
            
            minutes_text = response.choices[0].message.content
            
            if not minutes_text or len(minutes_text.strip()) == 0:
                self.log("  警告: API 返回内容为空")
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    minutes_text = response.choices[0].message.reasoning_content
            
            self.log(f"  ✓ 整合完成，会议纪要长度: {len(minutes_text)} 字符")
            return minutes_text
            
        except Exception as e:
            self.log(f"  ✗ 整合失败: {e}")
            # 尝试使用备用模型
            if model != 'deepseek-chat':
                self.log("  尝试使用 deepseek-chat 模型...")
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[
                        {"role": "system", "content": "你是一个专业的会议记录员，擅长整合信息并生成结构化的会议纪要。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=16000,
                    timeout=180
                )
                minutes_text = response.choices[0].message.content
                self.log(f"  ✓ 备用模型完成，长度: {len(minutes_text)} 字符")
                return minutes_text
            else:
                raise
    
    def _process_minutes_full(self, client, model, prompt_template, content):
        """直接处理完整文本生成会议纪要（短文本用）"""
        full_prompt = prompt_template + content
        
        self.log(f"  发送请求 (长度: {len(full_prompt)} 字符)...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的会议记录员，擅长从会议文本中提取关键信息并生成结构化的会议纪要。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=16000,
                timeout=180
            )
            
            minutes_text = response.choices[0].message.content
            
            if not minutes_text or len(minutes_text.strip()) == 0:
                self.log("  警告: API 返回内容为空")
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    minutes_text = response.choices[0].message.reasoning_content
            
            self.log(f"  ✓ 完成，返回长度: {len(minutes_text)} 字符")
            return minutes_text
            
        except Exception as e:
            self.log(f"  ✗ 调用失败: {e}")
            # 尝试使用备用模型
            if model != 'deepseek-chat':
                self.log("  尝试使用 deepseek-chat 模型...")
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[
                        {"role": "system", "content": "你是一个专业的会议记录员，擅长从会议文本中提取关键信息并生成结构化的会议纪要。"},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=16000,
                    timeout=180
                )
                minutes_text = response.choices[0].message.content
                self.log(f"  ✓ 备用模型完成，返回长度: {len(minutes_text)} 字符")
                return minutes_text
            else:
                raise
    
    def _save_minutes(self, minutes_text):
        """保存会议纪要到 Markdown 文件"""
        try:
            # 生成输出文件名：源文件名字_纪要.md
            input_path = Path(self.current_text_path)
            output_path = input_path.parent / f"{input_path.stem}_纪要.md"
            
            # 如果文件已存在，添加序号
            counter = 1
            original_output_path = output_path
            while output_path.exists():
                output_path = input_path.parent / f"{input_path.stem}_纪要_{counter}.md"
                counter += 1
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(minutes_text)
            
            self.log(f"\n✓ 会议纪要已保存到: {output_path}")
            
            if output_path != original_output_path:
                self.log(f"  (原文件已存在，使用新文件名)")
            
        except Exception as e:
            self.log(f"保存会议纪要失败: {e}")
            raise
            
    def start_transcription(self):
        if not self.current_audio_path:
            self.log("警告: 请先选择音频文件")
            return

        if self.transcribing:
            return

        self.transcribing = True
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.open_button.config(state=tk.DISABLED)
        self.start_time = time.time()

        self.log(f"\n{'='*50}")
        self.log(f"开始处理音频: {self.current_audio_path}")
        self.log(f"模型路径: {self.model_path}")
        self.log(f"{'='*50}\n")

        if not os.path.exists(self.model_path):
            self.log(f"错误: 模型路径不存在: {self.model_path}")
            self.reset_ui()
            return

        if not os.path.exists(self.current_audio_path):
            self.log(f"错误: 音频文件不存在: {self.current_audio_path}")
            self.reset_ui()
            return
        
        self.log("正在检查音频文件...")
        
        # 分割音频文件（带进度显示）
        self.segments = split_audio_with_progress(self.current_audio_path, log_callback=self.log)
        if self.segments is None:
            self.log("错误: 无法分割音频文件，请检查文件格式")
            self.reset_ui()
            return
        
        self.total_segments = len(self.segments)
        self.current_segment = 0
        self.all_results = []
        
        if self.total_segments > 1:
            self.log(f"\n音频分割详情:")
            for i, (path, start, end) in enumerate(self.segments):
                duration = end - start
                self.log(f"  片段{i+1}: {start:.0f}s - {end:.0f}s ({duration:.0f}秒)")
        else:
            self.log("音频时长较短，无需分割")
        
        self.log(f"\n总共 {self.total_segments} 个片段，开始转录...")
        self.process_next_segment()
        
    def monitor_transcription(self):
        check_interval = 1
        progress_interval = 10
        last_progress_time = time.time()
        
        while self.transcribing:
            try:
                current_time = time.time()
                
                while not self.log_queue.empty():
                    msg = self.log_queue.get_nowait()
                    self.log(msg)
                    self.last_log_time = current_time
                
                if not self.result_queue.empty():
                    result_data = self.result_queue.get_nowait()
                    if len(result_data) >= 2:
                        status = result_data[0]
                        data = result_data[1]
                        segment_idx = result_data[2] if len(result_data) > 2 else None

                        if status == "success":
                            self.process_result(data)
                        elif status == "error":
                            self.log(f"\n处理失败: {data}")
                            self.reset_ui()
                            return
                
                if self.transcribe_process and not self.transcribe_process.is_alive():
                    # 进程已结束，检查是否还有未处理的片段
                    if self.transcribing and self.current_segment < self.total_segments:
                        # 还有片段要处理，但新进程还没启动，继续等待
                        time.sleep(0.5)
                        continue
                    elif self.transcribing and self.current_segment >= self.total_segments:
                        # 所有片段都处理完了
                        self.log("\n所有片段处理完成")
                        return
                    elif self.transcribing:
                        self.log("\n转录进程异常结束")
                        self.reset_ui()
                    return
                
                # 如果 transcrive_process 为 None，说明正在等待启动新进程
                if self.transcribe_process is None and self.transcribing and self.current_segment < self.total_segments:
                    # 继续等待 process_next_segment 启动新进程
                    time.sleep(0.5)
                    continue
                
                elapsed_time = current_time - self.start_time
                
                if current_time - last_progress_time >= progress_interval:
                    last_progress_time = current_time
                    elapsed_minutes = int(elapsed_time // 60)
                    elapsed_seconds = int(elapsed_time % 60)
                    
                    if self.audio_duration:
                        estimated_time = self.audio_duration / 10
                        progress_percent = min((elapsed_time / estimated_time) * 100, 100)
                        est_minutes = int(estimated_time // 60)
                        est_seconds = int(estimated_time % 60)
                        self.log(f"已处理: {elapsed_minutes}分{elapsed_seconds}秒 / 预估: {est_minutes}分{est_seconds}秒 ({progress_percent:.1f}%)")
                    else:
                        self.log(f"已处理: {elapsed_minutes}分{elapsed_seconds}秒")
                
                if self.audio_duration:
                    estimated_time = self.audio_duration / 10
                    timeout_threshold = estimated_time * 3
                    
                    if elapsed_time > timeout_threshold and (current_time - self.last_log_time) > 60:
                        self.log(f"\n警告: 处理时间超过预估时间3倍，可能存在问题")
                        self.log(f"已用时间: {elapsed_time:.0f}秒, 预估时间: {estimated_time:.0f}秒")
                        self.log("建议点击中断按钮重新尝试")
                
                time.sleep(check_interval)
                
            except Exception as e:
                self.log(f"监控过程出错: {e}")
                break
                
    def process_result(self, result):
        if not self.transcribing:
            self.log("\n操作已中断")
            return
            
        processing_time = time.time() - self.start_time
        segments = result.get('segments', [])
        
        if segments:
            total_duration = segments[-1]['end']
            self.log(f"\n处理完成！")
            self.log(f"音频时长: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
            self.log(f"处理时间: {processing_time:.2f} 秒")
            self.log(f"处理速度: {total_duration/processing_time:.2f}x 实时速度")
            self.log(f"分段数量: {len(segments)}")
            
            self.log(f"\n{'='*50}")
            self.log("转录结果:")
            self.log(f"{'='*50}\n")
            
            for i, segment in enumerate(segments, 1):
                if not self.transcribing:
                    break
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                self.log(f"[{i}] {start:.2f}s - {end:.2f}s: {text}")
            
            if self.transcribing:
                self.all_results.append(result)
                self.current_segment += 1
                
                if self.current_segment < self.total_segments:
                    self.log(f"\n片段 {self.current_segment}/{self.total_segments} 完成，3秒后继续处理下一个片段...")
                    # 先重置 transcrive_process，让监控线程知道我们要继续
                    self.transcribe_process = None
                    # 延迟3秒后启动下一个片段
                    self.root.after(3000, self.process_next_segment)
                else:
                    self.log(f"\n所有 {self.total_segments} 个片段处理完成，正在合并结果...")
                    self.merge_and_save_results()
        else:
            self.log("未检测到语音内容")
            self.reset_ui()
        
    def process_next_segment(self):
        if self.current_segment >= self.total_segments:
            return
            
        segment_path, start_time, end_time = self.segments[self.current_segment]
        self.log(f"\n{'='*50}")
        self.log(f"处理片段 {self.current_segment + 1}/{self.total_segments}")
        self.log(f"时间范围: {start_time:.0f}s - {end_time:.0f}s")
        self.log(f"{'='*50}")
        
        self.log_queue = Queue()
        self.result_queue = Queue()
        self.last_log_time = time.time()
        
        try:
            self.transcribe_process = Process(
                target=transcribe_worker,
                args=(segment_path, self.model_path, self.result_queue, self.log_queue, 
                      self.current_segment, start_time)
            )
            self.transcribe_process.start()
            
            thread = threading.Thread(target=self.monitor_transcription)
            thread.daemon = True
            thread.start()
        except Exception as e:
            self.log(f"启动转录进程失败: {e}")
            self.reset_ui()
        
    def merge_and_save_results(self):
        if not self.all_results:
            self.log("没有可合并的结果")
            self.reset_ui()
            return
            
        self.log(f"\n{'='*50}")
        self.log("合并所有片段结果")
        self.log(f"{'='*50}\n")
        
        all_segments = []
        full_text_parts = []
        
        for i, result in enumerate(self.all_results):
            self.log(f"合并片段 {i+1}/{len(self.all_results)}...")
            if 'segments' in result:
                all_segments.extend(result['segments'])
            full_text_parts.append(result.get('text', ''))
        
        merged_text = '\n'.join(full_text_parts)
        
        self.log(f"\n{'='*50}")
        self.log("转录结果:")
        self.log(f"{'='*50}\n")
        
        for i, segment in enumerate(all_segments, 1):
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            self.log(f"[{i}] {start:.2f}s - {end:.2f}s: {text}")
        
        # 保存转录结果（非一键模式下）
        if not (hasattr(self, 'one_click_mode') and self.one_click_mode):
            self.save_result(merged_text, all_segments)
        
        processing_time = time.time() - self.start_time
        self.log(f"\n{'='*50}")
        self.log("转录完成！")
        self.log(f"总处理时间: {processing_time:.2f} 秒")
        self.log(f"{'='*50}")
        
        # 保存转录结果供一键模式使用
        self.transcription_result = merged_text
        
        self.cleanup_temp_files()
        self.reset_ui()
        
        # 如果是一键模式，继续下一步
        if hasattr(self, 'one_click_mode') and self.one_click_mode:
            self.root.after(1000, self._on_transcription_complete_for_one_click)
        
    def cleanup_temp_files(self):
        if self.segments and len(self.segments) > 1:
            for segment_path, _, _ in self.segments:
                try:
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
                except Exception as e:
                    pass
            
            temp_dir = os.path.dirname(self.segments[0][0]) if self.segments else None
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.log("临时文件已清理")
                except Exception as e:
                    pass
        
        self.segments = []
        self.all_results = []
        
    def save_result(self, text, segments=None):
        if not self.current_audio_path:
            self.log("错误: 没有当前音频文件路径")
            return
            
        try:
            audio_path = Path(self.current_audio_path)
            output_path = audio_path.with_suffix('.txt')
            
            self.log(f"\n正在保存结果到: {output_path}")
            self.log(f"音频路径: {audio_path}")
            self.log(f"segments数量: {len(segments) if segments else 0}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if segments:
                    # 带时间戳格式
                    f.write(f"会议纪要文本\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"原始文件: {audio_path.name}\n")
                    f.write(f"总片段数: {len(segments)}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    for i, segment in enumerate(segments, 1):
                        start = segment['start']
                        end = segment['end']
                        segment_text = segment['text'].strip()
                        
                        # 格式化时间为 MM:SS 格式
                        start_min = int(start // 60)
                        start_sec = int(start % 60)
                        end_min = int(end // 60)
                        end_sec = int(end % 60)
                        
                        # 时间戳和文本在一行显示
                        f.write(f"[{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}] {segment_text}\n")
                else:
                    # 纯文本格式（如果没有segments）
                    f.write(text)
                
            # 验证文件是否成功创建
            if output_path.exists():
                file_size = output_path.stat().st_size
                self.log(f"✓ 结果已成功保存到: {output_path}")
                self.log(f"  文件大小: {file_size} 字节")
            else:
                self.log(f"✗ 文件保存失败: {output_path} 不存在")
                
        except PermissionError as e:
            self.log(f"保存文件失败 (权限错误): {e}")
            self.log(f"请检查目录权限: {output_path.parent}")
        except Exception as e:
            self.log(f"保存文件失败: {e}")
            import traceback
            self.log(f"详细错误: {traceback.format_exc()}")
            
    def stop_transcription(self):
        if self.transcribing and self.transcribe_process:
            self.log("\n正在中断操作...")
            self.transcribing = False
            
            if self.transcribe_process.is_alive():
                self.transcribe_process.terminate()
                self.log("已发送终止信号")
                
                try:
                    self.transcribe_process.join(timeout=5)
                    if self.transcribe_process.is_alive():
                        self.log("进程未响应，强制终止")
                        self.transcribe_process.kill()
                        self.transcribe_process.join()
                except Exception as e:
                    self.log(f"终止进程时出错: {e}")
                    
            self.log("操作已中断")
            self.reset_ui()
            
    def reset_ui(self):
        self.transcribing = False
        self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.open_button.config(state=tk.NORMAL))
        
        if self.transcribe_process and self.transcribe_process.is_alive():
            try:
                self.transcribe_process.terminate()
                self.transcribe_process.join(timeout=3)
                if self.transcribe_process.is_alive():
                    self.transcribe_process.kill()
                    self.transcribe_process.join()
            except Exception as e:
                pass
        
        self.transcribe_process = None
        
    def on_closing(self):
        if self.transcribing:
            self.log("正在处理中，停止处理并退出...")
            self.stop_transcription()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MeetingMinutesApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
