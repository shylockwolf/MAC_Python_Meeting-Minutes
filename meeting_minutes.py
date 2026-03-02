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
        
        # 第二行按钮 - 文本处理
        text_frame = tk.Frame(self.root)
        text_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.open_text_button = tk.Button(text_frame, text="打开文本文件", command=self.open_text_file, width=15)
        self.open_text_button.pack(side=tk.LEFT, padx=5)
        
        self.format_button = tk.Button(text_frame, text="开始成文", command=self.start_formatting, width=15, state=tk.DISABLED)
        self.format_button.pack(side=tk.LEFT, padx=5)
        
        self.minutes_button = tk.Button(text_frame, text="生成会议纪要", command=self.generate_minutes, width=15, state=tk.DISABLED)
        self.minutes_button.pack(side=tk.LEFT, padx=5)
        
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
            messagebox.showwarning("警告", "正在处理中，请先中断当前操作")
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
            
    def open_text_file(self):
        if self.transcribing:
            messagebox.showwarning("警告", "正在处理中，请先中断当前操作")
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
            except Exception as e:
                self.log(f"读取文件失败: {e}")
                messagebox.showerror("错误", f"读取文件失败: {e}")
                
    def start_formatting(self):
        if not hasattr(self, 'current_text_content'):
            messagebox.showwarning("警告", "请先选择文本文件")
            return
            
        self.log(f"\n{'='*50}")
        self.log("开始成文处理...")
        self.log(f"{'='*50}\n")
        
        # TODO: 实现成文逻辑
        self.log("成文功能待实现")
        self.minutes_button.config(state=tk.NORMAL)
        
    def generate_minutes(self):
        if not hasattr(self, 'current_text_content'):
            messagebox.showwarning("警告", "请先选择文本文件并完成成文")
            return
            
        self.log(f"\n{'='*50}")
        self.log("生成会议纪要...")
        self.log(f"{'='*50}\n")
        
        # TODO: 实现会议纪要生成逻辑
        self.log("会议纪要生成功能待实现")
            
    def start_transcription(self):
        if not self.current_audio_path:
            messagebox.showwarning("警告", "请先选择音频文件")
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
            messagebox.showerror("错误", f"模型路径不存在: {self.model_path}")
            self.reset_ui()
            return
        
        if not os.path.exists(self.current_audio_path):
            self.log(f"错误: 音频文件不存在: {self.current_audio_path}")
            messagebox.showerror("错误", f"音频文件不存在: {self.current_audio_path}")
            self.reset_ui()
            return
        
        self.log("正在检查音频文件...")
        
        # 分割音频文件（带进度显示）
        self.segments = split_audio_with_progress(self.current_audio_path, log_callback=self.log)
        if self.segments is None:
            self.log("错误: 无法分割音频文件")
            messagebox.showerror("错误", "无法分割音频文件，请检查文件格式")
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
                            messagebox.showerror("错误", f"处理失败: {data}")
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
        
        # 保存两个版本：纯文本版和时间戳版
        self.save_result(merged_text, all_segments)
        
        processing_time = time.time() - self.start_time
        self.log(f"\n{'='*50}")
        self.log("转录完成！")
        self.log(f"总处理时间: {processing_time:.2f} 秒")
        self.log(f"{'='*50}")
        
        self.cleanup_temp_files()
        self.reset_ui()
        
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
            if messagebox.askokcancel("退出", "正在处理中，确定要退出吗？"):
                self.stop_transcription()
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    root = tk.Tk()
    app = MeetingMinutesApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
