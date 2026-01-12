"""
批量音频转录脚本 - 输出 Markdown 格式
使用 Whisper medium 模型，适合 8GB 显存显卡
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import whisper


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 HH:MM:SS 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def transcribe_to_markdown(audio_path: str, model, output_dir: str = None) -> str:
    """转录单个音频文件并生成 Markdown"""
    audio_path = Path(audio_path)
    print(f"\n正在转录: {audio_path.name}")
    
    # 转录，启用词级时间戳以获得更精确的分段
    result = model.transcribe(
        str(audio_path),
        language="zh",  # 中文，如需其他语言请修改
        task="transcribe",
        verbose=False,
        word_timestamps=True,
    )
    
    # 生成 Markdown 内容
    md_lines = [
        f"# {audio_path.stem}",
        "",
        f"> 转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> 源文件: `{audio_path.name}`  ",
        f"> 检测语言: {result.get('language', 'unknown')}",
        "",
        "---",
        "",
        "## 转录内容",
        "",
    ]
    
    # 按段落输出，带时间戳
    for segment in result["segments"]:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        md_lines.append(f"**[{start} - {end}]** {text}")
        md_lines.append("")
    
    # 添加纯文本版本（方便复制）
    md_lines.extend([
        "---",
        "",
        "## 纯文本",
        "",
        result["text"].strip(),
        "",
    ])
    
    md_content = "\n".join(md_lines)
    
    # 保存文件
    if output_dir:
        output_path = Path(output_dir) / f"{audio_path.stem}.md"
    else:
        output_path = audio_path.with_suffix(".md")
    
    output_path.write_text(md_content, encoding="utf-8")
    print(f"已保存: {output_path}")
    
    return str(output_path)


def main():
    # 支持的音频格式
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac"}
    
    # 获取输入路径
    if len(sys.argv) < 2:
        print("用法: python transcribe_to_md.py <音频文件或目录...> [-o 输出目录]")
        print("示例:")
        print("  python transcribe_to_md.py recording.mp3")
        print("  python transcribe_to_md.py ./recordings -o ./output")
        print("  python transcribe_to_md.py ./dir1 ./dir2 ./dir3 -o ./output")
        sys.exit(1)
    
    # 解析参数：支持多个输入目录 + 可选的 -o 输出目录
    args = sys.argv[1:]
    output_dir = None
    input_paths = []
    
    i = 0
    while i < len(args):
        if args[i] == "-o" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            input_paths.append(Path(args[i]))
            i += 1
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有音频文件
    audio_files = []
    for input_path in input_paths:
        if input_path.is_file():
            audio_files.append(input_path)
        elif input_path.is_dir():
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(input_path.glob(f"*{ext}"))
                audio_files.extend(input_path.glob(f"*{ext.upper()}"))
        else:
            print(f"警告: 路径不存在，跳过 - {input_path}")
    
    if not audio_files:
        print("未找到音频文件")
        sys.exit(1)
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 加载模型 (medium 模型，高精度，约 5GB 显存)
    print("\n正在加载 Whisper medium 模型...")
    print("如果下载失败，可以手动下载模型到 ~/.cache/whisper/ 目录")
    print("模型下载地址: https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt")
    try:
        model = whisper.load_model("medium", device="cuda")
    except Exception as e:
        print(f"\n模型加载失败: {e}")
        print("\n请尝试:")
        print("1. 检查网络连接")
        print("2. 使用代理: set HTTPS_PROXY=http://127.0.0.1:7890")
        print("3. 手动下载模型文件到: C:\\Users\\<用户名>\\.cache\\whisper\\medium.pt")
        sys.exit(1)
    print("模型加载完成")
    
    # 批量转录
    for audio_file in sorted(audio_files):
        try:
            transcribe_to_markdown(str(audio_file), model, output_dir)
        except Exception as e:
            print(f"转录失败 {audio_file.name}: {e}")
    
    print("\n全部完成!")


if __name__ == "__main__":
    main()
