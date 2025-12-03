import streamlit as st
import yt_dlp
import whisper
import torch
import os
import tempfile
from datetime import datetime
from moviepy.editor import VideoFileClip

# ====================== 云部署适配配置（固定，无需修改） ======================
MAX_VIDEO_DURATION = 1800  # 云部署限制：最大支持30分钟视频（避免超内存）
DEFAULT_TRANS_MODEL = "base"  # 平衡速度和准确率，云服务器CPU也能跑
ALLOW_AI_SUMMARY = True  # 允许AI总结（小白可关闭）

# ====================== 核心工具函数（云部署兼容） ======================
def get_video(video_url):
    """在线视频下载（云部署兼容：临时文件自动清理）"""
    temp_dir = tempfile.TemporaryDirectory()
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": os.path.join(temp_dir.name, "%(title)s.%(ext)s"),
        "quiet": True,
        "max_duration": MAX_VIDEO_DURATION,
        "no_warnings": True,
        "noplaylist": True  # 禁止下载播放列表，避免超时
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_path = ydl.prepare_filename(info)
            video_info = {
                "title": info.get("title", "未知视频"),
                "channel": info.get("uploader", "未知来源"),
                "duration": info.get("duration", 0),
                "upload_date": info.get("upload_date", ""),
                "url": video_url
            }
        return video_path, video_info, temp_dir
    except Exception as e:
        st.error(f"❌ 视频下载失败：{str(e)}")
        st.info("💡 可能原因：1. 链接不是公开视频；2. 视频超过30分钟；3. 不支持该平台")
        return None, None, None

def extract_audio(video_path, temp_dir):
    """提取音频（云部署兼容：固定格式，减少资源占用）"""
    audio_path = os.path.join(temp_dir.name, "temp_audio.mp3")
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path, bitrate="128k", verbose=False)
        return audio_path
    except Exception as e:
        st.error(f"❌ 音频提取失败：{str(e)}")
        return None

def audio_to_text(audio_path):
    """音频转文字（云部署优化：自动适配CPU，模型缓存）"""
    @st.cache_resource(show_spinner="正在加载转写模型...")  # 缓存模型，避免重复下载
    def load_whisper_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return whisper.load_model(DEFAULT_TRANS_MODEL, device=device)
    
    model = load_whisper_model()
    with st.spinner("正在转写文字...（视频越长越久）"):
        result = model.transcribe(audio_path, language="zh", fp16=torch.cuda.is_available())
    return {
        "text": result["text"],
        "segments": result.get("segments", [])
    }

def generate_summary(transcript_text, video_info, openai_key=""):
    """总结生成（小白友好：默认提取式，AI可选）"""
    # 提取式总结（无API依赖，默认启用）
    sentences = [s.strip() for s in transcript_text.split("。") if s.strip()]
    if len(sentences) <= 10:
        summary = "。".join(sentences) + "。"
        key_points = sentences[:5]
    else:
        summary = "。".join(sentences[:5] + sentences[-3:]) + "。"
        key_points = sentences[:5]
    
    base_summary = {
        "summary": summary,
        "key_points": key_points,
        "type": "快速提取总结（无API依赖）"
    }
    
    # AI总结（可选，小白可跳过）
    if ALLOW_AI_SUMMARY and openai_key:
        try:
            import openai
            openai.api_key = openai_key
            prompt = f"""请用简单易懂的语言总结以下视频内容，结构清晰：
1. 核心内容（1段话，不超过3行）
2. 3个关键要点（分点列，每点不超过20字）

视频标题：{video_info['title']}
视频原文：{transcript_text[:3000]}"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                timeout=20
            )
            ai_content = response.choices[0].message["content"].strip()
            
            # 提取关键点（小白友好：简化格式）
            ai_key_points = []
            for line in ai_content.split("\n"):
                line = line.strip()
                if line.startswith(("1.", "2.", "3.", "•", "-")):
                    ai_key_points.append(line.lstrip("123.•- ").strip())
            
            return {
                "summary": ai_content,
                "key_points": ai_key_points[:3],
                "type": "AI增强总结（GPT-3.5）"
            }
        except Exception as e:
            st.warning(f"⚠️ AI总结失败，自动使用快速提取总结：{str(e)}")
            return base_summary
    else:
        return base_summary

def format_markdown(summary, video_info, transcript):
    """Markdown格式化（小白友好：简洁明了，无复杂结构）"""
    # 处理时长和日期（小白能看懂）
    duration = video_info["duration"]
    duration_str = f"{duration//60}分{duration%60}秒" if duration else "未知"
    try:
        upload_date = datetime.strptime(video_info["upload_date"], "%Y%m%d").strftime("%Y年%m月%d日")
    except:
        upload_date = "未知"
    
    # 简化Markdown模板（无冗余信息）
    md = f"""# 视频总结：{video_info['title']}

## 📋 视频信息
- 标题：{video_info['title']}
- 来源：{video_info['channel']}
- 时长：{duration_str}
- 上传日期：{upload_date}
- 总结类型：{summary['type']}

## 📝 核心总结
{summary['summary']}

## 🔑 关键要点
"""
    for i, point in enumerate(summary['key_points'], 1):
        md += f"{i}. {point}\n"
    
    # 关键时间线（小白能快速定位重点）
    if transcript["segments"]:
        md += "\n## ⏱️ 快速时间线（前3个重点）\n"
        for seg in transcript["segments"][:3]:
            time = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
            md += f"- **{time}**：{seg['text'][:50]}...\n"
    
    md += f"\n---\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n网页工具：视频总结助手"
    return md

# ====================== 小白友好型网页界面 ======================
def main():
    # 页面配置（美观+简洁）
    st.set_page_config(
        page_title="小白专用视频总结工具",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="collapsed"  # 隐藏侧边栏，减少干扰
    )
    
    # 标题和说明（小白能看懂）
    st.title("📝 视频总结助手")
    st.markdown("### 👉 小白也能秒用：输入视频链接，自动生成总结")
    st.markdown("✅ 支持：B站、YouTube、抖音、小红书（公开链接）")
    st.markdown("⚠️ 限制：视频不超过30分钟，仅用于合规内容")
    st.divider()
    
    # 核心输入（只有1个输入框，小白不会迷路）
    video_url = st.text_input(
        "🔗 粘贴视频链接",
        placeholder="例：https://www.bilibili.com/video/BV1xx411c7mC",
        help="复制视频的公开分享链接，粘贴到这里"
    )
    
    # AI总结开关（默认关闭，小白可选择性开启）
    if ALLOW_AI_SUMMARY:
        with st.expander("✨ 可选：使用AI增强总结（更精准）", expanded=False):
            st.markdown("需要OpenAI API Key（免费额度足够用，获取教程：[点击查看](https://platform.openai.com/api-keys)）")
            openai_key = st.text_input("输入OpenAI API Key", type="password", placeholder="没有可以不填，用默认总结")
    else:
        openai_key = ""
    
    # 生成按钮（大按钮，醒目）
    start_btn = st.button("🚀 开始生成总结", type="primary", use_container_width=True)
    progress_bar = st.progress(0, text="未开始处理")
    
    # 处理流程（小白能看到进度）
    if start_btn and video_url:
        # 验证链接（简单校验，避免无效输入）
        if not (video_url.startswith("http://") or video_url.startswith("https://")):
            st.error("❌ 请输入有效的视频链接（以http/https开头）")
            return
        
        try:
            # 步骤1：下载视频
            progress_bar.progress(0.2, text="正在下载视频...")
            video_path, video_info, temp_dir = get_video(video_url)
            if not video_path:
                return
            st.success(f"✅ 视频下载成功：{video_info['title']}")
        
            # 步骤2：提取音频+转文字
            progress_bar.progress(0.5, text="正在提取音频并转文字...")
            audio_path = extract_audio(video_path, temp_dir)
            if not audio_path:
                return
            transcript = audio_to_text(audio_path)
            st.success("✅ 文字转写完成！")
        
            # 步骤3：生成总结
            progress_bar.progress(0.8, text="正在生成总结...")
            summary = generate_summary(transcript["text"], video_info, openai_key)
            st.success("✅ 总结生成完成！")
        
            # 步骤4：显示结果+下载
            progress_bar.progress(1.0, text="处理完成！")
            st.divider()
            
            # 结果展示（大字体，清晰）
            st.subheader("📊 总结结果")
            md_content = format_markdown(summary, video_info, transcript)
            st.markdown(md_content)
            
            # 下载按钮（小白一眼能看到）
            st.download_button(
                label="💾 下载总结（Markdown文件）",
                data=md_content,
                file_name=f"视频总结_{video_info['title']}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        except Exception as e:
            st.error(f"❌ 处理失败：{str(e)}")
        finally:
            # 清理临时文件（云部署避免占用空间）
            try:
                temp_dir.cleanup()
            except:
                pass
    
    # 空输入提示（小白不会困惑）
    elif start_btn:
        st.error("❌ 请先粘贴视频链接！")

if __name__ == "__main__":
    main()
