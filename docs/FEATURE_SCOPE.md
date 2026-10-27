# 当前维护范围

项目为本地视频提供 NPU 实时识别与字幕翻译。

| 输入 | 当前实现 | 播放器 |
| --- | --- | --- |
| 英语音频 | Moonshine Small → Marian 英中 | mpv / MPC-HC |
| 日语音频 | Granite → Marian 日英 → 英中 | mpv / MPC-HC |
| 中文音频 | SenseVoice 中文识别 | mpv 插件 |
| 已有中文文字或位图字幕 | 原样显示 | mpv 插件 |
| 英日文字字幕 | Marian 翻译 | mpv 插件 |
| 韩、西、法、德、葡、意、俄文字字幕 | 固定源语言→英语→中文 | mpv 插件 |

英文可显式选择已有 Granite 或 Moonshine Medium；不继续扩展候选模型。额外七种语言不支持无字幕 ASR，位图字幕不做 OCR。

## 执行与资源约束

- 所有识别和翻译神经网络必须在 NPU 执行；不能执行就报错，不回退到 CPU/GPU。
- 音频解码、特征处理、分词、SenseVoice 反分词及调度由主机完成。
- 有界预读、单任务 NPU 调度、字幕绝对期限和 seek 隔离保持不变。
- 暂停、继续、跳转、换音轨和退出是维护重点；不按影片或台词添加特例。
- 播放时不下载模型。只加载当前语种所需模型。

当前命令见 [README](../README.md) 和 [模型准备](MODEL_PREPARATION.md)；代码边界见 [架构说明](REALTIME_SUBTITLES_ARCHITECTURE.md)。

## 验证边界

使用无模型回归、必要的实际 NPU 推理与播放器生命周期检查。检查通过不代表识别内容或译文绝对正确，短片段实播不代表整片字幕质量验收。
