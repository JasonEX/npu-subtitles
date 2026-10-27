# 实时字幕架构

当前实现以播放会话为边界。模型不承担播放器控制，播放器适配器不决定字幕寿命，翻译不修改来源文本。目录采用普通 Python 包，不引入插件注册框架、服务容器或新的消息总线。

## 数据流

```text
watch.py / mpv_plugin.py
          ↓
npu_subtitles.app
  播放时间线 → 有界音频读取 → 单个 NPU 工作进程
                                  ↓
                     识别观察 / 封存词出现
                                  ↓
                    CaptionPipeline：提交与预算
                                  ↓
                      翻译 → PresentationController
                                  ↓
                         mpv / MPC-HC 显示
```

已有文字字幕从 mpv 路由进入 `translation.subtitles`，复用同一 Marian 后端和翻译阶段执行器，不经过 ASR。

## 模块边界

| 模块 | 职责 |
| --- | --- |
| `app.py` | 校验会话参数、协调时间线、解码与推理、记录阶段日志、退出清理 |
| `inference.py` | `InferenceSettings` 明确工作进程配置；一个进程串行执行 ASR 与翻译 |
| `timeline.py` | 媒体时钟、epoch、预读边界与落后重置 |
| `playback.py` | 有界音频读取、字幕输出和播放器创建 |
| `asr/` | 模型适配和跨窗口识别账本；不访问播放器或翻译器 |
| `captions/commit.py` | Moonshine 观察的稳定提交 |
| `captions/sentences.py` | Granite 已封存词的归属与分句 |
| `captions/timing.py` | 来源时间、提交预算与绝对期限 |
| `captions/presentation.py` | 显示资格、阅读机会、过期退休 |
| `captions/pipeline.py` | 连接提交、翻译预算与显示策略 |
| `translation/chain.py` | 分段与多阶段翻译，共用期限、取消信号，只提交完整译文 |
| `translation/marian.py` | 已准备静态图的 NPU 推理 |
| `players/` | mpv IPC 与 MPC-HC 通信协议 |
| `formats.py / ass.py` | 字幕数据与 SRT/ASS 序列化，无播放器或模型依赖 |
| `media.py / paths.py / devices.py` | 媒体工具、统一资源根目录、实际执行设备检查 |
| `tools/models/` | 固定来源下载、离线导出和准备验证 |

核心包禁止依赖 `tools`；模型准备可以依赖运行时模型格式。根目录仅保留用户入口 `watch.py` 与原版 mpv 启动的 `mpv_plugin.py`。旧 GUI、文件转写、Qwen 预读和比较脚本已删除，不保留转发文件。

## 时间与资源约束

- 一个推理工作进程只允许一个活动任务；不会同时加载七种字幕语言模型。
- 音频读取不超过播放器位置加预读上限，暂停后在该边界停止。
- seek、换音轨或换文件生成新的状态边界，旧音频和旧译文不能进入新会话。
- Granite 的封存词身份与已提交句子保持不变；Moonshine 继续使用自己的观察协议。
- 翻译期限来自来源时间，排队、分段和英语中转不能刷新期限；失败或取消不返回部分译文。
- 显示队列、观察历史、解码范围和进程退出均有界。必需的输入、设备和所有权检查保留。
- 模型只在处理需要时加载。导入路由、解析帮助不加载 NumPy、OpenVINO、GUI 或导出器。
- 主机数学库限制单线程。实际 NPU 设备由已编译图报告核对，不以请求参数冒充证据。

更细的词归属与时间合同见 [流式字幕合同](STREAMING_CAPTION_CONTRACTS.md)。

## 依赖与维护

`pyproject.toml` 声明依赖，`uv.lock` 锁定版本。默认仅包含播放依赖；`prepare`、`test` 按用途启用。Granite 的 PyTorch 转换环境单独维护。

模型、编译缓存与运行日志保存在本地，并排除在版本管理之外。

验证分为无模型回归、真实 NPU 推理与播放器生命周期。单元检查通过不等同于字幕质量通过；短片段实播不等同于整片验收。
