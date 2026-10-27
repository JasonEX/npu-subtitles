# NPU Subtitles

**用 Intel NPU 为本地视频生成实时双语字幕。**

支持 Windows 上的 mpv 和 MPC-HC。识别与翻译模型在本机 NPU 上运行；无需云端推理服务。可从音频生成字幕，也可以通过 mpv 插件翻译已有文字字幕。

## 支持什么

| 输入 | 输出与处理方式 | 使用方式 |
| --- | --- | --- |
| 英语音频 | 英语识别 → 中文字幕，显示英中双语 | 启动器、mpv 插件 |
| 日语音频 | 日语识别 → 英语中转 → 中文字幕，显示日中双语 | 启动器、mpv 插件 |
| 中文音频 | SenseVoice 中文识别 | mpv 插件 |
| 已有中文字幕 | 原样显示，支持繁体、双语和位图字幕 | mpv 插件 |
| 英语、日语文字字幕 | 翻译成中文，保留原文 | mpv 插件 |
| 韩、西、法、德、葡、意、俄文字字幕 | 翻译成英语再译成中文，显示英中双语 | mpv 插件 |

英语默认使用 Moonshine Streaming Small，可选 Medium 或 Granite；日语使用 Granite TurboCTC；翻译使用 Marian。模型按语言按需准备与加载。

额外七种语言目前仅支持文字字幕翻译，不支持音频识别。非中文位图字幕不做 OCR。识别与翻译可能错词、漏句；中转翻译可能累积误差。

## 环境要求

- Windows 10/11 x64，带 NPU 的 Intel Core Ultra，以及可用的 Intel NPU 驱动。
- Python 3.12 和 [uv](https://docs.astral.sh/uv/)。
- FFmpeg：`ffmpeg`、`ffprobe` 可从 PATH 找到。
- mpv 或 MPC-HC。推荐将播放器加入 PATH；也可以通过命令行指定位置。

识别、翻译神经网络必须在 NPU 上执行，不支持 CPU/GPU 推理回退。音频解码、特征处理、分词、反分词和播放调度仍由 CPU 完成。

## 快速开始：英语视频

在 PowerShell 中执行：

```powershell
git clone https://github.com/JasonEX/npu-subtitles.git
cd npu-subtitles
uv sync --locked

# 检查是否能看到 NPU
uv run --locked -m tools.check_devices

# 首次下载并准备英语识别、英中翻译模型
uv run --locked --group prepare -m tools.models.moonshine
uv run --locked --group prepare -m tools.models.mtran
```

准备完成后，把视频拖到 **`watch.cmd`**，选择播放器和英语。也可以双击它，再选择视频文件。

命令行播放：

```powershell
uv run --locked watch.py --player mpv "./video.mkv"
uv run --locked watch.py --player mpc-hc "./video.mkv"
```

首次编译模型可能较慢，后续使用本地编译缓存。播放期间不会自动下载模型；缺少模型或 NPU 不可用时会报错。

## 日语、中文和其他字幕语言

日语需要 Granite、日英和英中模型。Granite 的离线导出使用独立环境，避免将 PyTorch 和 Transformers 带入播放环境。按 [模型准备](docs/MODEL_PREPARATION.md) 完成后：

```powershell
uv run --locked watch.py --player mpv --asr granite --language ja "./video.mkv"
```

也可以在 `watch.cmd` 中选择日语。

为 mpv 插件准备中文识别或其他语言的文字字幕翻译：

```powershell
uv run --locked --group prepare -m tools.models.sensevoice
# 按需选择语言；省略 --languages 会准备全部七种语言
uv run --locked --group prepare -m tools.models.subtitles --languages ko fr
```

文字字幕的非中文翻译路径都需要英中模型。各语言完整命令与准备检查见 [模型准备](docs/MODEL_PREPARATION.md)。

## mpv 自动字幕插件

安装后可直接用 mpv 打开视频，不必每次通过启动器：

```powershell
uv run --locked -m tools.install_mpv_plugin
# 播放器未加入 PATH 时，指定其实际位置
uv run --locked -m tools.install_mpv_plugin --mpv "./player/mpv.exe"
```

安装器优先使用 `MPV_HOME`、播放器旁已有的 `portable_config`，否则使用 `%APPDATA%/mpv`。它写入本项目的脚本、选项和安装记录，不修改 `mpv.conf`。

- 已有中文字幕优先显示。日语音轨优先使用日语文字字幕；其他文字字幕按支持的语言翻译。
- 没有可用文字字幕时，识别受支持的中、英、日音频。
- **Ctrl+N** 开关自动字幕；手动切换字幕后，插件让出控制。
- 换片、切换音轨和关闭播放器时，清理当前字幕任务。

使用 `--no-config`、`--load-scripts=no` 或其他配置目录会影响自动加载。MPC-HC 通过启动器使用，不提供内部插件。

卸载：

```powershell
uv run --locked -m tools.install_mpv_plugin --uninstall
```

若安装时指定了 `--mpv` 或 `--config-dir`，卸载时使用同样的参数。安装记录包含当前项目位置；移动项目前先卸载插件，移动后再安装。

## 常用选项

| 参数 | 用途 |
| --- | --- |
| `--audio-track N` | 指定音轨，从 0 开始编号 |
| `--lookahead-seconds 0` | 关闭提前读取；默认 8 秒，最大 12 秒 |
| `--start 60 --duration 30` | 从第 60 秒开始播放 30 秒 |
| `--asr granite --language ja` | 日语识别 |
| `--moonshine medium` | 使用已准备的 Moonshine Medium |
| `--mpv PATH` / `--mpc-hc PATH` | 指定播放器位置 |

```powershell
uv run --locked watch.py --help
```

项目采用有界预读，不预先转写整部影片。暂停和跳转会影响任务调度；翻译超过字幕期限时可能保留原文或丢弃过期结果。

## 隐私与本地数据

音视频、识别文本和译文在本机处理。模型准备阶段会连接模型托管服务下载文件；本项目的字幕推理路径不调用云端识别或翻译 API。

`models/`、`.cache/`、`outputs/`、虚拟环境和本地素材均不提交到仓库。运行日志可能包含媒体路径、识别文本和译文，存放在 `outputs/`；mpv 的本地安装配置也包含项目路径。报告问题时请先删去这些内容中的个人信息，不要直接上传完整日志或配置。

仓库不包含视频、模型权重、私人测试记录或历史探索报告。下载的模型有各自的许可证，见 `third_party/` 和模型目录中的来源记录。

## 常见问题

**没有找到播放器或 FFmpeg**：将对应程序加入 PATH；播放器也可用 `--mpv` 或 `--mpc-hc` 指定。拖放启动器使用 PATH 和播放器的标准安装位置。

**没有找到 NPU，或模型编译失败**：先运行设备检查，确认驱动和模型已准备。项目不会自动换用 CPU/GPU。

**插件没有字幕**：确认安装到了 mpv 实际使用的配置目录、没有关闭脚本加载，并已准备对应语言模型。启动错误见 `outputs/launch/`，插件日志见 `outputs/plugin/`。

**字幕有延迟或内容不准**：实时识别需要积累上下文，翻译还有独立耗时。可调整提前读取长度，但更大的预读不保证更好的准确率。短音频、混合语言和缺少语言标签的字幕尤其容易识别错误。

## 开发

```powershell
uv run --locked --group test -m unittest discover -s tests -v
```

测试不需要模型；音频容器检查需要 FFmpeg。`tools/` 还提供实际设备与播放器检查，其中部分需要本地测试素材和已准备模型。

```text
watch.py / watch.cmd     播放启动器
mpv_plugin.py           mpv 插件后端
npu_subtitles/          识别、翻译、字幕调度和播放器接入
player/mpv/             mpv Lua 插件
tools/models/           模型下载与离线转换
tools/                  安装与验证工具
tests/                  回归测试
docs/                   功能、模型准备与架构说明
third_party/            第三方许可和模型来源
```

依赖由 `pyproject.toml` 与 `uv.lock` 管理；`prepare` 用于模型准备，`test` 用于回归测试。`requirements.txt` 是默认播放环境的锁定导出。

进一步阅读：[功能范围](docs/FEATURE_SCOPE.md) · [架构说明](docs/REALTIME_SUBTITLES_ARCHITECTURE.md) · [字幕时序约束](docs/STREAMING_CAPTION_CONTRACTS.md)。

## 许可证

项目源码采用 [MIT License](LICENSE)。第三方适配代码及模型分别遵循其来源许可证，相关声明保存在 [third_party](third_party/)。
