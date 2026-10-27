# 模型准备

所有命令从项目根目录执行。下载器固定来源并验证哈希，模型放在 `models/`，NPU 编译缓存放在 `.cache/`。下载、转换与播放分开执行。

## 英语

```powershell
uv run --locked --group prepare -m tools.models.moonshine
uv run --locked --group prepare -m tools.models.mtran
```

Moonshine Medium 用同一命令加 `--size medium`；播放时加 `--moonshine medium`。静态图缓存按模型来源、配置和形状隔离。

## Granite 与日语

Granite 转换需要独立环境，避免播放依赖包含 PyTorch 和 Transformers：

```powershell
uv run --locked --group prepare -m tools.models.granite --download-only
$previousEnvironment = $env:UV_PROJECT_ENVIRONMENT
try {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-granite-export'
    uv --cache-dir .cache/uv sync --locked --project tools/granite-export
    if ($LASTEXITCODE -ne 0) { throw 'Granite environment preparation failed' }
    .venv-granite-export\Scripts\python.exe -m tools.models.granite --export-only
    if ($LASTEXITCODE -ne 0) { throw 'Granite export failed' }
} finally {
    if ($null -eq $previousEnvironment) {
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    } else {
        $env:UV_PROJECT_ENVIRONMENT = $previousEnvironment
    }
}
uv run --locked --group prepare -m tools.models.japanese_translation
```

日语翻译复用英中模型，并使用日英模型中转；显示保留日文原文。两个阶段共用同一期限和取消信号。

## 中文识别与其他文字字幕

```powershell
uv run --locked --group prepare -m tools.models.sensevoice
uv run --locked --group prepare -m tools.models.subtitles
# 也可只准备需要的语言
uv run --locked --group prepare -m tools.models.subtitles --languages ko fr
```

七种额外文字字幕语言需要通过准备检查才会被自动选用。运行时只加载当前需要的方向和英中模型。

模型许可和来源记录位于 `third_party/` 与各模型的 `source.json`。日语还需要先完成上面的英中模型准备。
