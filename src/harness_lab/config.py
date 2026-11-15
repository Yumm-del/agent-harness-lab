# -*- coding: utf-8 -*-
"""项目路径、.env 读取和 API 配置。

这里不依赖 python-dotenv：格式很简单，自己解析可以少一个依赖，
也方便在 README 里解释「密钥只从环境变量或 .env 读，不写进代码」。
"""

from __future__ import annotations

import os
from pathlib import Path

# config.py 位于 <project>/src/harness_lab/config.py
# parents[0]=harness_lab, parents[1]=src, parents[2]=项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: str | Path | None = None, override: bool = False) -> Path | None:
    """读取一个 .env 文件，把键值写进 os.environ。

    参数：
      path     — .env 路径；缺省用项目根目录的 .env。
      override — 是否覆盖已经存在的环境变量。默认 False，
                 这样 shell 里显式 export 的值优先于文件。

    返回：
      实际读取的文件路径；文件不存在时返回 None。

    原理：
      逐行解析 KEY=VALUE，跳过空行和 # 注释；
      支持 `export KEY=VALUE`、单双引号包裹的值。
      不处理多行值（API key 不需要）。
    """
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return env_path


def get_api_key() -> str:
    """读取 DeepSeek / OpenAI 兼容接口的 API Key。

    优先 DEEPSEEK_API_KEY，其次 OPENAI_API_KEY。
    缺失时给出可执行的修复提示，而不是等到 API 调用才报 401。
    """
    load_env_file()
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "缺少 API Key。请在项目根目录创建 .env（可从 .env.example 复制），"
            "写入 DEEPSEEK_API_KEY=sk-...；或在 shell 里 export DEEPSEEK_API_KEY。"
        )
    return key


def get_base_url() -> str:
    """DeepSeek 默认地址；也允许通过环境变量切换到其他 OpenAI 兼容服务。"""
    load_env_file()
    return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def get_default_model() -> str:
    """默认模型；实验配置里可以逐个 spec 覆盖。"""
    load_env_file()
    return os.environ.get("HARNESS_MODEL", "deepseek-chat")


def ensure_project_dirs() -> None:
    """创建运行产物目录（runs/results），失败时让调用方决定是否中断。"""
    (PROJECT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
