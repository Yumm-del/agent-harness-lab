# -*- coding: utf-8 -*-
"""Usage 归一化测试，重点覆盖 DeepSeek 的缓存字段。"""

from __future__ import annotations

from types import SimpleNamespace

from harness_lab.models import Usage


def test_usage_from_dict_reads_cache_fields():
    usage = Usage.from_api(
        {
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "total_tokens": 1280,
            "prompt_cache_hit_tokens": 960,
            "prompt_cache_miss_tokens": 240,
        }
    )
    assert usage.cache_hit_tokens == 960
    assert usage.cache_miss_tokens == 240


def test_usage_from_object_falls_back_to_model_extra():
    """OpenAI SDK 会把未声明字段放进 model_extra，不能只用 getattr。"""
    usage = SimpleNamespace(
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        model_extra={"prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 100},
    )
    result = Usage.from_api(usage)
    assert result.cache_hit_tokens == 400
    assert result.cache_miss_tokens == 100


def test_usage_add_sums_cache_fields():
    a = Usage(prompt_tokens=10, cache_hit_tokens=6, cache_miss_tokens=4)
    b = Usage(prompt_tokens=20, cache_hit_tokens=12, cache_miss_tokens=8)
    a.add(b)
    assert a.prompt_tokens == 30
    assert a.cache_hit_tokens == 18
    assert a.cache_miss_tokens == 12
