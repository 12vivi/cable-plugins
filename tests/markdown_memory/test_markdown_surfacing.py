"""每轮浮现：关键词与向量两路，换个说法也能想起来；只注入相关条目；向量不可用时退回关键词。"""
from collections.abc import Mapping
import logging
from pathlib import Path

import pytest

from agent.plugin_composition import ServiceKey
from plugins.context.contract import MATERIALS_V4 as MATERIALS
from plugins.ledger.log import SessionAttributes
from tests.markdown_memory.harness import add_markdown_memory, append_input
from tests.test_default_reply import application

DEVICES = "\n".join((
    "---", "description: 家里的网络与存储设备", "---",
    "- NAS 管理页面在 5000 端口",
    "- 路由器管理员账号存在 1Password 里",
    "- 打印机墨盒型号是 HP 63",
    "- 书房的灯是智能灯泡",
    "- 电视盒子在客厅",
    "- 扫地机器人每周三清扫",
    "- 门锁电池半年换一次",
    "- 投影仪放在卧室",
    "",
))
TOPICS = {
    "家里设备.md": DEVICES,
    "饮品偏好.md": "\n".join(("---", "description: 用户喜欢的饮品", "---", "- 爱喝乌龙茶", "- 不喝含糖饮料", "")),
    "旅行.md": "\n".join(("---", "description: 出行计划", "---", "- 2026-12-02 去上海出差", "")),
}


async def surfaced(host, log, identity: str, text: str) -> str:
    """追加一条真实用户输入，经真实材料入口准备本轮，返回 Markdown 浮现提醒的正文。"""
    append_input(log, "s1", identity, text)
    root = host.live_root
    assert root is not None
    async with root.context.open_service(MATERIALS) as service, service.bind() as materials:
        prepared = await materials.prepare(log.reader("s1").snapshot(), "conversation")
    reminders = prepared["reminders"] if isinstance(prepared, Mapping) else prepared.reminders
    texts = [str(item["text"] if isinstance(item, Mapping) else item.text) for item in reminders]
    return "\n".join(text for text in texts if "## 相关记忆" in text)


@pytest.mark.asyncio
async def test_meaning_surfaces_relevant_entries_only(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """MEM-006：向量通道按意思浮现并只注入相关条目；外部 embedding 失败时带日志退回关键词，轮次照常完成。"""
    async with application(tmp_path, replying=False, compaction=True, extra_sources=add_markdown_memory(
            0, embeddings=True, topics=TOPICS)) as (log, host):
        log.ensure_session("s1", SessionAttributes())
        calls = host.live_root.context.require(ServiceKey("fixture.embed-calls"))

        # 1. 换个说法、没有共同字词也能想起来：只给相关的那条，其余提示用 read_file 读。
        text = await surfaced(host, log, "q1", "我那个私有云盘的后台网址是多少来着")
        assert "### topics/家里设备.md" in text and "：家里的网络与存储设备" in text
        assert "- NAS 管理页面在 5000 端口" in text
        assert "打印机" not in text and "扫地机器人" not in text
        assert "（另有 7 条未展示，需要时用 read_file 读取完整文件）" in text
        assert "饮品偏好" not in text and "旅行" not in text

        # 2. 条目向量按正文缓存：第二条输入只需要给查询本身做向量化。
        text = await surfaced(host, log, "q2", "最近想换点别的饮料")
        assert "### topics/饮品偏好.md" in text and "- 不喝含糖饮料" in text
        assert "家里设备" not in text
        assert calls[-1] == ["最近想换点别的饮料"]

        # 3. 和记忆无关的输入不浮现任何主题。
        assert await surfaced(host, log, "q3", "今天晚饭吃什么好呢") == ""

        # 4. 外部 embedding 失败：记一次降级日志，按关键词照常浮现（至少命中两个不同的词）。
        with caplog.at_level(logging.WARNING):
            text = await surfaced(host, log, "q4", "触发故障：打印机墨盒型号是多少")
        assert "### topics/家里设备.md" in text and "- 打印机墨盒型号是 HP 63" in text
        assert "NAS" not in text
        assert "向量通道不可用" in caplog.text


@pytest.mark.asyncio
async def test_keywords_alone_still_surface(tmp_path: Path) -> None:
    """没有 embedding provider 时只用关键词：至少命中两个不同的词才浮现，并且只给命中的条目。"""
    async with application(tmp_path, replying=False, compaction=True, extra_sources=add_markdown_memory(
            0, topics=TOPICS)) as (log, host):
        log.ensure_session("s1", SessionAttributes())
        text = await surfaced(host, log, "q1", "我还是爱喝乌龙茶吗")
        assert "### topics/饮品偏好.md" in text and "- 爱喝乌龙茶" in text
        assert "不喝含糖饮料" not in text
        # 只命中一个词（“墨盒”）的不浮现；换个说法在没有向量时想不起来。
        assert await surfaced(host, log, "q2", "该换墨盒了") == ""
        assert await surfaced(host, log, "q3", "我那个私有云盘的后台网址是多少来着") == ""
