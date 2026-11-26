"""对话停下后由真实后台提炼进 Markdown 记忆，不必等上下文压缩；同一 Turn 只学一次。"""
import asyncio
from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import sqlite3

import pytest

from agent.plugin_composition import ServiceKey
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.content.plugin import check_text
from plugins.context.contract import MATERIALS_V4 as MATERIALS
from plugins.ledger.contract import ContentPart, Input, Output
from plugins.ledger.log import SessionAttributes
from tests.test_default_reply import application

MARKDOWN_PROMPT = "你维护 Cable 的长期 Markdown 记忆"


def add_markdown_memory(idle_minutes: int, *, unit_seconds: int = 0):
    """安装真实 markdown_memory；只把空闲时钟缩到测试尺度（一"分钟"= unit_seconds 秒），并在模型边界按来源原文回答。"""
    def install(sources: Path) -> None:
        target = sources / "markdown_memory"
        shutil.copytree(Path(__file__).parents[2] / "plugins/markdown_memory", target,
                        ignore=shutil.ignore_patterns("__pycache__"))
        module = target / "plugin.py"
        text = module.read_text(encoding="utf-8")
        for old, new in (("_IDLE_SWEEP_SECONDS = 60.0", "_IDLE_SWEEP_SECONDS = 0.05"),
                         ("_IDLE_SWEEP_INITIAL_DELAY_SECONDS = 15.0", "_IDLE_SWEEP_INITIAL_DELAY_SECONDS = 0.0"),
                         ("_IDLE_UNIT_SECONDS = 60", f"_IDLE_UNIT_SECONDS = {unit_seconds}")):
            assert old in text
            text = text.replace(old, new)
        module.write_text(text, encoding="utf-8")
        from agent.plugin_composition.config_input import save_config
        save_config(sources.parent / "workspace/plugin-data/markdown_memory-builtin",
                    {"extract_idle_minutes": idle_minutes})
        # 与发行版 bundle 相同：组合配置把 Prompt 贡献权授予 markdown_memory，摘要仍归 compaction。
        save_config(sources.parent / "workspace/plugin-data/context-builtin", {
            "summary_source": ["compaction", "compaction"],
            "prompt_sources": {"markdown_memory": "markdown_memory"},
        })
        # agent 曾用不合规文件名写过的主题：后台照常给出参照，但必须标成只读。
        topics = sources.parent / "workspace/memory/topics"
        topics.mkdir(parents=True, exist_ok=True)
        (topics / "My Notes.md").write_text("---\ndescription: 手写笔记\n---\n- 旧笔记\n", encoding="utf-8")
        provider = sources / "test_provider/plugin.py"
        source = provider.read_text(encoding="utf-8")
        marker = 'if "[Source messages]" in str(request.messages):'
        assert marker in source
        provider.write_text(source.replace(
            marker,
            f'if {MARKDOWN_PROMPT!r} in str(request.messages):\n'
            '                return markdown_reply(request)\n'
            f'            {marker}',
        ) + '''

def markdown_reply(request):
    """按本批来源原文给出操作：只有提到乌龙茶的那轮值得记，其余返回空操作。"""
    import json
    prompt = "".join(str(message["content"]) for message in request.messages)
    rows = json.loads(prompt.split("本次精确来源：\\n", 1)[1])
    users = [row for row in rows if row["author"] == "user" and "乌龙茶" in json.dumps(row, ensure_ascii=False)]
    if not users:
        return LLMResponse(json.dumps({"ops": []}))
    reply = next(row for row in rows[rows.index(users[0]):] if row["author"] == "assistant")
    day = users[0]["at"][:10]
    return LLMResponse(json.dumps({"ops": [
        {"op": "add", "doc": "MEMORY.md", "section": "## 用户偏好", "line": "- 爱喝乌龙茶",
         "message_ids": [users[0]["message_id"]]},
        {"op": "create", "doc": "topics/出行计划.md", "description": "用户的出行与日程安排",
         "line": "- " + day + " 那周的周三去上海出差；Cable 答应前一天提醒收拾行李",
         "message_ids": [users[0]["message_id"], reply["message_id"]]},
    ]}, ensure_ascii=False))
''', encoding="utf-8")
    return install


def append_turn(log, session: str, suffix: str, question: str, answer: str) -> None:
    log.writer(session, author="user", source="conversation", body_types=(Input,),
               content={"text": check_text}).append(f"{session}-u{suffix}", Input((ContentPart("text", question),)))
    log.writer(session, author="assistant", source="conversation", body_types=(Output,),
               content={"text": check_text}).append(
        f"{session}-a{suffix}", Output((ContentPart("text", answer),), "complete"))


async def eventually(check, *, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("等待超时")
        await asyncio.sleep(0.05)


def markdown_requests(host) -> list[str]:
    calls = host.live_root.context.require(ServiceKey("fixture.calls"))
    prompts = ["".join(str(message["content"]) for message in call.messages) for call in calls]
    return [prompt for prompt in prompts if MARKDOWN_PROMPT in prompt]


def source_ids(prompt: str) -> list[str]:
    return [row["message_id"] for row in json.loads(prompt.split("本次精确来源：\n", 1)[1])]


@pytest.mark.asyncio
async def test_idle_conversation_is_learned_once_without_compaction(tmp_path: Path) -> None:
    """MEM-002/MEM-011（空闲提炼）：没压缩过的短对话停下后也进长期记忆，学过的 Turn 不再送去提炼。"""
    memory = tmp_path / "workspace/memory"
    async with application(tmp_path, replying=False, compaction=True,
                           extra_sources=add_markdown_memory(30)) as (log, host):
        log.ensure_session("s1", SessionAttributes())
        append_turn(log, "s1", "1", "我最近爱喝乌龙茶，下周三要去上海出差", "好的，周二晚上提醒你收拾行李")

        # 1. 真实后台在会话停下后提炼；中文名主题可以新建，用户偏好写进 MEMORY.md。
        await eventually(lambda: "- 爱喝乌龙茶" in (memory / "MEMORY.md").read_text(encoding="utf-8"))
        topic = memory / "topics/出行计划.md"
        await eventually(topic.exists)
        prompts = markdown_requests(host)
        assert len(prompts) == 1
        assert source_ids(prompts[0]) == ["s1-u1", "s1-a1"]
        # 来源带接纳时的本地时间与星期，相对日期才能换算；不合规文件名的主题标成只读。
        stamps = re.findall(r'"at":"(\d{4}-\d{2}-\d{2}) \d{2}:\d{2} 周[一二三四五六日]"', prompts[0])
        assert len(stamps) == 2
        assert f"- {stamps[0]} 那周的周三去上海出差" in topic.read_text(encoding="utf-8")
        assert "### topics/My Notes.md（只读：文件名不合规，不能写入）" in prompts[0]
        with closing(sqlite3.connect(memory / "markdown-profile-writes.db")) as connection:
            learned = connection.execute(
                "SELECT message_id, source_ref FROM idle_learned WHERE session_key='s1' ORDER BY message_id").fetchall()
            receipts = {kind for (kind,) in connection.execute(
                "SELECT kind FROM consolidation_writes WHERE source_ref='idle:s1:s1-a1'")}
        assert learned == [("s1-a1", "idle:s1:s1-a1"), ("s1-u1", "idle:s1:s1-a1")]
        assert {"markdown_ops_v1", "markdown_idle_learned_v1", "markdown_before_v1",
                "markdown_ops_applied_v1"} <= receipts

        # 2. 再说一轮：只把新的 Turn 送去提炼，已学过的不重复出现。
        await asyncio.sleep(0.3)
        assert len(markdown_requests(host)) == 1
        append_turn(log, "s1", "2", "谢谢", "不客气")
        await eventually(lambda: len(markdown_requests(host)) == 2)
        assert source_ids(markdown_requests(host)[1]) == ["s1-u2", "s1-a2"]
        await asyncio.sleep(0.3)
        assert len(markdown_requests(host)) == 2
        assert (memory / "MEMORY.md").read_text(encoding="utf-8").count("- 爱喝乌龙茶") == 1

        # 3. 注入的使用规则写明提炼时机、回忆时找不到依据就说记不清，以及中文主题名规则。
        root = host.live_root
        assert root is not None
        async with root.context.open_service(MATERIALS) as service, service.bind() as materials:
            prepared = str(await materials.prepare(log.reader("s1").snapshot(), "conversation"))
        assert "对话停下约 30 分钟后、以及上下文压缩后自动提炼" in prepared
        assert "找不到依据就直说记不清" in prepared
        assert "- topics/出行计划.md — 用户的出行与日程安排" in prepared


@pytest.mark.asyncio
async def test_summary_and_idle_paths_never_learn_a_turn_twice(tmp_path: Path) -> None:
    """MEM-004/MEM-011：长对话先被真实压缩摘要学走前段，停下后空闲提炼只补学摘要没覆盖的尾部。"""
    memory = tmp_path / "workspace/memory"
    async with application(tmp_path, replying=True, compaction=True,
                           extra_sources=add_markdown_memory(3, unit_seconds=1)) as (log, host):
        root = host.live_root
        assert root is not None
        accept = root.context.require(CHANNEL_INPUT)

        async def reply(identity: str, text: str) -> None:
            accepted = await accept("test:room", identity, ChannelInboundMessage(
                "test", "user", "room", text, datetime.now(UTC), {}))

            def finished() -> bool:
                rows = log.reader("test:room").snapshot()
                return any(isinstance(row.body, Output) and row.body.finish == "complete" and row.seq > accepted.seq
                           for row in rows)
            await eventually(finished)

        # 1. 先有一轮真实对话建立会话，再补上足以触发软水位的长历史；靠后、会进入摘要的一段提到乌龙茶。
        await reply("u1", "你好")
        filler = "这是一段很长的闲聊，用来把上下文撑到压缩水位。" * 50
        for index in range(16):
            fact = "我最近爱喝乌龙茶。" if index == 14 else ""
            append_turn(log, "test:room", f"f{index}", f"第 {index} 段：{fact}{filler}", f"收到第 {index} 段。{filler}")
        await reply("u2", "我们刚才聊到哪了？")

        # 2. 这一轮真实请求触发了压缩摘要，跟随者处理过引用摘要的 Output。
        rows = log.reader("test:room").snapshot()
        summarized = [row for row in rows if isinstance(row.body, Output)
                      and any(isinstance(part, ContentPart) and part.kind == "context.summary"
                              for part in row.body.parts)]
        assert summarized, "长历史应当触发一次真实的压缩摘要"

        # 3. 摘要路径学到乌龙茶（摘要覆盖的消息空闲路径不再碰，放不下而省略的按 MEM-011 也不恢复资格）；
        #    会话停下后，空闲路径补学摘要没覆盖的最后一轮。
        await eventually(lambda: "- 爱喝乌龙茶" in (memory / "MEMORY.md").read_text(encoding="utf-8"), timeout=20)

        def idle_rows() -> set[str]:
            with closing(sqlite3.connect(memory / "markdown-profile-writes.db")) as connection:
                return {message_id for (message_id,) in connection.execute(
                    "SELECT message_id FROM idle_learned WHERE session_key='test:room'")}
        await eventually(lambda: summarized[-1].message_id in idle_rows(), timeout=20)
        with closing(sqlite3.connect(memory / "markdown-profile-writes.db")) as connection:
            orders = connection.execute(
                "SELECT count(*) FROM consolidation_writes WHERE kind='markdown_projection_order_v1'").fetchone()[0]
        assert orders >= 1

        # 4. 两条路径加起来，每条消息最多被送去提炼一次；乌龙茶只记了一条。
        await asyncio.sleep(0.5)
        sent = [identity for prompt in markdown_requests(host) for identity in source_ids(prompt)]
        assert sent and len(sent) == len(set(sent))
        assert (memory / "MEMORY.md").read_text(encoding="utf-8").count("- 爱喝乌龙茶") == 1


@pytest.mark.asyncio
async def test_idle_learning_can_be_turned_off(tmp_path: Path) -> None:
    """extract_idle_minutes=0 时恢复为只在上下文压缩后提炼，注入的规则也如实说明。"""
    memory = tmp_path / "workspace/memory"
    async with application(tmp_path, replying=False, compaction=True,
                            extra_sources=add_markdown_memory(0)) as (log, host):
        log.ensure_session("s1", SessionAttributes())
        append_turn(log, "s1", "1", "我最近爱喝乌龙茶", "记住了")
        await asyncio.sleep(0.5)
        assert markdown_requests(host) == []
        assert "乌龙茶" not in (memory / "MEMORY.md").read_text(encoding="utf-8")
        root = host.live_root
        assert root is not None
        async with root.context.open_service(MATERIALS) as service, service.bind() as materials:
            prepared = str(await materials.prepare(log.reader("s1").snapshot(), "conversation"))
        assert "后台也会在上下文压缩后自动提炼" in prepared
