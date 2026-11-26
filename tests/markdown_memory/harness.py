"""markdown_memory 集成测试的共用接线：真实插件目录与真实宿主，只替换模型与 embedding 的外部边界。"""
import asyncio
from collections.abc import Callable, Mapping
import json
from pathlib import Path
import shutil

from agent.plugin_composition import ServiceKey
from plugins.content.plugin import check_text
from plugins.ledger.contract import ContentPart, Input, Output

MARKDOWN_PROMPT = "你维护 Cable 的长期 Markdown 记忆"

# 只在模型边界按来源原文回答：提到乌龙茶的那轮写进 MEMORY.md 和中文名主题，其余返回空操作。
_MARKDOWN_REPLY = '''

def markdown_reply(request):
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
'''

# 确定性 embedding：几个概念词表决定方向，少量按正文哈希的分量让无关文本彼此也不完全相同。
# 正文含“触发故障”时模拟外部 embedding 服务失败。
_EMBEDDINGS = '''

markdown_apply = apply
async def apply(ctx):
    import hashlib
    import math
    from plugins.models.contract import EMBEDDINGS, EmbeddingResult, EmbeddingSpaceDescriptor, TransportError
    await markdown_apply(ctx)
    concepts = (("NAS", "云盘", "存储", "管理页面", "后台网址"), ("路由器", "WiFi", "网关"),
                ("茶", "乌龙", "饮品", "饮料"), ("出差", "上海", "行程", "机票"), ("打印机", "墨盒"))
    calls = []
    def vector(text):
        values = [float(sum(text.count(word) for word in words)) for words in concepts]
        digest = hashlib.sha256(text.encode()).digest()
        values += [digest[index] / 255 * 0.1 for index in range(4)] + [0.3]
        norm = math.sqrt(sum(value * value for value in values))
        return tuple(value / norm for value in values)
    class Embedding:
        descriptor = EmbeddingSpaceDescriptor(
            plugin_snapshot_id="fixture", model_revision=0, model="fixture",
            model_id="fixture", driver_id="fixture", driver_contract_version="1",
            connection_id="fixture", auth_identity="fixture", connection_fingerprint="fixture",
            dimensions=len(concepts) + 5, normalization="l2", capability_digest="fixture",
        )
        async def embed(self, texts):
            calls.append(list(texts))
            if any("触发故障" in text for text in texts):
                raise TransportError("fixture embedding 暂时不可用").exception()
            return EmbeddingResult(tuple(vector(text) for text in texts))
    class Embeddings:
        def describe(self, *, model_id=None):
            return Embedding.descriptor
        @asynccontextmanager
        async def bind(self, *, model_id=None):
            yield Embedding()
    await ctx.provide(EMBEDDINGS, Embeddings())
    await ctx.provide(ServiceKey("fixture.embed-calls"), calls)
'''


def add_markdown_memory(idle_minutes: int, *, unit_seconds: int = 0, embeddings: bool = False,
                        topics: Mapping[str, str] | None = None) -> Callable[[Path], None]:
    """安装真实 markdown_memory；空闲时钟缩到测试尺度（一"分钟"= unit_seconds 秒）。"""
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
        folder = sources.parent / "workspace/memory/topics"
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in (topics or {}).items():
            (folder / name).write_text(content, encoding="utf-8")
        provider = sources / "test_provider/plugin.py"
        source = provider.read_text(encoding="utf-8")
        marker = 'if "[Source messages]" in str(request.messages):'
        assert marker in source
        source = source.replace(
            marker,
            f'if {MARKDOWN_PROMPT!r} in str(request.messages):\n'
            '                return markdown_reply(request)\n'
            f'            {marker}',
        ) + _MARKDOWN_REPLY
        provider.write_text(source + (_EMBEDDINGS if embeddings else ""), encoding="utf-8")
    return install


def append_input(log, session: str, identity: str, text: str) -> None:
    log.writer(session, author="user", source="conversation", body_types=(Input,),
               content={"text": check_text}).append(identity, Input((ContentPart("text", text),)))


def append_turn(log, session: str, suffix: str, question: str, answer: str) -> None:
    append_input(log, session, f"{session}-u{suffix}", question)
    log.writer(session, author="assistant", source="conversation", body_types=(Output,),
               content={"text": check_text}).append(
        f"{session}-a{suffix}", Output((ContentPart("text", answer),), "complete"))


async def eventually(check: Callable[[], bool], *, timeout: float = 10.0) -> None:
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
