from contextvars import ContextVar
import hashlib
import json

import pytest

from jitmind.agents.research_agent import ResearchAgent
from jitmind.observability import InMemoryTraceSink, JsonlTraceSink, TraceRecorder
from jitmind.schemas import Hit


def test_trace_redacts_query_content_by_default() -> None:
    recorder = TraceRecorder("private customer question")
    with recorder.stage("retriever.keyword", input_count=1) as stage:
        stage.output_count = 3
    trace = recorder.finish()

    assert trace.query is None
    assert trace.query_sha256 == hashlib.sha256(
        b"private customer question"
    ).hexdigest()
    assert trace.query_chars == 25
    assert trace.stages[0].output_count == 3
    assert trace.duration_ms >= trace.stages[0].duration_ms


def test_stage_records_failure_without_swallowing_it() -> None:
    recorder = TraceRecorder("question")
    with pytest.raises(RuntimeError):
        with recorder.stage("retriever.graph"):
            raise RuntimeError("database unavailable")

    assert recorder.trace.stages[0].status == "error"
    assert recorder.trace.stages[0].error_type == "RuntimeError"


def test_jsonl_sink_emits_one_parseable_record(tmp_path) -> None:
    trace = TraceRecorder("question").finish()
    path = tmp_path / "retrieval.jsonl"

    JsonlTraceSink(path).emit(trace)

    record = json.loads(path.read_text().strip())
    assert record["trace_id"] == trace.trace_id
    assert record["query"] is None


def test_research_agent_records_retrieval_and_fusion_evidence() -> None:
    agent = ResearchAgent.__new__(ResearchAgent)
    agent.trace_capture_content = False
    agent.trace_sink = InMemoryTraceSink()
    agent._trace_context = ContextVar("test_trace", default=None)
    agent._last_trace = None
    recorder = agent._begin_trace("where does Alice work?")
    hit = Hit(
        page_id="7",
        snippet="Alice works at Acme",
        source="keyword",
        meta={"rrf_score": 0.2},
    )

    result = agent._timed_retrieval(
        "retriever.keyword", lambda queries, top_k: [[hit]], ["Alice"], 5
    )
    agent._record_fusion_trace({"keyword": result[0]}, 1, [hit])
    trace = agent._finish_trace(recorder)

    assert [stage.name for stage in trace.stages] == [
        "retriever.keyword",
        "retrieval.fusion",
    ]
    assert trace.stages[0].output_count == 1
    assert trace.stages[1].attributes["top_evidence"][0]["page_id"] == "7"
    assert agent.explain_last_retrieval()["query"] is None
    assert agent.trace_sink.traces == [trace]

