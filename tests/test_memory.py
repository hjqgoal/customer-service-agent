import json
from typing import Any

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from customer_service_agent.memory import CustomerProfileRepository, CustomerProfileUpdate
from customer_service_agent.memory.compaction import ConversationCompactor
from tests.fakes import FakeToolCallingChatModel


class RecordingStructuredSummaryModel(FakeToolCallingChatModel):
    _recorded_inputs: list[str] = PrivateAttr(default_factory=list)

    @property
    def recorded_inputs(self) -> list[str]:
        return self._recorded_inputs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        source = str(messages[-1].content)
        self._recorded_inputs.append(source)
        if source.startswith("待合并摘要"):
            candidates = json.loads(source.split("\n\n", maxsplit=1)[1])
            facts = list(
                dict.fromkeys(
                    fact for candidate in candidates for fact in candidate["confirmed_facts"]
                )
            )
            current_request = next(
                (
                    candidate["current_request"]
                    for candidate in reversed(candidates)
                    if candidate["current_request"]
                ),
                "",
            )
        else:
            facts = [
                marker for marker in ("BEGIN_FACT", "MIDDLE_FACT", "END_FACT") if marker in source
            ]
            current_request = facts[-1] if facts else ""

        payload = {
            "confirmed_facts": facts,
            "current_request": current_request,
            "tool_results": [],
            "promises": [],
            "pending_actions": [],
        }
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=json.dumps(payload, ensure_ascii=False)))
            ]
        )


def test_customer_profile_repository_merges_and_deletes(tmp_path):
    repository = CustomerProfileRepository(tmp_path / "profiles.sqlite3")

    repository.upsert(
        "customer-1",
        CustomerProfileUpdate(
            customer_name="李明",
            order_ids=["ORDER-1"],
            preferences=["邮件联系"],
        ),
    )
    profile = repository.upsert(
        "customer-1",
        CustomerProfileUpdate(
            email="liming@example.com",
            order_ids=["ORDER-1", "ORDER-2"],
        ),
    )

    assert profile.customer_name == "李明"
    assert profile.email == "liming@example.com"
    assert profile.order_ids == ["ORDER-1", "ORDER-2"]
    assert profile.preferences == ["邮件联系"]
    assert repository.delete("customer-1") is True
    assert repository.get("customer-1") is None


def test_compactor_summarizes_every_chunk_and_reduces_structured_results():
    model = RecordingStructuredSummaryModel()
    compactor = ConversationCompactor(
        model,
        recent_turns=2,
        summary_max_chars=1000,
        summary_input_max_chars=600,
    )
    padding = "上下文" * 120
    messages = [
        HumanMessage(content=f"BEGIN_FACT {padding}"),
        AIMessage(content=padding),
        HumanMessage(content=f"MIDDLE_FACT {padding}"),
        AIMessage(content=padding),
        HumanMessage(content=f"END_FACT {padding}"),
    ]

    summary = compactor._summarize(messages)

    assert summary.confirmed_facts == ["BEGIN_FACT", "MIDDLE_FACT", "END_FACT"]
    assert summary.current_request == "END_FACT"
    map_inputs = [
        source for source in model.recorded_inputs if source.startswith("需要压缩的历史片段")
    ]
    merge_inputs = [source for source in model.recorded_inputs if source.startswith("待合并摘要")]
    assert len(map_inputs) >= 3
    assert merge_inputs
    assert "MIDDLE_FACT" in "".join(map_inputs)


def test_compactor_rejects_old_free_form_summary_shape():
    compactor = ConversationCompactor(
        FakeToolCallingChatModel(),
        recent_turns=2,
    )

    with pytest.raises(ValueError):
        compactor._parse_model_summary(AIMessage(content='{"summary": "一段自然语言"}'))
