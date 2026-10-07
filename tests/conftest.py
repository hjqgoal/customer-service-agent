"""测试夹具。"""

import pytest

from customer_service_agent.config import Settings


@pytest.fixture()
def test_settings(tmp_path) -> Settings:
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    return Settings(
        _env_file=None,
        app_env="test",
        agent_mode="single",
        chat_api_key="test-chat-key",
        chat_base_url="https://chat.example.test/v1",
        chat_model_name="test-chat-model",
        embedding_api_key="test-embedding-key",
        embedding_base_url="https://embedding.example.test/v1",
        embedding_model_name="test-embedding-model",
        memory_db_path=tmp_path / "memory.sqlite3",
        multi_agent_db_path=tmp_path / "multi.sqlite3",
        after_sales_db_path=tmp_path / "after_sales.sqlite3",
        profile_db_path=tmp_path / "profiles.sqlite3",
        rag_db_path=tmp_path / "rag.sqlite3",
        ticket_db_path=tmp_path / "tickets.sqlite3",
        knowledge_base_dir=kb_dir,
        agent_verbose=False,
        memory_recent_turns=2,
    )
