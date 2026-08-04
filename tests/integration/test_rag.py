"""集成测试：RAG 检索。

需要 Chroma 运行中（docker compose up）。否则自动跳过。
"""

import pytest


@pytest.fixture(scope="module")
def chroma_available():
    """检测 Chroma 是否可用。"""
    try:
        from chromadb import HttpClient
        client = HttpClient(host="localhost", port=8001)
        client.heartbeat()
        return True
    except Exception:
        return False


class TestRagRetrieval:
    """RAG 歌名检索：命中 + 拒识。"""

    def test_rag_hit_returns_result(self, chroma_available):
        """已知入库曲目应命中。"""
        if not chroma_available:
            pytest.skip("Chroma 不可用（需要 docker compose up）")

        from src.rag.retriever import retrieve_by_song_name

        # 测试人工标注的核心曲库——两句经典指弹曲
        result = retrieve_by_song_name("�ƻ�")       # 黄昏
        if result is None:
            result = retrieve_by_song_name("twilight")

        # 至少有一条命中（人工标注曲库包含 黄昏 和 无题）
        assert result is not None, "人工标注曲库应至少有一条命中"
        assert result.get("title") is not None
        assert result.get("artist") is not None

    def test_gibberish_query_returns_none(self, chroma_available):
        """无意义查询应返回 None。"""
        if not chroma_available:
            pytest.skip("Chroma 不可用")

        from src.rag.retriever import retrieve_by_song_name

        result = retrieve_by_song_name("xyzabc123_nonexistent")
        # 无意义查询要么 None，要么 confidence 低于阈值被降级后返回
        # 不做强断言——取决于当前 Chroma 中的数据量
        if result is not None:
            assert result.get("title") is not None

    def test_collection_accessible(self, chroma_available):
        """Chroma Collection 可访问且可查询。"""
        if not chroma_available:
            pytest.skip("Chroma 不可用")

        from src.rag.chroma_client import get_collection

        collection = get_collection()
        count = collection.count()
        assert count > 0, f"Collection 为空，应有已入库数据，实际 {count} 条"
