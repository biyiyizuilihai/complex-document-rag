import unittest

from complex_document_rag.retrieval.reranking import SiliconFlowReranker, rerank_retrieval_bundle


class _FakeNode:
    def __init__(self, text, score=0.0, metadata=None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _FakeResponse(self.payload)


class SiliconFlowRerankerTests(unittest.TestCase):
    def test_reranks_nodes_using_returned_indices(self):
        session = _FakeSession(
            {
                "results": [
                    {"index": 2, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.67},
                ]
            }
        )
        reranker = SiliconFlowReranker(
            api_key="secret",
            api_base="https://api.siliconflow.cn/v1",
            model_name="BAAI/bge-reranker-v2-m3",
            session=session,
            top_n=2,
        )
        nodes = [
            _FakeNode("文本块A", score=0.3),
            _FakeNode("文本块B", score=0.2),
            _FakeNode("文本块C", score=0.1),
        ]

        reranked = reranker.rerank("测试问题", nodes)

        self.assertEqual([node.text for node in reranked], ["文本块C", "文本块A"])
        self.assertAlmostEqual(reranked[0].score, 0.98)
        self.assertEqual(session.calls[0]["url"], "https://api.siliconflow.cn/v1/rerank")
        self.assertEqual(session.calls[0]["json"]["query"], "测试问题")
        self.assertEqual(session.calls[0]["json"]["documents"], ["文本块A", "文本块B", "文本块C"])

    def test_returns_original_nodes_when_api_key_missing(self):
        reranker = SiliconFlowReranker(
            api_key="",
            api_base="https://api.siliconflow.cn/v1",
            model_name="BAAI/bge-reranker-v2-m3",
        )
        nodes = [_FakeNode("文本块A"), _FakeNode("文本块B")]

        reranked = reranker.rerank("测试问题", nodes)

        self.assertIs(reranked, nodes)

    def test_rerank_retrieval_bundle_reranks_each_modality(self):
        class _StubReranker:
            def __init__(self):
                self.calls = []

            def rerank(self, query, nodes, top_n=None):
                self.calls.append((query, [node.text for node in nodes], top_n))
                return list(reversed(nodes[:top_n]))

        reranker = _StubReranker()
        retrieval = {
            "text_results": [_FakeNode("文本A"), _FakeNode("文本B")],
            "image_results": [_FakeNode("图片A"), _FakeNode("图片B")],
            "table_results": [_FakeNode("表格A"), _FakeNode("表格B")],
        }

        reranked = rerank_retrieval_bundle(
            query="测试问题",
            retrieval=retrieval,
            reranker=reranker,
            top_n_map={"text_results": 2, "image_results": 1, "table_results": 2},
        )

        self.assertEqual([node.text for node in reranked["text_results"]], ["文本B", "文本A"])
        self.assertEqual([node.text for node in reranked["image_results"]], ["图片A"])
        self.assertEqual([node.text for node in reranked["table_results"]], ["表格B", "表格A"])
        self.assertEqual(
            reranker.calls,
            [
                ("测试问题", ["文本A", "文本B"], 2),
                ("测试问题", ["图片A", "图片B"], 1),
                ("测试问题", ["表格A", "表格B"], 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
