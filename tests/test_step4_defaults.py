import unittest

from complex_document_rag.step4_basic_query import get_default_test_queries


class Step4DefaultQueryTests(unittest.TestCase):
    def test_default_queries_match_current_pdf_domain(self):
        queries = get_default_test_queries()

        self.assertEqual(len(queries), 3)
        self.assertIn("警示标识", queries[0])
        self.assertIn("紧急出口", queries[1])
        self.assertIn("苯", queries[2])
        self.assertNotIn("用户登录失败后怎么处理？", queries)


if __name__ == "__main__":
    unittest.main()
