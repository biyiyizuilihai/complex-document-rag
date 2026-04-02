"""`web` 包的标准导出入口。"""

from complex_document_rag.web.backend import QueryBackend, get_query_backend
from complex_document_rag.web.routes import app, create_app

__all__ = ["QueryBackend", "app", "create_app", "get_query_backend"]
