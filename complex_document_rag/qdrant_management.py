from __future__ import annotations

import os

from qdrant_client import QdrantClient, models

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_IMAGE_COLLECTION,
    QDRANT_TABLE_COLLECTION,
    QDRANT_TEXT_COLLECTION,
)


def managed_collection_names() -> list[str]:
    return [
        QDRANT_TEXT_COLLECTION,
        QDRANT_IMAGE_COLLECTION,
        QDRANT_TABLE_COLLECTION,
    ]


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _doc_filter_conditions(
    doc_id: str,
    source_path: str | None = None,
) -> list[models.FieldCondition]:
    should_conditions: list[models.FieldCondition] = [
        models.FieldCondition(
            key="source_doc_id",
            match=models.MatchValue(value=doc_id),
        )
    ]
    if source_path:
        abs_source_path = os.path.abspath(source_path)
        source_basename = os.path.basename(abs_source_path)
        should_conditions.extend(
            [
                models.FieldCondition(
                    key="source_path",
                    match=models.MatchValue(value=abs_source_path),
                ),
                models.FieldCondition(
                    key="source_document_path",
                    match=models.MatchValue(value=abs_source_path),
                ),
                models.FieldCondition(
                    key="source_document_path",
                    match=models.MatchValue(value=source_basename),
                ),
            ]
        )
    return should_conditions


def count_doc_vectors(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
    source_path: str | None = None,
) -> int:
    response = client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            should=_doc_filter_conditions(doc_id=doc_id, source_path=source_path)
        ),
        exact=True,
    )
    return int(response.count)


def delete_doc_vectors(
    client: QdrantClient,
    doc_id: str,
    source_path: str | None = None,
    collection_names: list[str] | None = None,
) -> None:
    selector = models.FilterSelector(
        filter=models.Filter(
            should=_doc_filter_conditions(doc_id=doc_id, source_path=source_path)
        )
    )
    for collection_name in collection_names or managed_collection_names():
        client.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )


def drop_managed_collections(
    client: QdrantClient,
    collection_names: list[str] | None = None,
) -> None:
    for collection_name in collection_names or managed_collection_names():
        client.delete_collection(collection_name=collection_name)
