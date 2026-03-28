from __future__ import annotations

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


def delete_doc_vectors(
    client: QdrantClient,
    doc_id: str,
    collection_names: list[str] | None = None,
) -> None:
    selector = models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id),
                )
            ]
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
