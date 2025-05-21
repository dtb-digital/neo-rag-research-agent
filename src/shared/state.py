"""Shared functions for state management."""

import hashlib
import uuid
from typing import Any, Literal, Optional, Union

from langchain_core.documents import Document


def _generate_uuid(page_content: str) -> str:
    """Generate a UUID for a document based on page content."""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(
                    item.get("page_content", "")
                )

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                # Generer en mer robust unik ID som kombinerer flere felt for å minimere duplikatfjerning
                doc_id_parts = []
                
                # Legg til dokument-ID hvis tilgjengelig
                if item.metadata.get("id"):
                    doc_id_parts.append(f"id:{item.metadata['id']}")
                    
                # Legg til paragraf-nummer hvis tilgjengelig
                if item.metadata.get("paragraf_nr"):
                    doc_id_parts.append(f"p:{item.metadata['paragraf_nr']}")
                    
                # Legg til kapittel hvis tilgjengelig
                if item.metadata.get("kapittel_nr"):
                    doc_id_parts.append(f"k:{item.metadata['kapittel_nr']}")
                    
                # Legg til chunk_id hvis tilgjengelig
                if item.metadata.get("chunk_id"):
                    doc_id_parts.append(f"c:{item.metadata['chunk_id']}")
                
                # Legg til en kort hash av innholdet
                content_hash = _generate_uuid(item.page_content[:50])[:8]
                doc_id_parts.append(f"h:{content_hash}")
                
                # Kombiner alle deler til en unik ID
                doc_specific_id = "_".join(doc_id_parts) if doc_id_parts else item.page_content[:50]
                
                # Generer endelig ID
                item_id = _generate_uuid(doc_specific_id) if doc_specific_id else item.metadata.get("uuid", "")
                
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list
