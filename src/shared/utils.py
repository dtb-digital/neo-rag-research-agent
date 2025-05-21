"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    
    # Strukturert formatering for lovdata-dokumenter med spesialbehandling av metadata
    # Bruk strukturert XML for å gjøre det lettere for Claude å lese
    struktur_elementer = ["lov_id", "lov_navn", "lov_tittel", "kapittel_nr", 
                          "kapittel_tittel", "paragraf_nr", "paragraf_tittel"]
    
    # Samle strukturerte metadata-elementer
    struktur_metadata = ""
    for key in struktur_elementer:
        if key in metadata and metadata[key]:
            struktur_metadata += f'<{key}>{metadata[key]}</{key}>\n'
    
    # Samle resterende metadata i generisk format
    andre_metadata = ""
    for k, v in metadata.items():
        if k not in struktur_elementer and v is not None:
            andre_metadata += f'<meta key="{k}">{v}</meta>\n'
    
    # Kombinert XML-output
    return f"""<document>
<content>
{doc.page_content}
</content>
<metadata>
{struktur_metadata}
{andre_metadata}
</metadata>
</document>"""


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)
