"""
Export Chroma static collections into DKG-ready asset payloads.

This does NOT call DKG. It prepares JSONL you can POST to your DKG node.
Each line = one asset (asset-per-doc). Chunks retain their original
chunk_id and chunk_hash for 1:1 matching.
"""

import argparse
import json
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional

from .provider import StaticProvider
from ..config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _fetch_all_chunks(collection) -> List[Dict[str, Any]]:
    """Fetch all chunks from a Chroma collection (paged)."""
    total = collection.count()
    page_size = 200
    chunks = []
    offset = 0

    while offset < total:
        batch = collection.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        docs = batch.get("documents", [])
        metas = batch.get("metadatas", [])

        for doc, meta in zip(docs, metas):
            chunks.append(
                {
                    "content": doc,
                    "metadata": meta or {},
                }
            )
        offset += page_size
        logger.info("Fetched %s/%s chunks", min(offset, total), total)

    return chunks


def _convert_to_jsonld(asset: Dict[str, Any], context: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert custom asset format to valid JSON-LD.
    
    Args:
        asset: Asset data dictionary
        context: JSON-LD context (defaults to "https://schema.org" if not provided)
                 Can be a string URL or a dict for multiple contexts
    
    DKG-compatible format:
    - @context: JSON-LD context (defaults to schema.org)
    - @type: schema.org type (e.g., "CreativeWork", "Dataset", "Collection")
    - Standard schema.org properties
    - Custom data goes in additionalProperty (valid schema.org pattern)
    """
    chunks = asset.get("chunks", [])
    metadata = asset.get("metadata", {})
    
    if context is None:
        context = asset.get("@context", "https://schema.org")
    
    additional_props = [
        {"@type": "PropertyValue", "name": "source", "value": asset.get("source", "")},
        {"@type": "PropertyValue", "name": "isLatest", "value": str(asset.get("is_latest", True))},
        {"@type": "PropertyValue", "name": "subdirectory", "value": metadata.get("subdirectory", "")},
    ]
    
    has_part = []
    for ch in chunks:
        ch_meta = ch.get("metadata", {})
        chunk_props = [
            {"@type": "PropertyValue", "name": "chunkId", "value": ch.get("chunk_id", "")},
            {"@type": "PropertyValue", "name": "chunkHash", "value": ch.get("chunk_hash", "")},
            {"@type": "PropertyValue", "name": "startOffset", "value": str(ch_meta.get("start_offset", ""))},
            {"@type": "PropertyValue", "name": "endOffset", "value": str(ch_meta.get("end_offset", ""))},
            {"@type": "PropertyValue", "name": "source", "value": ch_meta.get("source", "")},
        ]
        has_part.append({
            "@type": "CreativeWork",
            "identifier": ch.get("chunk_id", ""),
            "text": ch.get("content", ""),
            "additionalProperty": chunk_props,
        })
    
    jsonld = {
        "@context": context,
        "@type": "CreativeWork",
        "name": asset.get("title", asset.get("doc_id", "Untitled Document")),
        "description": metadata.get("description", "") or "Documentation asset",
        "url": metadata.get("source_url", "") or None,
        "identifier": asset.get("doc_id", ""),
        "version": asset.get("version", 1),
        "dateCreated": metadata.get("created_at", "") or None,
        "dateModified": metadata.get("updated_at", "") or None,
        "keywords": metadata.get("tags", []) or [],
        "additionalProperty": additional_props,
        "hasPart": has_part,
    }
    
    jsonld = {k: v for k, v in jsonld.items() if v is not None}
    
    return jsonld


def _group_by_doc(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Group chunks by doc_id into asset payloads.

    Asset payload shape (per line):
    {
        "asset_ual": null,
        "doc_id": "...",
        "title": "...",
        "source": "...",
        "version": meta.version,
        "is_latest": meta.is_latest,
        "metadata": {...},
        "chunks": [
            {
                "chunk_id": "...",
                "chunk_hash": "...",
                "content": "...",
                "metadata": {...}
            }
        ]
    }
    """
    assets = defaultdict(lambda: {"chunks": []})

    for chunk in chunks:
        meta = chunk["metadata"]
        doc_id = meta.get("doc_id") or meta.get("document_id") or "unknown_doc"
        asset = assets[doc_id]

        if "doc_id" not in asset:
            asset["asset_ual"] = None
            asset["doc_id"] = doc_id
            asset["title"] = meta.get("title", "")
            asset["source"] = meta.get("source", "static_documentation")
            asset["version"] = meta.get("version", 1)
            asset["is_latest"] = meta.get("is_latest", True)
            asset["metadata"] = {
                "source_url": meta.get("source_url", ""),
                "subdirectory": meta.get("subdirectory", ""),
                "tags": meta.get("tags", []),
                "created_at": meta.get("created_at", ""),
                "updated_at": meta.get("updated_at", ""),
                "description": meta.get("description", ""),
            }

        asset["chunks"].append(
            {
                "chunk_id": meta.get("chunk_id") or chunk["id"],
                "chunk_hash": meta.get("chunk_hash", ""),
                "content": chunk["content"],
                "metadata": {
                    "start_offset": meta.get("start_offset"),
                    "end_offset": meta.get("end_offset"),
                    "chunk_tokens": meta.get("chunk_tokens"),
                    "chunk_size": meta.get("chunk_size"),
                    "version": meta.get("version", 1),
                    "is_latest": meta.get("is_latest", True),
                    "source": meta.get("source", ""),
                },
            }
        )

    return assets


def export_docs_per_doc(output_path: str, context: Optional[str] = None, source_filter: Optional[str] = None) -> None:
    """
    Export the docs collection to JSONL (asset-per-doc).
    
    Args:
        output_path: Path to write JSONL file
        context: Optional JSON-LD context (defaults to "https://schema.org")
        source_filter: Optional source type to filter by (e.g., "polka_wiki"). 
                      If None, exports all sources.
    """
    provider = StaticProvider(
        openai_api_key=Config.OPENAI_API_KEY,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        embedding_model=Config.OPENAI_EMBEDDING_MODEL,
    )
    collection = provider.get_docs_collection()
    chunks = _fetch_all_chunks(collection)
    
    if source_filter:
        filtered_chunks = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if meta.get("source") == source_filter:
                filtered_chunks.append(chunk)
        chunks = filtered_chunks
        logger.info(f"Filtered to {len(chunks)} chunks from source '{source_filter}'")
    
    assets = _group_by_doc(chunks)

    with open(output_path, "w", encoding="utf-8") as f:
        for asset in assets.values():
            jsonld = _convert_to_jsonld(asset, context=context)
            f.write(json.dumps(jsonld, ensure_ascii=False) + "\n")

    logger.info("Exported %s assets to %s", len(assets), output_path)


def export_single_assets(output_path: str, context: Optional[str] = None) -> None:
    """
    Export two assets total:
    - One aggregated asset for docs (all chunks)
    - One aggregated asset for AAG (all chunks)
    """
    provider = StaticProvider(
        openai_api_key=Config.OPENAI_API_KEY,
        chroma_persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        embedding_model=Config.OPENAI_EMBEDDING_MODEL,
    )

    exports = []

    docs_collection = provider.get_docs_collection()
    docs_chunks = _fetch_all_chunks(docs_collection)
    docs_asset = {
        "asset_ual": None,
        "asset_id": "static_docs",
        "title": "Static Docs Corpus",
        "source": "static_docs",
        "version": 1,
        "is_latest": True,
        "metadata": {
            "collection": "static_docs",
            "chunk_count": len(docs_chunks),
        },
        "chunks": [],
    }
    for ch in docs_chunks:
        meta = ch["metadata"]
        docs_asset["chunks"].append(
            {
                "chunk_id": meta.get("chunk_id"),
                "chunk_hash": meta.get("chunk_hash", ""),
                "content": ch["content"],
                "metadata": {
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "source_url": meta.get("source_url"),
                    "subdirectory": meta.get("subdirectory"),
                    "tags": meta.get("tags", []),
                    "start_offset": meta.get("start_offset"),
                    "end_offset": meta.get("end_offset"),
                    "version": meta.get("version", 1),
                    "is_latest": meta.get("is_latest", True),
                    "source": meta.get("source", ""),
                },
            }
        )
    exports.append(docs_asset)

    aag_collection = provider.get_aag_collection()
    aag_chunks = _fetch_all_chunks(aag_collection)
    aag_asset = {
        "asset_ual": None,
        "asset_id": "static_aag",
        "title": "AAG Transcripts Corpus",
        "source": "static_aag",
        "version": 1,
        "is_latest": True,
        "metadata": {
            "collection": "static_aag",
            "chunk_count": len(aag_chunks),
        },
        "chunks": [],
    }
    for ch in aag_chunks:
        meta = ch["metadata"]
        aag_asset["chunks"].append(
            {
                "chunk_id": meta.get("chunk_id"),
                "chunk_hash": meta.get("chunk_hash", ""),
                "content": ch["content"],
                "metadata": {
                    "video_id": meta.get("video_id"),
                    "video_title": meta.get("video_title"),
                    "video_url": meta.get("video_url"),
                    "speaker": meta.get("speaker"),
                    "language": meta.get("language"),
                    "summary": meta.get("summary"),
                    "start_second": meta.get("start_second"),
                    "end_second": meta.get("end_second"),
                    "version": meta.get("version", 1),
                    "is_latest": meta.get("is_latest", True),
                    "source": meta.get("source", ""),
                },
            }
        )
    exports.append(aag_asset)

    with open(output_path, "w", encoding="utf-8") as f:
        for asset in exports:
            jsonld = _convert_to_jsonld(asset, context=context)
            f.write(json.dumps(jsonld, ensure_ascii=False) + "\n")

    logger.info(
        "Exported %s aggregated assets (docs + aag) to %s",
        len(exports),
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Export static collections to DKG-ready JSONL.")
    parser.add_argument(
        "--output",
        type=str,
        default="dkg_docs_export.jsonl",
        help="Path to write JSONL (one asset per line).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["per_doc", "single"],
        default="per_doc",
        help="per_doc: one asset per document; single: two assets total (docs + aag).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="JSON-LD context URL (defaults to 'https://schema.org'). Can be a URL string.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Filter by source type (e.g., 'polka_wiki', 'polkassembly_doc'). If not specified, exports all sources.",
    )
    args = parser.parse_args()
    if args.mode == "per_doc":
        export_docs_per_doc(args.output, context=args.context, source_filter=args.source)
    else:
        export_single_assets(args.output, context=args.context)


if __name__ == "__main__":
    main()

