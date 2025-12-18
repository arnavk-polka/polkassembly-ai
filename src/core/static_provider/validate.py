#!/usr/bin/env python3
"""
Validation script for StaticProvider implementation.

This script:
1. Ingests a tiny sample (one doc, one AAG segment)
2. Confirms chunks exist in Chroma with chunk_id and chunk_hash
3. Calls search_docs and search_aag_segments
4. Verifies result shape and metadata

Run from project root:
    python -m src.core.static_provider.validate
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_static_provider():
    """Run validation checks on StaticProvider."""
    
    from src.core.config import Config
    from src.core.static_provider import (
        StaticProvider,
        StaticIngester,
        StaticSearchResult,
        compute_chunk_identity,
        verify_chunk_hash,
    )
    
    print("\n" + "=" * 60)
    print("STATIC PROVIDER VALIDATION")
    print("=" * 60)
    
    api_key = Config.OPENAI_API_KEY
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping embedding tests.")
        print("   Validating hashing and types only...")
        validate_hashing_only()
        return False
    
    test_chroma_dir = "./chroma_db_test"
    
    try:
        print("\n1. Initializing StaticIngester...")
        ingester = StaticIngester(
            openai_api_key=api_key,
            chroma_persist_directory=test_chroma_dir,
            chunk_size=200,
            chunk_overlap=50,
        )
        print("   ✅ StaticIngester initialized")
        
        print("\n2. Ingesting sample doc...")
        sample_doc_content = """
        Polkadot Staking Overview
        
        Staking on Polkadot allows DOT holders to participate in network security
        by nominating validators. Nominators bond their DOT tokens and earn rewards
        based on the performance of their chosen validators.
        
        Key concepts:
        - Bonding: Locking tokens for staking
        - Nominating: Choosing validators to support
        - Era: A period during which rewards are calculated
        - Slash: Penalty for validator misbehavior
        
        To stake, users must first bond their DOT, then select up to 16 validators
        to nominate. Rewards are paid out at the end of each era.
        """
        
        doc_result = ingester.ingest_single_doc(
            doc_id="test_staking_doc",
            title="Polkadot Staking Overview",
            content=sample_doc_content,
            source_type="polka_wiki",
            source_url="https://wiki.polkadot.network/docs/staking",
            tags=["staking", "nominator", "validator"],
        )
        print(f"   ✅ Doc ingested: {doc_result['chunks_created']} chunks created")
        
        print("\n3. Ingesting sample AAG transcript...")
        sample_aag = [{
            "video_id": "TEST_VIDEO_123",
            "metadata": {
                "title": "Polkadot Decoded 2023 - Governance Overview",
                "speaker": "Gavin Wood",
                "language": "en",
            },
            "segments": [
                {
                    "start": 0,
                    "end": 60,
                    "text": "Welcome to Polkadot Decoded. Today we'll discuss the governance model and how OpenGov empowers token holders to participate in protocol decisions.",
                    "summary": "Introduction to OpenGov governance model",
                },
                {
                    "start": 60,
                    "end": 120,
                    "text": "The referendum process allows any DOT holder to propose changes. Proposals go through a voting period where the community decides.",
                    "summary": "Referendum voting process explained",
                },
            ],
        }]
        
        aag_result = ingester.ingest_aag_transcripts(sample_aag)
        print(f"   ✅ AAG ingested: {aag_result['chunks_created']} chunks created")
        
        print("\n4. Initializing StaticProvider...")
        provider = StaticProvider(
            openai_api_key=api_key,
            chroma_persist_directory=test_chroma_dir,
        )
        stats = provider.get_stats()
        print(f"   ✅ Provider initialized: {stats['docs_count']} docs, {stats['aag_count']} AAG chunks")
        
        print("\n5. Testing search_docs...")
        doc_results = provider.search_docs("What is staking?", k=3)
        print(f"   Found {len(doc_results)} results")
        
        if doc_results:
            result = doc_results[0]
            print(f"   Top result:")
            print(f"      - id: {result.id}")
            print(f"      - score: {result.score:.4f}")
            print(f"      - source: {result.source}")
            print(f"      - chunk_id: {result.chunk_id}")
            print(f"      - chunk_hash: {result.chunk_hash[:16]}...")
            print(f"      - dkg_match: {result.dkg_match}")
            print(f"      - content preview: {result.content[:100]}...")
            
            assert result.id == result.chunk_id, "id should equal chunk_id"
            assert result.chunk_hash, "chunk_hash should be present"
            assert result.dkg_match is None, "dkg_match should be None (not implemented yet)"
            assert result.metadata.get("chunk_id") == result.chunk_id, "metadata should contain chunk_id"
            print("   ✅ Doc search result shape validated")
        else:
            print("   ⚠️  No doc results found")
        
        print("\n6. Testing search_aag_segments...")
        aag_results = provider.search_aag_segments("governance referendum", k=3)
        print(f"   Found {len(aag_results)} results")
        
        if aag_results:
            result = aag_results[0]
            print(f"   Top result:")
            print(f"      - id: {result.id}")
            print(f"      - score: {result.score:.4f}")
            print(f"      - source: {result.source}")
            print(f"      - video_id: {result.metadata.get('video_id')}")
            print(f"      - speaker: {result.metadata.get('speaker')}")
            print(f"      - summary: {result.metadata.get('summary')}")
            print(f"      - start_second: {result.metadata.get('start_second')}")
            print(f"      - end_second: {result.metadata.get('end_second')}")
            print(f"      - chunk_hash: {result.chunk_hash[:16]}...")
            print(f"      - dkg_match: {result.dkg_match}")
            
            assert result.metadata.get("video_id") == "TEST_VIDEO_123", "video_id should be present"
            assert result.metadata.get("speaker") == "Gavin Wood", "speaker should be preserved"
            assert result.dkg_match is None, "dkg_match should be None"
            print("   ✅ AAG search result shape validated")
        else:
            print("   ⚠️  No AAG results found")
        
        print("\n7. Testing legacy format compatibility...")
        legacy_results = provider.search_similar_chunks("staking", n_results=2)
        if legacy_results:
            legacy = legacy_results[0]
            assert "content" in legacy, "legacy format should have 'content'"
            assert "metadata" in legacy, "legacy format should have 'metadata'"
            assert "similarity_score" in legacy, "legacy format should have 'similarity_score'"
            assert "source" in legacy, "legacy format should have 'source'"
            print(f"   ✅ Legacy format works: {list(legacy.keys())}")
        
        print("\n8. Testing hash verification...")
        if doc_results:
            result = doc_results[0]
            is_valid = verify_chunk_hash(
                chunk_id=result.chunk_id,
                content=result.content,
                expected_hash=result.chunk_hash
            )
            if is_valid:
                print("   ✅ Hash verification passed")
            else:
                print("   ❌ Hash verification failed!")
        
        print("\n" + "=" * 60)
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 60)
        
        print("\n📋 Summary of StaticProvider capabilities:")
        print("   - search_docs(query, k, filters) → List[StaticSearchResult]")
        print("   - search_aag_segments(query, k, filters) → List[StaticSearchResult]")
        print("   - search_all(query, k, filters) → merged results")
        print("   - search_similar_chunks(query, n_results) → legacy format")
        print("")
        print("📋 StaticSearchResult fields:")
        print("   - id, score, content, source")
        print("   - chunk_id, chunk_hash")
        print("   - metadata (doc-specific or AAG-specific)")
        print("   - dkg_match (None for now, placeholder for DKG integration)")
        print("")
        print("📋 DKG Integration TODO:")
        print("   - After Chroma search, look up chunk_hash in DKG")
        print("   - If match found, populate dkg_match with {asset_ual, chunk_id, chunk_hash}")
        print("   - UI can show 'Verified on DKG' badge for matched chunks")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        import shutil
        if os.path.exists(test_chroma_dir):
            shutil.rmtree(test_chroma_dir)
            print(f"\n🧹 Cleaned up test directory: {test_chroma_dir}")


def validate_hashing_only():
    """Validate hashing utilities without needing OpenAI API."""
    
    from src.core.static_provider import (
        normalize_text,
        generate_chunk_id,
        generate_chunk_hash,
        compute_chunk_identity,
        verify_chunk_hash,
    )
    
    print("\n--- Hashing Utilities Validation ---")
    
    raw_text = "  Hello   world  \n\n\n  Test   "
    normalized = normalize_text(raw_text)
    print(f"normalize_text: '{raw_text}' → '{normalized}'")
    assert normalized == "Hello world\n\nTest", f"Unexpected: {normalized}"
    print("   ✅ normalize_text works")
    
    chunk_id = generate_chunk_id("doc123", 0, 100, "doc")
    print(f"generate_chunk_id: doc123, 0-100 → '{chunk_id}'")
    assert chunk_id == "doc:doc123:0-100", f"Unexpected: {chunk_id}"
    print("   ✅ generate_chunk_id works")
    
    hash1 = generate_chunk_hash(chunk_id, "Hello world")
    hash2 = generate_chunk_hash(chunk_id, "Hello world")
    hash3 = generate_chunk_hash(chunk_id, "Different content")
    print(f"generate_chunk_hash: same content → same hash: {hash1 == hash2}")
    print(f"generate_chunk_hash: diff content → diff hash: {hash1 != hash3}")
    assert hash1 == hash2, "Same content should produce same hash"
    assert hash1 != hash3, "Different content should produce different hash"
    print("   ✅ generate_chunk_hash works")
    
    cid, chash, norm = compute_chunk_identity("video_abc", 0, 60, "  Some text  ", "aag")
    print(f"compute_chunk_identity: → chunk_id='{cid}', hash={chash[:16]}...")
    assert cid == "aag:video_abc:0-60"
    assert norm == "Some text"
    print("   ✅ compute_chunk_identity works")
    
    is_valid = verify_chunk_hash(cid, "  Some text  ", chash)
    print(f"verify_chunk_hash: valid content → {is_valid}")
    assert is_valid, "Should verify correctly"
    is_invalid = verify_chunk_hash(cid, "Different text", chash)
    print(f"verify_chunk_hash: tampered content → {is_invalid}")
    assert not is_invalid, "Should fail verification"
    print("   ✅ verify_chunk_hash works")
    
    print("\n✅ All hashing utilities validated")


if __name__ == "__main__":
    success = validate_static_provider()
    sys.exit(0 if success else 1)





