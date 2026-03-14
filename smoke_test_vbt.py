"""
VBT smoke test for chunk-embed: cross-lingual semantic search with Vijñāna Bhairava Tantra.

Exercises the full pipeline with real Sanskrit/English tantric corpus data:
  ingest Sanskrit verses → ingest English translations → cross-lingual retrieval
  → thematic ranking → similarity ordering → filter verification

Data sourced from ~/Projects/sa-embedding/vbt_corpus.py (168 VBT verses).
A curated subset of ~15 verses spanning six practice domains is used here.

Requires:
    - Local BGE-M3 model at ./local_bge_m3/BAAI/bge-m3
    - PostgreSQL with pgvector: createdb chunk_embed_test && psql chunk_embed_test -c "CREATE EXTENSION vector;"

Usage:
    cd ~/Projects/chunk-embed
    uv run python smoke_test_vbt.py
"""

import json
import sys

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from chunk_embed.parse import parse_chunks
from chunk_embed.embed import BgeM3Embedder, embed_chunks
from chunk_embed.store import ensure_schema, upsert_document, insert_chunks, search_chunks

DB_URL = "postgresql://localhost/chunk_embed_test"

# --- Curated VBT subset: 15 verses across 6 practice domains ---
# Index references are to the full VBT_CORPUS in sa-embedding/vbt_corpus.py

VERSES = {
    # Dialogue marker (negative control)
    "dialogue": {
        "sa": "भैरव उवाच ।",
        "en": "Bhairava said:",
        "idx": 7,
    },
    # Breath practices (prāṇa)
    "breath_ascending": {
        "sa": "ऊर्ध्वे प्राणो ह्य् अधो जीवो विसर्गात्मा परोच्चरेत् / उत्पत्तिद्वितयस्थाने भरणाद् भरिता स्थितिः",
        "en": "Upwards moves the breath, downwards the living soul; their movement brings forth creation and dissolution. In these two places of origin, sustenance arises from their filling, which is existence.",
        "idx": 25,
    },
    "breath_middle": {
        "sa": "मरुतो ऽन्तर् बहिर् वापि वियद्युग्मानिवर्तनात् / भैरवी भैरवस्येत्थं भैरवो व्यज्यते वपुः",
        "en": "When the power that is of the nature of air neither moves outward nor inward, but remains expanded; in that state of perfect non-duality in the center, there arises the true form of Bhairava.",
        "idx": 27,
    },
    # Sound practices (nāda)
    "sound_anahata": {
        "sa": "अनाहते पात्रकर्णे ऽभग्नशब्दे सरिद्द्रुते / शब्दब्रह्मणि निष्णातः परं ब्रह्माधिगच्छति",
        "en": "One who is immersed in the soundless sound, who hears the unbroken current like a flowing river within the inner ear, who is steadfast in the sound-brahman, attains the supreme Brahman.",
        "idx": 39,
    },
    "sound_instruments": {
        "sa": "तन्त्र्यादिवाद्यशब्देषु दीर्घेषु क्रमसंस्थितेः / अनन्यचेताः प्रत्यन्ते परव्योमवपुर् भवेत्",
        "en": "When, amidst the sustained and successive sounds of stringed and other musical instruments, one's mind remains unwaveringly focused upon the end of each sound, one attains the infinite expanse of the supreme ether.",
        "idx": 42,
    },
    # Void/space practices (śūnyatā)
    "void_body_sky": {
        "sa": "तनूदेशे शून्यतैव क्षणमात्रं विभावयेत् / निर्विकल्पं निर्विकल्पो निर्विकल्पस्वरूपभाक्",
        "en": "For a moment, let one contemplate emptiness in the body's domain; Being thought-free, one becomes thought-free and attains the very essence of thought-free being.",
        "idx": 47,
    },
    "void_triple": {
        "sa": "पृष्ठशून्यं मूलशून्यं हृच्छून्यं भावयेत् स्थिरम् / युगपन् निर्विकल्पत्वान् निर्विकल्पोदयस् ततः",
        "en": "Contemplate firmly on the state where the back is void, the root is void, and the heart is void; At once, due to the absence of alternatives, arises the dawn of non-duality.",
        "idx": 46,
    },
    # Bliss practices (ānanda)
    "bliss_union": {
        "sa": "शक्तिसंगमसंक्षुब्धशक्त्यावेशावसानिकम् / यत् सुखं ब्रह्मतत्त्वस्य तत् सुखं स्वाक्यम् उच्यते",
        "en": "The joy that arises from the union and culmination of energies, from the stirring and ultimate immersion in power—that is the bliss of the Brahman principle; such bliss is called one's own true bliss.",
        "idx": 70,
    },
    "bliss_meeting": {
        "sa": "लेहनामन्थनकोटैः स्त्रीसुखस्य भरात् स्मृतेः / शक्त्यभावे ऽपि देवेशि भावद् एवं महासुखम्",
        "en": "When great joy is attained, or a beloved friend is seen after a long time, contemplate the surge of bliss that arises, merge into it, and let your mind become one with that.",
        "idx": 73,
    },
    # Gaze practices (dṛṣṭi)
    "gaze_sky": {
        "sa": "निरालम्बं सथिरं शून्यम् अम्बरं यावद् आश्रयेत् / सम्पत्स्यते दृशम् तत्र मेरुपृष्ठाव्लम्बनम्",
        "en": "Gazing unceasingly at the clear sky, with the mind stilled, O Goddess, in that very moment one attains the form of Bhairava.",
        "idx": 86,
    },
    # Mind practices (manas)
    "mind_unsupported": {
        "sa": "निराधारं मनः कृत्वा विकल्पान् न विकल्पयेत् / तदात्मपरमात्मत्वे भैरवो मृगलोचने",
        "en": "Having made the mind unsupported, let one not entertain any thoughts; in that identity of self and supreme self, O gazelle-eyed one, is the state of Bhairava.",
        "idx": 110,
    },
    # Non-dual realization (advaita)
    "nondual_i_am_shiva": {
        "sa": "सर्वज्ञः सर्वकर्ता च व्यापकः परमेश्वरः / स एवाहं शैवधर्मा इति दार्ढ्याच् छिवो भवेत्",
        "en": "He who knows all, does all, all-pervading, the Supreme Lord—He indeed am I, steadfast in Shaiva dharma; thus, by firmness, one becomes Shiva.",
        "idx": 111,
    },
    "nondual_equal": {
        "sa": "समः शत्रौ च मित्रे च समो मानावमानयोः / ब्रह्मणः परिपूर्णत्वाद् इति ज्ञात्वा सुखी भवेत्",
        "en": "He who is equal toward both enemy and friend, who remains the same in honor and dishonor—knowing that the Self is complete like Brahman—such a person becomes truly happy.",
        "idx": 127,
    },
    # Worship redefinition
    "worship_true": {
        "sa": "न वह्निर् न च तत्रोर्मिर् न स्वरूपं मरीचिषु / शैवस्यैव स्वरूपं तद् विमलं विश्वपूरणम्",
        "en": "Worship is not with flowers and the like; it is the firm intent made upon the attributeless vast ether of consciousness. That alone is worship—when, with devotion, one's mind dissolves therein.",
        "idx": 153,
    },
    # Bliss of vikalpa-freedom (cross-lingual anchor)
    "bliss_vikalpa_free": {
        "sa": "अन्तःस्वानुभवानन्दा विकल्पोन्मुक्तगोचरा / यावस्था भरिताकारा भैरवी भैरवात्मनः",
        "en": "Internal self direct experiential bliss, free from the conceptualization domain; that state full of bhairavī forms. [This is] the nature of bhairava.",
        "idx": 16,
    },
}

# Build text-chunker JSON documents from the curated subset

def _build_doc(lang: str) -> dict:
    """Build a text-chunker-style JSON doc from VERSES."""
    # Group verses by thematic domain for heading_context
    domain_labels = {
        "dialogue": "Dialogue",
        "breath_ascending": "Prāṇa Practices",
        "breath_middle": "Prāṇa Practices",
        "sound_anahata": "Nāda Practices",
        "sound_instruments": "Nāda Practices",
        "void_body_sky": "Śūnyatā Practices",
        "void_triple": "Śūnyatā Practices",
        "bliss_union": "Ānanda Practices",
        "bliss_meeting": "Ānanda Practices",
        "bliss_vikalpa_free": "Ānanda Practices",
        "gaze_sky": "Dṛṣṭi Practices",
        "mind_unsupported": "Manas Practices",
        "nondual_i_am_shiva": "Advaita Realization",
        "nondual_equal": "Advaita Realization",
        "worship_true": "Ritual Redefinition",
    }

    text_key = "sa" if lang == "sanskrit" else "en"
    chunks = []
    for i, (key, v) in enumerate(VERSES.items()):
        chunks.append({
            "text": v[text_key],
            "chunk_type": "paragraph",
            "heading_context": ["Vijñāna Bhairava Tantra", domain_labels[key]],
            "heading_level": None,
            "page_number": None,
            "source_line_start": i * 2 + 1,
            "source_line_end": i * 2 + 2,
        })

    return {
        "total_chunks": len(chunks),
        "mode": "document",
        "chunks": chunks,
    }


SANSKRIT_DOC = _build_doc("sanskrit")
ENGLISH_DOC = _build_doc("english")

# Verse keys in insertion order (for index mapping)
VERSE_KEYS = list(VERSES.keys())


def run_vbt_smoke_test():
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: {detail}")
            failed += 1

    # --- 1. Load model ---
    print("Loading BGE-M3 model...")
    embedder = BgeM3Embedder()
    check("Model loaded", embedder.dimension == 1024)

    # --- 2. Embed both documents ---
    docs = [
        ("sanskrit", SANSKRIT_DOC, "/test/vbt_sanskrit.md"),
        ("english", ENGLISH_DOC, "/test/vbt_english.md"),
    ]

    all_results = {}
    for label, doc, source in docs:
        print(f"\n--- {label} document ({doc['total_chunks']} chunks) ---")
        raw = json.dumps(doc)
        parsed = parse_chunks(raw)
        check(f"Parse {label}", parsed.total_chunks == doc["total_chunks"])

        embeddings = embed_chunks(parsed.chunks, embedder, batch_size=32)
        check(f"Embed {label} count", len(embeddings) == len(parsed.chunks))
        norms = [float(np.linalg.norm(v)) for v in embeddings]
        check(f"Embed {label} normalized", all(abs(n - 1.0) < 1e-4 for n in norms),
              f"norms min={min(norms):.4f} max={max(norms):.4f}")

        all_results[label] = (parsed, embeddings, source)

    # --- 3. Cross-lingual similarity sanity checks ---
    print("\n--- Cross-lingual similarity (embedding-level) ---")
    sa_embs = all_results["sanskrit"][1]
    en_embs = all_results["english"][1]

    # Same verse in Sanskrit and English should be more similar
    # than two unrelated verses
    breath_idx = VERSE_KEYS.index("breath_ascending")
    bliss_idx = VERSE_KEYS.index("bliss_union")
    dialogue_idx = VERSE_KEYS.index("dialogue")

    sim_breath_sa_en = float(np.dot(sa_embs[breath_idx], en_embs[breath_idx]))
    sim_breath_sa_bliss_en = float(np.dot(sa_embs[breath_idx], en_embs[bliss_idx]))
    check("Cross-lingual: breath(sa) ↔ breath(en) > breath(sa) ↔ bliss(en)",
          sim_breath_sa_en > sim_breath_sa_bliss_en,
          f"same={sim_breath_sa_en:.3f}, cross={sim_breath_sa_bliss_en:.3f}")
    print(f"    breath(sa) ↔ breath(en):  {sim_breath_sa_en:.4f}")
    print(f"    breath(sa) ↔ bliss(en):   {sim_breath_sa_bliss_en:.4f}")

    # Thematically related verses (same domain) should be more similar
    # than unrelated verses (different domain)
    breath_mid_idx = VERSE_KEYS.index("breath_middle")
    sim_breath_pair = float(np.dot(sa_embs[breath_idx], sa_embs[breath_mid_idx]))
    sim_breath_dialogue = float(np.dot(sa_embs[breath_idx], sa_embs[dialogue_idx]))
    check("Thematic: breath↔breath > breath↔dialogue (Sanskrit)",
          sim_breath_pair > sim_breath_dialogue,
          f"related={sim_breath_pair:.3f}, unrelated={sim_breath_dialogue:.3f}")

    void_body_idx = VERSE_KEYS.index("void_body_sky")
    void_triple_idx = VERSE_KEYS.index("void_triple")
    sim_void_pair = float(np.dot(sa_embs[void_body_idx], sa_embs[void_triple_idx]))
    sim_void_bliss = float(np.dot(sa_embs[void_body_idx], sa_embs[bliss_idx]))
    check("Thematic: void↔void > void↔bliss (Sanskrit)",
          sim_void_pair > sim_void_bliss,
          f"related={sim_void_pair:.3f}, unrelated={sim_void_bliss:.3f}")

    # --- 4. Store and search ---
    print("\n--- Database: ingest + semantic search ---")
    with psycopg.connect(DB_URL, autocommit=False) as conn:
        register_vector(conn)
        ensure_schema(conn)

        for label, (parsed, embeddings, source) in all_results.items():
            doc_id = upsert_document(conn, source, parsed.mode, parsed.total_chunks)
            insert_chunks(conn, doc_id, parsed.chunks, embeddings)

        doc_count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        check("Document count", doc_count == 2, f"got {doc_count}")
        check("Chunk count", chunk_count == len(VERSES) * 2,
              f"got {chunk_count}, expected {len(VERSES) * 2}")

        # Unicode round-trip: verify Devanagari survives storage
        row = conn.execute(
            "SELECT text FROM chunks WHERE text LIKE '%भैरव उवाच%'"
        ).fetchone()
        check("Devanagari round-trip", row is not None and "भैरव उवाच" in row[0])

        # --- 5. Cross-lingual retrieval via search_chunks ---
        print("\n--- Cross-lingual retrieval ---")

        # English query → Sanskrit verse should rank high
        query_breath = "breath ascending descending prāṇa life force"
        q_emb = embedder.embed([query_breath])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:3]]
        breath_sa_text = VERSES["breath_ascending"]["sa"]
        check("Retrieval: 'prāṇa breath' → breath verse in top 3",
              any(breath_sa_text in t for t in top_texts),
              f"top 3 texts: {[t[:40] for t in top_texts]}")

        query_sound = "unstruck sound flowing river inner ear nāda"
        q_emb = embedder.embed([query_sound])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:3]]
        sound_sa_text = VERSES["sound_anahata"]["sa"]
        check("Retrieval: 'anāhata nāda' → sound verse in top 3",
              any(sound_sa_text in t for t in top_texts),
              f"top 3 texts: {[t[:40] for t in top_texts]}")

        query_nondual = "I am Shiva all-pervading supreme lord"
        q_emb = embedder.embed([query_nondual])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:3]]
        nondual_sa_text = VERSES["nondual_i_am_shiva"]["sa"]
        check("Retrieval: 'I am Śiva' → nondual verse in top 3",
              any(nondual_sa_text in t for t in top_texts),
              f"top 3 texts: {[t[:40] for t in top_texts]}")

        query_void = "body emptiness void thought-free śūnyatā"
        q_emb = embedder.embed([query_void])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:5]]
        void_sa_text_1 = VERSES["void_body_sky"]["sa"]
        void_sa_text_2 = VERSES["void_triple"]["sa"]
        check("Retrieval: 'śūnyatā void' → void verse in top 5",
              any(void_sa_text_1 in t or void_sa_text_2 in t for t in top_texts),
              f"top 5 texts: {[t[:40] for t in top_texts]}")

        # --- 6. Sanskrit query → English translation retrieval ---
        print("\n--- Sanskrit query → English retrieval ---")

        # Sanskrit keyword query → English translations should rank high
        query_sa_breath = "प्राण जीव उत्पत्ति भरण"
        q_emb = embedder.embed([query_sa_breath])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_english.md")
        top_texts = [r.text for r in results[:3]]
        breath_en_text = VERSES["breath_ascending"]["en"]
        check("Sa→En retrieval: 'प्राण जीव' → breath translation in top 3",
              any(breath_en_text in t for t in top_texts),
              f"top 3 texts: {[t[:50] for t in top_texts]}")

        query_sa_sound = "अनाहत शब्द नाद ब्रह्म"
        q_emb = embedder.embed([query_sa_sound])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_english.md")
        top_texts = [r.text for r in results[:5]]
        sound_en_text = VERSES["sound_anahata"]["en"]
        sound_en_text_2 = VERSES["sound_instruments"]["en"]
        check("Sa→En retrieval: 'अनाहत शब्द' → sound translation in top 5",
              any(sound_en_text in t or sound_en_text_2 in t for t in top_texts),
              f"top 5 texts: {[t[:50] for t in top_texts]}")

        query_sa_void = "शून्य निर्विकल्प देह"
        q_emb = embedder.embed([query_sa_void])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_english.md")
        top_texts = [r.text for r in results[:5]]
        void_en_1 = VERSES["void_body_sky"]["en"]
        void_en_2 = VERSES["void_triple"]["en"]
        check("Sa→En retrieval: 'शून्य निर्विकल्प' → void translation in top 5",
              any(void_en_1 in t or void_en_2 in t for t in top_texts),
              f"top 5 texts: {[t[:50] for t in top_texts]}")

        # Full Sanskrit verse as query → its own English translation should rank #1
        q_emb = embedder.embed([VERSES["nondual_i_am_shiva"]["sa"]])[0]
        results = search_chunks(conn, q_emb, top_k=3, source_path="/test/vbt_english.md")
        check("Sa→En retrieval: full nondual verse → own translation is #1",
              results[0].text == VERSES["nondual_i_am_shiva"]["en"],
              f"got: {results[0].text[:60]}")

        q_emb = embedder.embed([VERSES["bliss_union"]["sa"]])[0]
        results = search_chunks(conn, q_emb, top_k=3, source_path="/test/vbt_english.md")
        check("Sa→En retrieval: full bliss verse → own translation is #1",
              results[0].text == VERSES["bliss_union"]["en"],
              f"got: {results[0].text[:60]}")

        # --- 7. Sanskrit query → Sanskrit (monolingual thematic retrieval) ---
        print("\n--- Sanskrit → Sanskrit retrieval ---")

        # Sanskrit keyword query → same-domain Sanskrit verses should rank high
        query_sa_prana = "प्राण जीव श्वास मरुत"
        q_emb = embedder.embed([query_sa_prana])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:3]]
        check("Sa→Sa: 'प्राण जीव श्वास मरुत' → breath verses in top 3",
              any(VERSES["breath_ascending"]["sa"] in t or VERSES["breath_middle"]["sa"] in t
                  for t in top_texts),
              f"top 3: {[t[:40] for t in top_texts]}")

        query_sa_nada = "शब्द नाद अनाहत तन्त्री"
        q_emb = embedder.embed([query_sa_nada])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:3]]
        check("Sa→Sa: 'शब्द नाद अनाहत तन्त्री' → sound verses in top 3",
              any(VERSES["sound_anahata"]["sa"] in t or VERSES["sound_instruments"]["sa"] in t
                  for t in top_texts),
              f"top 3: {[t[:40] for t in top_texts]}")

        query_sa_sunya = "शून्य निर्विकल्प पृष्ठ मूल हृदय"
        q_emb = embedder.embed([query_sa_sunya])[0]
        results = search_chunks(conn, q_emb, top_k=5, source_path="/test/vbt_sanskrit.md")
        top_texts = [r.text for r in results[:3]]
        check("Sa→Sa: 'शून्य निर्विकल्प' → void verses in top 3",
              any(VERSES["void_body_sky"]["sa"] in t or VERSES["void_triple"]["sa"] in t
                  for t in top_texts),
              f"top 3: {[t[:40] for t in top_texts]}")

        # Full verse as query → same-domain verse should outrank dialogue marker
        # breath_ascending → breath_middle should rank above "भैरव उवाच"
        q_emb = embedder.embed([VERSES["breath_ascending"]["sa"]])[0]
        results = search_chunks(conn, q_emb, top_k=15, source_path="/test/vbt_sanskrit.md")
        non_self = [r for r in results if r.text != VERSES["breath_ascending"]["sa"]]
        non_self_texts = [r.text for r in non_self]
        breath_mid_rank = next((i for i, t in enumerate(non_self_texts) if t == VERSES["breath_middle"]["sa"]), None)
        dialogue_rank = next((i for i, t in enumerate(non_self_texts) if t == VERSES["dialogue"]["sa"]), len(non_self_texts))
        check("Sa→Sa: breath verse → breath_middle ranks above dialogue",
              breath_mid_rank is not None and breath_mid_rank < dialogue_rank,
              f"breath_middle rank={breath_mid_rank}, dialogue rank={dialogue_rank}")

        # void_triple → void_body_sky should rank above dialogue
        q_emb = embedder.embed([VERSES["void_triple"]["sa"]])[0]
        results = search_chunks(conn, q_emb, top_k=15, source_path="/test/vbt_sanskrit.md")
        non_self = [r for r in results if r.text != VERSES["void_triple"]["sa"]]
        non_self_texts = [r.text for r in non_self]
        void_body_rank = next((i for i, t in enumerate(non_self_texts) if t == VERSES["void_body_sky"]["sa"]), None)
        dialogue_rank = next((i for i, t in enumerate(non_self_texts) if t == VERSES["dialogue"]["sa"]), len(non_self_texts))
        check("Sa→Sa: void_triple → void_body_sky ranks above dialogue",
              void_body_rank is not None and void_body_rank < dialogue_rank,
              f"void_body rank={void_body_rank}, dialogue rank={dialogue_rank}")

        # Dialogue marker should NOT be top non-self for any thematic query
        q_emb = embedder.embed([VERSES["bliss_union"]["sa"]])[0]
        results = search_chunks(conn, q_emb, top_k=15, source_path="/test/vbt_sanskrit.md")
        non_self = [r for r in results if r.text != VERSES["bliss_union"]["sa"]]
        check("Sa→Sa: bliss verse → dialogue is NOT top non-self",
              non_self[0].text != VERSES["dialogue"]["sa"],
              f"top non-self: {non_self[0].text[:40]}")

        # --- 8. Cross-document retrieval ---
        print("\n--- Cross-document retrieval ---")

        # Searching across both docs: same query should find both Sanskrit and English
        q_emb = embedder.embed(["worship not with flowers consciousness"])[0]
        results = search_chunks(conn, q_emb, top_k=10)
        sources = {r.source_path for r in results}
        check("Cross-doc: results from both Sanskrit and English docs",
              "/test/vbt_sanskrit.md" in sources and "/test/vbt_english.md" in sources,
              f"sources found: {sources}")

        # --- 9. Search filters ---
        print("\n--- Search filters ---")

        q_emb = embedder.embed(["meditation practice"])[0]

        # Source filter
        sa_only = search_chunks(conn, q_emb, source_path="/test/vbt_sanskrit.md")
        check("Source filter: Sanskrit only",
              all(r.source_path == "/test/vbt_sanskrit.md" for r in sa_only))

        en_only = search_chunks(conn, q_emb, source_path="/test/vbt_english.md")
        check("Source filter: English only",
              all(r.source_path == "/test/vbt_english.md" for r in en_only))

        # Threshold filter
        high_thresh = search_chunks(conn, q_emb, threshold=0.8)
        if high_thresh:
            check("Threshold filter: all results >= 0.8",
                  all(r.similarity >= 0.8 for r in high_thresh))
        else:
            check("Threshold filter: high threshold returns fewer results", True)

        low_thresh = search_chunks(conn, q_emb, threshold=0.0)
        check("Threshold filter: low threshold returns more results",
              len(low_thresh) >= len(high_thresh))

        # Top-k
        top3 = search_chunks(conn, q_emb, top_k=3)
        check("Top-k filter: top_k=3 returns at most 3", len(top3) <= 3)

        # --- 10. Similarity ordering in search results ---
        print("\n--- Similarity ordering via search ---")

        # Query with a breath verse embedding → breath verses should rank above dialogue
        breath_emb = sa_embs[breath_idx]
        results = search_chunks(conn, breath_emb, top_k=15, source_path="/test/vbt_sanskrit.md")

        breath_texts = {VERSES["breath_ascending"]["sa"], VERSES["breath_middle"]["sa"]}
        dialogue_text = VERSES["dialogue"]["sa"]

        breath_ranks = [i for i, r in enumerate(results) if r.text in breath_texts]
        dialogue_ranks = [i for i, r in enumerate(results) if r.text == dialogue_text]

        if breath_ranks and dialogue_ranks:
            check("Ranking: breath verses rank above dialogue marker",
                  min(breath_ranks) < min(dialogue_ranks),
                  f"breath ranks={breath_ranks}, dialogue ranks={dialogue_ranks}")
        else:
            check("Ranking: both breath and dialogue found in results",
                  False, f"breath_ranks={breath_ranks}, dialogue_ranks={dialogue_ranks}")

        # Rollback — don't leave test data
        conn.rollback()

    # --- Summary ---
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"VBT smoke test: {passed}/{total} checks passed")
    if failed:
        print(f"FAILED: {failed} check(s)")
        sys.exit(1)
    else:
        print("All checks passed!")


if __name__ == "__main__":
    run_vbt_smoke_test()
