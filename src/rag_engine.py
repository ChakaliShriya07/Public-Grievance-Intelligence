import os
import re

import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "grievance_chunks.csv"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "grievance_embeddings.npy"
)

FAISS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "grievance_faiss.index"
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# LOAD HISTORICAL RESOURCES
# ============================================================

print(
    "Loading historical grievance resources..."
)

chunks_df = pd.read_csv(
    CHUNKS_PATH
)


embeddings = np.load(
    EMBEDDINGS_PATH,
    mmap_mode="r"
)


index = faiss.read_index(
    FAISS_INDEX_PATH
)


# ============================================================
# ALIGN CHUNKS WITH EMBEDDINGS
# ============================================================

embedding_chunks_df = chunks_df[
    chunks_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
    != ""
].reset_index(drop=True)


if len(embedding_chunks_df) != index.ntotal:

    raise ValueError(
        f"Alignment error: "
        f"CSV rows={len(embedding_chunks_df)}, "
        f"FAISS vectors={index.ntotal}"
    )


if len(embeddings) != index.ntotal:

    raise ValueError(
        "Embedding/index mismatch: "
        f"embeddings={len(embeddings)}, "
        f"index={index.ntotal}"
    )


print(
    f"Historical resources loaded: "
    f"{index.ntotal:,} vectors"
)


# ============================================================
# LOAD MODELS
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


# ============================================================
# LOW-INFORMATION HISTORICAL COMPLAINT DETECTION
# ============================================================

LOW_INFORMATION_PHRASES = [
    "no answer",
    "no response",
    "not specific",
    "not clear",
    "matter is not specific",
    "matter not specific",
    "incoherent",
    "grievance is incoherent",
    "provide detailed information",
    "provide a detailed information",
    "provide more details",
    "provide details of the grievance",
]


def is_low_information_grievance(text):
    """
    Detects historical records that contain little or
    no useful grievance information.

    These records should not dominate RAG evidence.
    """

    if not text:

        return True

    text = str(
        text
    ).strip().lower()

    if not text:

        return True


    # --------------------------------------------------------
    # Exact / near-exact low-information complaints
    # --------------------------------------------------------

    normalized = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    if normalized in {
        "no answer",
        "no response",
        "no information",
        "not specific",
        "not clear",
        "nil",
    }:

        return True


    # --------------------------------------------------------
    # Repeated vague-response phrases
    # --------------------------------------------------------

    matches = 0

    for phrase in LOW_INFORMATION_PHRASES:

        if phrase in normalized:

            matches += 1


    # If the record contains several vague-response
    # indicators, treat it as low-information.
    if matches >= 2:

        return True


    # --------------------------------------------------------
    # Very short complaints
    # --------------------------------------------------------

    words = normalized.split()

    if len(words) <= 3:

        return True


    return False


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_query_keywords(query):
    """
    Extracts meaningful issue keywords from a query.

    This provides a lightweight lexical signal in addition
    to FAISS semantic similarity.
    """

    query = str(
        query
    ).lower()


    # --------------------------------------------------------
    # Important grievance-domain terms
    # --------------------------------------------------------

    important_terms = [
        "water",
        "supply",
        "drinking",
        "shortage",
        "availability",
        "irregular",
        "pressure",
        "tap",
        "pipeline",
        "leakage",
        "electricity",
        "power",
        "road",
        "roads",
        "pothole",
        "drainage",
        "sewerage",
        "garbage",
        "waste",
        "sanitation",
        "hospital",
        "health",
        "school",
        "education",
        "pension",
        "ration",
        "corruption",
        "police",
        "transport",
        "railway",
        "bank",
        "salary",
        "employment",
        "land",
        "property",
        "certificate",
        "document",
        "passport",
        "tax",
    ]


    found = []

    for term in important_terms:

        if term in query:

            found.append(
                term
            )


    return found


# ============================================================
# LEXICAL RELEVANCE SCORE
# ============================================================

def calculate_keyword_score(
    query,
    text
):
    """
    Calculates a simple lexical relevance score.

    This does NOT replace semantic retrieval.

    It only gives a small bonus to historical records
    that actually contain the important issue terms.
    """

    if not text:

        return 0.0


    keywords = extract_query_keywords(
        query
    )


    if not keywords:

        return 0.0


    text_lower = str(
        text
    ).lower()


    matched = 0


    for keyword in keywords:

        if keyword in text_lower:

            matched += 1


    return (
        matched /
        len(keywords)
    )


# ============================================================
# RETRIEVE HISTORICAL GRIEVANCES
# ============================================================

def retrieve_grievances(
    query,
    top_k=5,
    candidate_k=100
):
    """
    Retrieve historically relevant grievances.

    Pipeline:

        Query
          ↓
        SentenceTransformer
          ↓
        FAISS semantic retrieval
          ↓
        Candidate filtering
          ↓
        Cross-Encoder reranking
          ↓
        Lexical relevance adjustment
          ↓
        Low-information filtering
          ↓
        Duplicate grievance removal
          ↓
        Final evidence
    """

    if not query or not str(query).strip():

        return pd.DataFrame()


    query = str(
        query
    ).strip()


    # ========================================================
    # STEP 1 — QUERY EMBEDDING
    # ========================================================

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(
        "float32"
    )


    # ========================================================
    # STEP 2 — FAISS CANDIDATE RETRIEVAL
    # ========================================================

    search_k = min(
        candidate_k,
        index.ntotal
    )


    scores, indices = index.search(
        query_embedding,
        search_k
    )


    candidate_rows = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue


        row = (
            embedding_chunks_df
            .iloc[int(idx)]
            .copy()
        )


        text = str(
            row.get(
                "text",
                ""
            )
        ).strip()


        if not text:

            continue


        row["faiss_score"] = float(
            score
        )


        candidate_rows.append(
            row
        )


    if not candidate_rows:

        return pd.DataFrame()


    candidates_df = pd.DataFrame(
        candidate_rows
    ).reset_index(
        drop=True
    )


    # ========================================================
    # STEP 3 — REMOVE OBVIOUS LOW-INFORMATION RECORDS
    # ========================================================

    candidates_df[
        "_low_information"
    ] = candidates_df[
        "text"
    ].apply(
        is_low_information_grievance
    )


    # We do not immediately delete every vague record because
    # the semantic search may occasionally find useful context.
    # Instead, they receive a very large ranking penalty later.


    # ========================================================
    # STEP 4 — CROSS-ENCODER RERANKING
    # ========================================================

    pairs = []


    for _, row in candidates_df.iterrows():

        pairs.append(
            (
                query,
                str(
                    row.get(
                        "text",
                        ""
                    )
                )
            )
        )


    reranker_scores = reranker.predict(
        pairs
    )


    candidates_df[
        "reranker_score"
    ] = np.asarray(
        reranker_scores,
        dtype="float32"
    )


    # ========================================================
    # STEP 5 — KEYWORD RELEVANCE
    # ========================================================

    candidates_df[
        "keyword_score"
    ] = candidates_df[
        "text"
    ].apply(
        lambda text: calculate_keyword_score(
            query,
            text
        )
    )


    # ========================================================
    # STEP 6 — COMBINED RELEVANCE SCORE
    # ========================================================

    candidates_df[
        "combined_score"
    ] = (
        candidates_df[
            "reranker_score"
        ]
        +
        (
            candidates_df[
                "keyword_score"
            ]
            * 2.0
        )
    )


    # ========================================================
    # STEP 7 — PENALIZE LOW-INFORMATION RECORDS
    # ========================================================

    candidates_df.loc[
        candidates_df[
            "_low_information"
        ],
        "combined_score"
    ] -= 5.0


    # ========================================================
    # STEP 8 — SORT BY FINAL RELEVANCE
    # ========================================================

    candidates_df = candidates_df.sort_values(
        [
            "combined_score",
            "faiss_score"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # STEP 9 — REMOVE DUPLICATE GRIEVANCES
    # ========================================================

    final_rows = []

    seen_grievances = set()


    for _, row in candidates_df.iterrows():

        registration_no = str(
            row.get(
                "registration_no",
                ""
            )
        ).strip()


        if registration_no:

            if (
                registration_no
                in seen_grievances
            ):

                continue


            seen_grievances.add(
                registration_no
            )


        final_rows.append(
            row
        )


        if len(final_rows) >= top_k:

            break


    # ========================================================
    # STEP 10 — RETURN RESULTS
    # ========================================================

    if not final_rows:

        return pd.DataFrame()


    results_df = pd.DataFrame(
        final_rows
    ).reset_index(
        drop=True
    )


    # Remove internal helper column from final output.
    if "_low_information" in results_df.columns:

        results_df = results_df.drop(
            columns=[
                "_low_information"
            ]
        )


    return results_df


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def format_evidence(
    results
):

    if results is None:

        return (
            "No historical evidence "
            "was retrieved."
        )


    if isinstance(
        results,
        pd.DataFrame
    ):

        if results.empty:

            return (
                "No historical evidence "
                "was retrieved."
            )


        rows = results.to_dict(
            orient="records"
        )


    else:

        rows = results


    if not rows:

        return (
            "No historical evidence "
            "was retrieved."
        )


    output = []


    for i, result in enumerate(
        rows,
        start=1
    ):

        category = result.get(
            "category",
            "Unknown"
        )


        state = result.get(
            "state",
            "Unknown"
        )


        organization = result.get(
            "org_code",
            "Unknown"
        )


        resolution_days = result.get(
            "resolution_days",
            "Unknown"
        )


        text = result.get(
            "text",
            ""
        )


        faiss_score = result.get(
            "faiss_score",
            "Unknown"
        )


        reranker_score = result.get(
            "reranker_score",
            "Unknown"
        )


        keyword_score = result.get(
            "keyword_score",
            "Unknown"
        )


        combined_score = result.get(
            "combined_score",
            "Unknown"
        )


        output.append(
            f"""
Historical Evidence {i}:
Category: {category}
State: {state}
Organization: {organization}
Historical resolution days: {resolution_days}
FAISS similarity score: {faiss_score}
Reranker score: {reranker_score}
Keyword relevance score: {keyword_score}
Combined relevance score: {combined_score}

Grievance:
{text}
""".strip()
        )


    return "\n\n".join(
        output
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "water supply problems, "
        "irregular water availability, "
        "shortage of drinking water, "
        "inadequate water supply, "
        "low water pressure"
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "HISTORICAL RAG RETRIEVAL TEST"
    )

    print(
        "=" * 60
    )


    print(
        "\nQuery:"
    )

    print(
        test_query
    )


    results = retrieve_grievances(
        test_query,
        top_k=5,
        candidate_k=100
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "RETRIEVED HISTORICAL EVIDENCE"
    )

    print(
        "=" * 60
    )


    print(
        format_evidence(
            results
        )
    )