"""
AI Grievance Intelligence Engine

Combines:
1. Current live grievance data from MongoDB
2. Historical grievance evidence from RAG
3. Gemini for grounded administrative intelligence

Important:
- Current facts must come from live data.
- Historical evidence must come from the historical dataset.
- Historical examples must not be presented as current facts.
- No unsupported predictions.
- No PII exposure.
"""

from google import genai

from src.live_intelligence import (
    build_live_intelligence_context,
    get_complaints_by_category,
)

from src.rag_engine import retrieve_grievances


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.5-flash"

client = genai.Client()


# ============================================================
# CURRENT CATEGORY DETECTION
# ============================================================

def get_current_categories():
    """
    Get categories currently present in MongoDB.

    These categories are used to create focused historical
    RAG searches instead of sending a broad generic query.
    """

    try:
        category_data = get_complaints_by_category()

        if not category_data:
            return []

        if hasattr(category_data, "index"):
            categories = list(category_data.index)
        elif isinstance(category_data, dict):
            categories = list(category_data.keys())
        else:
            categories = []

        categories = [
            str(category).strip()
            for category in categories
            if category is not None
            and str(category).strip()
            and str(category).strip().lower() not in {
                "unknown",
                "not specified",
                "none",
                "nan",
            }
        ]

        return list(dict.fromkeys(categories))

    except Exception as e:
        print(f"Error detecting current categories: {e}")
        return []


# ============================================================
# HISTORICAL QUERY BUILDER
# ============================================================

def build_historical_query(category):
    """
    Build a focused historical RAG query for the current
    grievance category.
    """

    return (
        f"Historical government grievances related to {category}. "
        f"Find past complaints describing problems, service issues, "
        f"shortages, delays, inadequate service, recurring issues, "
        f"and similar public grievances related to {category}. "
        f"Focus on actual grievance descriptions and official "
        f"resolution information where available."
    )


# ============================================================
# HISTORICAL EVIDENCE RETRIEVAL
# ============================================================

def get_historical_evidence_for_current_issues(
    categories,
    top_k_per_category=3,
    final_top_k=5,
):
    """
    Retrieve historical evidence separately for each current
    grievance category.

    This prevents unrelated historical records from being
    introduced into the AI analysis.
    """

    all_results = []

    if not categories:
        return []

    for category in categories:

        query = build_historical_query(category)

        try:
            results = retrieve_grievances(
                query,
                top_k=top_k_per_category,
                candidate_k=100,
            )

            if results is None:
                continue

            # Convert DataFrame to records if necessary
            if hasattr(results, "to_dict"):
                records = results.to_dict("records")
            elif isinstance(results, list):
                records = results
            else:
                records = [results]

            for record in records:

                if not isinstance(record, dict):
                    continue

                record_copy = dict(record)

                record_copy["current_issue_category"] = category

                all_results.append(record_copy)

        except Exception as e:
            print(
                f"Historical retrieval failed for "
                f"category '{category}': {e}"
            )

    # --------------------------------------------------------
    # Sort by relevance if available
    # --------------------------------------------------------

    def relevance_score(record):

        possible_fields = [
            "combined_score",
            "relevance_score",
            "score",
            "reranker_score",
        ]

        for field in possible_fields:

            value = record.get(field)

            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                pass

        return 0.0

    all_results.sort(
        key=relevance_score,
        reverse=True,
    )

    # --------------------------------------------------------
    # Document-level deduplication
    # --------------------------------------------------------

    unique_results = []
    seen_ids = set()

    for record in all_results:

        registration_no = record.get("registration_no")

        if registration_no:

            registration_no = str(registration_no)

            if registration_no in seen_ids:
                continue

            seen_ids.add(registration_no)

        unique_results.append(record)

        if len(unique_results) >= final_top_k:
            break

    return unique_results


# ============================================================
# HISTORICAL EVIDENCE FORMATTER
# ============================================================

def format_historical_evidence(evidence):

    if not evidence:
        return "No sufficiently relevant historical evidence was retrieved."

    formatted = []

    for i, record in enumerate(evidence, start=1):

        if not isinstance(record, dict):
            continue

        category = record.get(
            "current_issue_category",
            "Unknown",
        )

        state = record.get(
            "state",
            "Not available",
        )

        org_code = record.get(
            "org_code",
            "Not available",
        )

        historical_category = record.get(
            "category",
            "Not available",
        )

        resolution_days = record.get(
            "resolution_days",
            "Not available",
        )

        text = record.get(
            "text",
            record.get(
                "subject_content_text",
                "Not available",
            ),
        )

        if text is None:
            text = "Not available"

        text = str(text).strip()

        # Limit extremely long evidence
        if len(text) > 1200:
            text = text[:1200] + "..."

        formatted.append(
            f"""
Historical Evidence {i}

Current Issue Category:
{category}

Historical Category:
{historical_category}

State / Region:
{state}

Organisation:
{org_code}

Historical Resolution Days:
{resolution_days}

Historical Grievance:
{text}
""".strip()
        )

    if not formatted:
        return "No sufficiently relevant historical evidence was retrieved."

    return "\n\n".join(formatted)


# ============================================================
# COMBINED EVIDENCE
# ============================================================

def build_combined_evidence(question):

    live_context = build_live_intelligence_context()

    current_categories = get_current_categories()

    historical_evidence = (
        get_historical_evidence_for_current_issues(
            current_categories,
            top_k_per_category=3,
            final_top_k=5,
        )
    )

    historical_context = format_historical_evidence(
        historical_evidence
    )

    combined_context = f"""
============================================================
LIVE CURRENT GRIEVANCE DATA
============================================================

{live_context}


============================================================
CURRENT ISSUE CATEGORIES
============================================================

{current_categories}


============================================================
HISTORICAL RAG EVIDENCE
============================================================

{historical_context}


============================================================
USER QUESTION
============================================================

{question}
"""

    return combined_context


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(question):

    evidence = build_combined_evidence(question)

    prompt = f"""
You are an AI assistant for a Public Grievance Intelligence
Platform used by administrators to understand citizen grievance
patterns.

You have two different evidence sources:

1. LIVE CURRENT DATA
   - Current complaints stored in MongoDB.
   - Current status, categories, locations, departments,
     unresolved complaints and priority candidates.

2. HISTORICAL RAG EVIDENCE
   - Past government grievance records.
   - Used only for historical comparison, similar past issues,
     recurring patterns and historical resolution examples.

Your answer must strictly distinguish between these two sources.

============================================================
GROUNDING RULES
============================================================

RULE 1 — CURRENT FACTS

Only state something as a current fact if it is directly
supported by the live MongoDB data.

Examples:

Correct:
"There are 5 current complaints."

Correct:
"3 complaints are currently Submitted."

Incorrect:
"The water pipeline is currently blocked."

The historical evidence cannot be used to establish a current
fact.


RULE 2 — HISTORICAL EVIDENCE

Historical records describe past grievances.

Always make it clear when an example comes from historical data.

Use wording such as:

"Historical records show..."

"A past grievance reported..."

"Historical evidence includes..."

Never present a historical event as something currently
happening.


RULE 3 — HISTORICAL EVIDENCE IS NOT PROOF OF CURRENT CAUSE

If a historical grievance mentions:

- illegal water extraction
- pipeline blockage
- poor engineering
- departmental delay
- infrastructure failure
- staff inspection

you may say that this is a historical example or an
investigation point.

You MUST NOT say that the same problem currently exists unless
the live data explicitly confirms it.


RULE 4 — INVESTIGATION POINTS

When historical evidence suggests something administrators
could investigate, clearly label it as an investigation point.

Example:

"Historical grievances suggest that pipeline pressure,
distribution interruptions, or unauthorized extraction could be
considered as investigation points."

Do NOT say:

"The current problem is caused by unauthorized extraction."


RULE 5 — NO UNSUPPORTED PREDICTIONS

Do not predict:

- whether a complaint will be resolved
- exact future resolution time
- future department performance
- future complaint volume
- individual citizen outcomes

unless a validated predictive model is explicitly provided.


RULE 6 — HISTORICAL RESOLUTION TIME

Historical resolution days are benchmarks or context.

They are NOT predictions for an individual current complaint.

Correct:

"Historical records provide a benchmark of past resolution
times."

Incorrect:

"This complaint will be resolved within 10 days."


RULE 7 — PRIORITY

If the live intelligence engine identifies complaints as
priority candidates, report the actual priority score and
reasons.

Do not automatically call them "critical" unless the system
explicitly defines that severity level.


RULE 8 — SMALL SAMPLE CAUTION

If the number of current complaints is small, do not make strong
statistical or causal claims.

For example, if all 5 current complaints belong to one category,
say:

"All current complaints in the available sample are related to
Water Supply."

Do NOT automatically conclude:

"This proves a systemic problem."


RULE 9 — LOCATION CLUSTERING

A concentration of complaints in one location can be described
as a geographic concentration.

It may justify further investigation.

Do not claim that the location concentration proves a systemic
cause.


RULE 10 — PII

Never expose:

- citizen names
- email addresses
- phone numbers
- home addresses
- personal identifiers
- registration details of unrelated citizens

Use only aggregated or operationally necessary information.


RULE 11 — LANGUAGE

Historical grievances may contain Hindi or other languages.

You may understand multilingual evidence, but the final answer
must ALWAYS be written in English.

Do not reproduce long original-language grievance text.


============================================================
ANALYSIS STRUCTURE
============================================================

Use these sections:

### Current Situation

Describe only directly observed current live data.

Include where useful:

- total current complaints
- categories
- status distribution
- unresolved complaints
- department assignment
- geographic concentration
- priority candidates

### Historical Comparison

Compare the current issue with relevant historical evidence.

Clearly identify historical examples as historical.

Mention recurring patterns only when supported by the retrieved
evidence.

### Key Insight

Give a concise evidence-based interpretation.

Clearly distinguish:

- observed fact
- historical evidence
- reasonable inference

### Administrative Implication

Suggest practical actions administrators may consider.

Separate confirmed facts from investigation points.

For example:

- assigning currently unassigned complaints
- reviewing workload
- conducting field verification
- checking infrastructure
- comparing current performance with historical benchmarks

Do not claim that an investigation finding is already confirmed.


============================================================
EVIDENCE
============================================================

{evidence}


============================================================
QUESTION
============================================================

{question}

Provide a concise, professional and evidence-grounded answer.
"""


    return prompt


# ============================================================
# MAIN AI FUNCTION
# ============================================================

def ask_ai_grievance_intelligence(question):

    try:

        prompt = build_prompt(question)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if response is None:
            return "No response was generated."

        text = getattr(
            response,
            "text",
            None,
        )

        if text:
            return text.strip()

        return "The AI model returned an empty response."

    except Exception as e:

        return (
            "AI Grievance Intelligence failed.\n\n"
            f"Error: {str(e)}"
        )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AI GRIEVANCE INTELLIGENCE TEST")
    print("=" * 70)

    question = (
        "What are the major current grievance issues and "
        "how do they compare with historical complaints?"
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerating intelligence...\n")

    answer = ask_ai_grievance_intelligence(question)

    print("=" * 70)
    print("AI RESPONSE")
    print("=" * 70)
    print(answer)

    print("\n")