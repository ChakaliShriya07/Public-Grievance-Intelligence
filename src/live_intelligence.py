from pymongo import MongoClient
from datetime import datetime, timezone
from statistics import median


# ============================================================
# MongoDB Configuration
# ============================================================

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "public_grievance_intelligence"
MONGO_COLLECTION_NAME = "complaints"


# ============================================================
# MongoDB Connection
# ============================================================

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=2500
)

db = client[MONGO_DB_NAME]
complaints_collection = db[MONGO_COLLECTION_NAME]


# ============================================================
# Get all live complaints
# ============================================================

def get_all_live_complaints():
    """
    Returns all current complaints from MongoDB.
    """

    complaints = list(
        complaints_collection.find(
            {},
            {
                "_id": 0,
                "grievance_id": 1,
                "citizen_email": 1,
                "category": 1,
                "description": 1,
                "location": 1,
                "status": 1,
                "assigned_department": 1,
                "admin_remarks": 1,
                "submitted_at": 1,
                "updated_at": 1,
                "status_history": 1,
                "feedback": 1,
            }
        )
    )

    return complaints


# ============================================================
# Basic live statistics
# ============================================================

def get_live_summary():
    """
    Returns current operational statistics from MongoDB.
    """

    complaints = get_all_live_complaints()

    total = len(complaints)

    submitted = sum(
        1 for c in complaints
        if c.get("status") == "Submitted"
    )

    under_review = sum(
        1 for c in complaints
        if c.get("status") == "Under Review"
    )

    resolved = sum(
        1 for c in complaints
        if c.get("status") == "Resolved"
    )

    rejected = sum(
        1 for c in complaints
        if c.get("status") == "Rejected"
    )

    unresolved = submitted + under_review

    return {
        "total_complaints": total,
        "submitted": submitted,
        "under_review": under_review,
        "resolved": resolved,
        "rejected": rejected,
        "unresolved": unresolved,
    }


# ============================================================
# Complaints by category
# ============================================================

def get_complaints_by_category():
    """
    Returns the number of live complaints for each category.
    """

    complaints = get_all_live_complaints()

    category_counts = {}

    for complaint in complaints:

        category = complaint.get("category") or "Unknown"

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    return dict(
        sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ============================================================
# Complaints by location
# ============================================================

def get_complaints_by_location():
    """
    Returns the number of live complaints for each location.
    """

    complaints = get_all_live_complaints()

    location_counts = {}

    for complaint in complaints:

        location = complaint.get("location") or "Unknown"

        location_counts[location] = (
            location_counts.get(location, 0) + 1
        )

    return dict(
        sorted(
            location_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ============================================================
# Complaints by department
# ============================================================

def get_complaints_by_department():
    """
    Returns the number of live complaints assigned to each
    department.
    """

    complaints = get_all_live_complaints()

    department_counts = {}

    for complaint in complaints:

        department = (
            complaint.get("assigned_department")
            or "Not Assigned"
        )

        department_counts[department] = (
            department_counts.get(department, 0) + 1
        )

    return dict(
        sorted(
            department_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ============================================================
# Current status distribution
# ============================================================

def get_status_distribution():
    """
    Returns current live complaint status distribution.
    """

    complaints = get_all_live_complaints()

    status_counts = {}

    for complaint in complaints:

        status = complaint.get("status") or "Unknown"

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    return dict(
        sorted(
            status_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )


# ============================================================
# Find similar live complaints
# ============================================================

def get_similar_live_complaints(
    category=None,
    location=None,
    exclude_grievance_id=None
):
    """
    Finds current complaints similar to a given complaint.

    Matching is based on category and location.
    """

    complaints = get_all_live_complaints()

    similar = []

    category = (category or "").strip().lower()
    location = (location or "").strip().lower()

    for complaint in complaints:

        grievance_id = complaint.get("grievance_id")

        if grievance_id == exclude_grievance_id:
            continue

        complaint_category = (
            complaint.get("category") or ""
        ).strip().lower()

        complaint_location = (
            complaint.get("location") or ""
        ).strip().lower()

        category_match = (
            category
            and complaint_category == category
        )

        location_match = (
            location
            and location in complaint_location
        )

        if category_match or location_match:

            similar.append(complaint)

    return similar


# ============================================================
# Similar complaint statistics
# ============================================================

def get_similar_complaint_stats(
    category=None,
    location=None,
    exclude_grievance_id=None
):
    """
    Calculates privacy-safe aggregate statistics for
    similar current complaints.
    """

    similar = get_similar_live_complaints(
        category=category,
        location=location,
        exclude_grievance_id=exclude_grievance_id
    )

    total = len(similar)

    if total == 0:
        return {
            "similar_count": 0,
            "resolved": 0,
            "submitted": 0,
            "under_review": 0,
            "rejected": 0,
            "resolution_rate": 0,
            "median_resolution_days": None,
        }

    resolved = sum(
        1 for c in similar
        if c.get("status") == "Resolved"
    )

    submitted = sum(
        1 for c in similar
        if c.get("status") == "Submitted"
    )

    under_review = sum(
        1 for c in similar
        if c.get("status") == "Under Review"
    )

    rejected = sum(
        1 for c in similar
        if c.get("status") == "Rejected"
    )

    resolution_days = []

    for complaint in similar:

        if complaint.get("status") != "Resolved":
            continue

        submitted_at = complaint.get("submitted_at")
        updated_at = complaint.get("updated_at")

        if submitted_at and updated_at:

            try:

                difference = (
                    updated_at - submitted_at
                ).total_seconds() / 86400

                if difference >= 0:
                    resolution_days.append(
                        difference
                    )

            except Exception:
                pass

    median_resolution = None

    if resolution_days:
        median_resolution = median(resolution_days)

    resolution_rate = (
        resolved / total * 100
    )

    return {
        "similar_count": total,
        "resolved": resolved,
        "submitted": submitted,
        "under_review": under_review,
        "rejected": rejected,
        "resolution_rate": round(
            resolution_rate,
            2
        ),
        "median_resolution_days": (
            round(median_resolution, 2)
            if median_resolution is not None
            else None
        ),
    }


# ============================================================
# Unresolved complaints
# ============================================================

def get_unresolved_complaints():
    """
    Returns complaints that still require action.
    """

    complaints = get_all_live_complaints()

    unresolved = [
        complaint
        for complaint in complaints
        if complaint.get("status")
        in ["Submitted", "Under Review"]
    ]

    return unresolved


# ============================================================
# Priority candidates
# ============================================================

def get_priority_complaints():
    """
    Identifies current complaints that may require
    administrative attention.

    This is a rule-based screening layer, not an ML prediction.
    """

    complaints = get_unresolved_complaints()

    priority = []

    now = datetime.now(timezone.utc)

    for complaint in complaints:

        submitted_at = complaint.get("submitted_at")

        if not submitted_at:
            continue

        try:

            if submitted_at.tzinfo is None:
                submitted_at = submitted_at.replace(
                    tzinfo=timezone.utc
                )

            age_days = (
                now - submitted_at
            ).total_seconds() / 86400

        except Exception:
            continue

        score = 0
        reasons = []

        # Older unresolved complaint
        if age_days >= 7:
            score += 3
            reasons.append(
                "Unresolved for more than 7 days"
            )

        elif age_days >= 3:
            score += 2
            reasons.append(
                "Unresolved for more than 3 days"
            )

        # Complaint still submitted
        if complaint.get("status") == "Submitted":
            score += 1
            reasons.append(
                "Not yet under review"
            )

        # No department assigned
        department = complaint.get(
            "assigned_department"
        )

        if not department or department == "Not Assigned":
            score += 2
            reasons.append(
                "Department not assigned"
            )

        if score > 0:

            priority.append(
                {
                    "grievance_id": complaint.get(
                        "grievance_id"
                    ),
                    "category": complaint.get(
                        "category"
                    ),
                    "location": complaint.get(
                        "location"
                    ),
                    "status": complaint.get(
                        "status"
                    ),
                    "assigned_department": complaint.get(
                        "assigned_department"
                    ),
                    "age_days": round(
                        age_days,
                        2
                    ),
                    "priority_score": score,
                    "reasons": reasons,
                }
            )

    priority.sort(
        key=lambda x: (
            x["priority_score"],
            x["age_days"]
        ),
        reverse=True
    )

    return priority


# ============================================================
# Generate live intelligence context
# ============================================================

def build_live_intelligence_context():
    """
    Creates a compact text representation of current
    MongoDB intelligence that can later be provided to Gemini.
    """

    summary = get_live_summary()

    categories = get_complaints_by_category()

    locations = get_complaints_by_location()

    departments = get_complaints_by_department()

    statuses = get_status_distribution()

    priority = get_priority_complaints()

    lines = []

    lines.append("LIVE COMPLAINT INTELLIGENCE")
    lines.append("===========================")

    lines.append(
        f"Total current complaints: "
        f"{summary['total_complaints']}"
    )

    lines.append(
        f"Submitted: {summary['submitted']}"
    )

    lines.append(
        f"Under Review: {summary['under_review']}"
    )

    lines.append(
        f"Resolved: {summary['resolved']}"
    )

    lines.append(
        f"Rejected: {summary['rejected']}"
    )

    lines.append(
        f"Unresolved: {summary['unresolved']}"
    )

    lines.append("")

    lines.append("Current categories:")

    for category, count in list(
        categories.items()
    )[:10]:

        lines.append(
            f"- {category}: {count}"
        )

    lines.append("")

    lines.append("Current locations:")

    for location, count in list(
        locations.items()
    )[:10]:

        lines.append(
            f"- {location}: {count}"
        )

    lines.append("")

    lines.append("Current departments:")

    for department, count in list(
        departments.items()
    )[:10]:

        lines.append(
            f"- {department}: {count}"
        )

    lines.append("")

    lines.append("Current status distribution:")

    for status, count in statuses.items():

        lines.append(
            f"- {status}: {count}"
        )

    lines.append("")

    lines.append(
        f"Priority candidates: {len(priority)}"
    )

    for item in priority[:10]:

        lines.append(
            f"- {item['grievance_id']} | "
            f"{item['category']} | "
            f"{item['location']} | "
            f"{item['status']} | "
            f"Priority score: "
            f"{item['priority_score']}"
        )

    return "\n".join(lines)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("\nMongoDB Live Intelligence Test")
    print("=" * 40)

    summary = get_live_summary()

    print("\nLive Summary:")
    print(summary)

    print("\nCategories:")
    print(get_complaints_by_category())

    print("\nLocations:")
    print(get_complaints_by_location())

    print("\nDepartments:")
    print(get_complaints_by_department())

    print("\nStatus Distribution:")
    print(get_status_distribution())

    print("\nPriority Complaints:")

    for complaint in get_priority_complaints():
        print(complaint)

    print("\nGenerated Intelligence Context:")
    print(build_live_intelligence_context())