import pandas as pd

# Direct import from sibling package (works when functions/ is in sys.path)
from sales_report.utils_firestore import commit_to_firestore_in_batches

def commit_to_firestore(df: pd.DataFrame, collection_name: str, delete_existing: bool = True):
    """
    Commits a DataFrame to a Firestore collection in batches, using the shared utility function.
    """
    commit_to_firestore_in_batches(df, collection_name, delete_existing_collection=delete_existing)
