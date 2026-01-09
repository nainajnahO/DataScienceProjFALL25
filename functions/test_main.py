# test_main.py
try:
# Welcome to Cloud Functions for Firebase for Python!
    # To get started, simply uncomment the below code or create your own.
    # Deploy with `firebase deploy`
    from firebase_functions import https_fn, options
    from firebase_admin import initialize_app, firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
    from firebase_admin import credentials
    from firebase_admin import storage
    import google.cloud.firestore
    import logging
    from google.cloud import pubsub_v1
    import hashlib

    # The Cloud Functions for Firebase SDK to set up triggers and logging.
    from firebase_functions import scheduler_fn
    from typing import Any
    from dateutil.relativedelta import relativedelta
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timedelta
    from reportlab.pdfgen import canvas
    #from fuzzywuzzy import process
    import requests
    import pathlib
    import imaplib
    import email
    import pytz
    import copy
    import re
    import io
    import boto3
    import os         # For file system operations (e.g., os.path.exists, os.makedirs)
    import pandas as pd  # For DataFrame manipulations (e.g., pd.read_csv, pd.DataFrame)
    import numpy as np   # For operations like np.ndarray conversion
    from io import StringIO 

    from sales_report import *
    import sales_report.combined_sales_report as sales_report
    print("All imports succeeded!")
except Exception as e:
    print("Error during import:", e)