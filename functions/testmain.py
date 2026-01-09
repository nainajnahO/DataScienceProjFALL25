#from firebase_functions import https_fn, options, firestore_fn
#from firebase_admin import initialize_app, firestore
#import google.cloud.firestore
import logging

from typing import Any

from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
import io
import os
import tempfile
import pathlib

#from firebase_admin import credentials
#from firebase_admin import storage

from sales_report import *

report_type = "Påfyllare" # Påfyllare, Slutkund, Leverantör, Placering, Total 

refiller = "ICA Nära Stabby" # Påfyllare (Fyll bara i om rapporttyp är Påfyllare)
customer = "" # NOT YET IMPLEMENTED Slutkund (Fyll bara i om rapporttyp är Slutkund)
provider = "" # NOT YET IMPLEMENTED Leverantör (Fyll bara i om rapporttyp är Leverantör)
placement = "" # Placering (Fyll bara i om rapporttyp är Placering), välj bland dessa: ['WORK', 'SPORTS GROUNDS', 'GYM', 'MALL', 'WAITING ROOMS', 'SCHOOLS, UNIV', 'LEISURE ENTERT. VENUES', 'Varuautomat', 'Kaffeautomat', 'PETROL STATION']

time_plot_type = 1 # 1 - Intäkter och antal köp på y-axeln, 2 - Snittförsäljning per automat jämförs med året innan på y-axeln

time = "2024-03" # År, Månad. t.ex. "2023" för år eller "2023-7" för månad
automater = '' # Lämna tom för at ta med alla automater, fyll i automatnamn för att bara ta med vissa automater
gpt_call = True # Sätt till True för att inkludera ChatGPT genererade texter. Sätt till False för att testa utan GPT-texter och därmed minska körtiden
title = "test" # Titel på rapporten tillika namnet på pdf filen. Sätt t.ex. till "Friskis & Svettis" för att få skapa "files/processed/Säljanalys: Friskis & Svettis 2023.pdf"

# Generate PDF content
pdf_io = io.BytesIO()
pdf = canvas.Canvas(pdf_io, pagesize=page_size)

categoriesPage = False
cousinPage = False
pricePage = False

#combined.create_report(pdf, report_type, time, automater, gpt_call, refiller, customer, provider, placement, title, time_plot_type, categoriesPage, cousinPage, pricePage)

# save PDF to file locally
pdf.save()
pdf_io.seek(0)
pdf_bytes = pdf_io.read()



print("Done!")