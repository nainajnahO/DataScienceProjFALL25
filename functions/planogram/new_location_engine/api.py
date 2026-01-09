import sys
import os
import requests
from flask import Flask, request, jsonify
from pathlib import Path

# DEBUG: Print current python executable
print(f"DEBUG: Python Executable: {sys.executable}")
print(f"DEBUG: Python Version: {sys.version}")

# Add functions root directory to path so we can import 'planogram' and other modules
# This file is in functions/planogram/new_location_engine/
# We need functions/ to be in sys.path
FILE_DIR = Path(__file__).resolve().parent
FUNCTIONS_DIR = FILE_DIR.parents[1]
if str(FUNCTIONS_DIR) not in sys.path:
    print(f"Adding {FUNCTIONS_DIR} to sys.path")
    sys.path.append(str(FUNCTIONS_DIR))

# Add current directory (for backward compatibility if needed)
sys.path.append(str(FILE_DIR))

try:
    # Import the new LocationForecaster
    # Since we added FUNCTIONS_DIR to path, we can import from planogram
    from planogram.new_location_engine.location_forecast import LocationForecaster
    print("Successfully imported LocationForecaster")
except ImportError as e:
    print(f"Error importing LocationForecaster: {e}")
    sys.exit(1)

app = Flask(__name__)
service = None

def get_lat_lon(address):
    """
    Geocode address using Nominatim (OpenStreetMap).
    Please respect usage policy: max 1 req/sec, providing User-Agent.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {
            'User-Agent': 'PlanogramApp/1.0 (internal tool)'
        }
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200 and len(response.json()) > 0:
            data = response.json()[0]
            return float(data['lat']), float(data['lon'])
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None

@app.route('/predict', methods=['POST'])
def predict():
    global service
    if not service:
        try:
            print("Attempting lazy initialization of LocationForecaster...")
            service = LocationForecaster()
            print("LocationForecaster initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize service: {e}")
            return jsonify({'error': f"Service initialization failed: {str(e)}"}), 500

    try:
        data = request.json
        products = data.get('products', [])
        address = data.get('address', '')
        machine_category = data.get('category', 'WORK')

        if not products or not address:
            return jsonify({'error': 'Missing products or address'}), 400

        lat, lon = get_lat_lon(address)
        if lat is None or lon is None:
            return jsonify({'error': 'Could not geocode address'}), 404

        print(f"Predicting for {address} ({lat}, {lon}) - Category: {machine_category}")
        
        # Call valid forecast service
        df_results = service.predict(
            latitude=lat, 
            longitude=lon, 
            products=products,
            machine_category=machine_category
        )
        
        # Calculate total sales from results
        total_sales = 0
        if not df_results.empty and 'rule_based_prediction' in df_results.columns:
             total_sales = df_results['rule_based_prediction'].sum()
        
        return jsonify({
            'total_monthly_sales': round(total_sales, 2),
            'currency': 'SEK', 
            'details': df_results.to_dict(orient='records')
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Lazy load AutofillService
autofill_service = None

@app.route('/autofill', methods=['POST'])
def autofill():
    global autofill_service
    if not autofill_service:
        try:
            print("Attempting lazy initialization of AutofillService...")
            from planogram.new_location_engine.autofill_wrapper import AutofillService
            autofill_service = AutofillService()
            print("AutofillService initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize AutofillService: {e}")
            return jsonify({'error': f"Service initialization failed: {str(e)}"}), 500

    try:
        data = request.json
        slots = data.get('slots', [])
        machine_id = data.get('machineId', 'unknown-machine')
        action = data.get('action', 'fill') # 'fill' or 'optimize'
        
        address = data.get('address', '')
        static_weights = data.get('static_weights')
        dynamic_weights = data.get('dynamic_weights')
        
        if not slots:
             return jsonify({'error': 'Missing slots data'}), 400

        print(f"Autofill request: {action} for {machine_id} ({len(slots)} slots). Address: {address}")
        
        lat, lon = None, None
        if address:
             lat, lon = get_lat_lon(address)
        
        if action == 'optimize':
            updated_slots = autofill_service.optimize(
                slots, machine_id, lat=lat, lon=lon, 
                static_weights=static_weights, dynamic_weights=dynamic_weights
            )
        else:
            updated_slots = autofill_service.autofill(
                slots, machine_id, lat=lat, lon=lon,
                static_weights=static_weights, dynamic_weights=dynamic_weights
            )
            
        return jsonify({
            'success': True,
            'slots': updated_slots
        })

    except Exception as e:
        print(f"Autofill error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Prediction API...")
    # Initialize service eagerly so it's ready
    try: 
        service = LocationForecaster()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # We continue to run so the app doesn't crash on spawn, but requests will fail
    
    # Run on a specific port, e.g., 5001 to avoid conflicts
    app.run(host='127.0.0.1', port=5001)


