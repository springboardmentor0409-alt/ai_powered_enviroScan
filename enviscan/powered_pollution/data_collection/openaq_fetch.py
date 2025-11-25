import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError(" Missing API key")

HEADERS = {"X-API-Key": API_KEY}

# Fetch active monitoring locations (with sensors)
def get_locations(country="IN", limit=50):
    """
    Get list of available monitoring stations and their active sensors.
    """
    url = "https://api.openaq.org/v3/locations"
    params = {"country_id": country, "limit": limit}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])

    print(f"\n Found {len(results)} locations with metadata:\n")
    for loc in results:
        coords = loc.get("coordinates", {})
        sensors = loc.get("sensors", [])
        if sensors:
            print(f" {loc['name']} [{coords.get('latitude')}, {coords.get('longitude')}]")
            for s in sensors:
                param = s.get("parameter", {}).get("name")
                unit = s.get("parameter", {}).get("units")
                print(f"   • {param} ({unit})")
        else:
            print(f" {loc['name']} has no active sensors.")
    return results

# Extract and Save Latest Measurement Data
def get_latest_data(country="IN", limit=100):
    """
    Extract the latest pollutant readings directly from /v3/locations.
    Saves all current values into 'pollution_data.csv'.
    """
    url = "https://api.openaq.org/v3/locations"
    params = {"country_id": country, "limit": limit}
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    records = []

    for loc in results:
        coords = loc.get("coordinates", {})
        city = loc.get("city")
        name = loc.get("name")
        for sensor in loc.get("sensors", []):
            p = sensor.get("parameter", {})
            records.append({
                "city": city,
                "location": name,
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "parameter": p.get("name"),
                "unit": p.get("units"),
                "value": sensor.get("lastValue"),
                "lastUpdated": sensor.get("lastUpdated"),
            })

    if not records:
        print(" No measurement data found for this region.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.to_csv("pollution_data.csv", index=False)
    print(f"\n Data saved → pollution_data.csv ({len(df)} records)")
    return df

# Example Workflow
if __name__ == "__main__":
    print(" Fetching available Indian monitoring locations…")
    get_locations(country="IN", limit=20)

    print("\n Collecting latest pollutant readings…")
    df = get_latest_data(country="IN", limit=100)
    print("\nPreview of saved data:")
    print(df.head())
