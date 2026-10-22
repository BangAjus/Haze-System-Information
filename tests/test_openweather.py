from pipeline.assets.openweather import to_row

CITY = {
    "geonameid": 1642911,
    "city_name": "Jakarta",
    "country_code": "ID",
    "country_name": "Indonesia",
    "continent": "AS",
    "population": 8540121,
    "lat": -6.21462,
    "lon": 106.84513,
}

# Trimmed real responses (2026-10-22). "rain" and "gust" are absent when there is none.
WEATHER = {
    "dt": 1792636800,
    "main": {"temp": 32.65, "feels_like": 36.1, "humidity": 52, "pressure": 1008},
    "wind": {"speed": 4.1, "deg": 350},
    "clouds": {"all": 20},
    "visibility": 10000,
    "sys": {"sunrise": 1792622000, "sunset": 1792666000},
}
AIR = {
    "list": [
        {
            "dt": 1792638000,
            "main": {"aqi": 2},
            "components": {
                "co": 400.5,
                "no": 0.1,
                "no2": 20.0,
                "o3": 60.0,
                "so2": 5.0,
                "pm2_5": 12.16,
                "pm10": 29.56,
                "nh3": 3.0,
            },
        }
    ]
}


def test_maps_to_kaggle_column_names():
    row = to_row(CITY, WEATHER, AIR)
    assert row["geonameid"] == 1642911
    assert row["air_time_unix"] == 1792638000  # observation time used by staging
    assert row["weather_time_unix"] == 1792636800
    assert row["pm2_5"] == 12.16 and row["pm10"] == 29.56 and row["aqi"] == 2
    assert row["temp"] == 32.65 and row["geo_lat"] == -6.21462


def test_missing_rain_and_gust_are_handled():
    row = to_row(CITY, WEATHER, AIR)
    assert row["rain_1h"] == 0.0 and row["snow_1h"] == 0.0
    assert row["wind_gust"] is None


def test_daytime_from_sunrise_and_sunset():
    assert to_row(CITY, WEATHER, AIR)["is_daytime"] == 1
    night = {**WEATHER, "dt": 1792670000}
    assert to_row(CITY, night, AIR)["is_daytime"] == 0
