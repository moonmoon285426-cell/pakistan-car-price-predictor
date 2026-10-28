from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

folder = Path(__file__).parent

# Load the saved model and available input combinations
model = joblib.load(folder / "pakistan_car_price_model.pkl")
options = pd.read_csv(folder / "pakistan_car_options.csv")

st.title("Pakistan Used-Car Price Predictor")
st.write("Select a car and enter its details.")

st.info(
    "This learning model estimates historical asking prices "
    "from the 2023/2024 datasets—not current market prices."
)

# Make selection
make = st.selectbox(
    "Make",
    sorted(options["make"].dropna().unique())
)

# Show only models belonging to the selected make
make_options = options[options["make"] == make]

car_model = st.selectbox(
    "Model",
    sorted(make_options["model"].dropna().unique())
)

# Records for the selected make/model
car_options = make_options[
    make_options["model"] == car_model
]

years = sorted(
    car_options["year"].dropna().astype(int).unique(),
    reverse=True
)

year = st.selectbox("Car year", years)

km_driven = st.number_input(
    "Kilometres driven",
    min_value=0,
    value=50000,
    step=1000
)

engine_values = sorted(
    car_options["engine_cc"].dropna().unique()
)

if engine_values:
    engine_cc = st.selectbox(
        "Engine capacity (cc)",
        engine_values
    )
else:
    engine_cc = st.number_input(
        "Engine capacity (cc)",
        min_value=1,
        value=1000,
        step=100
    )

fuel_choices = sorted(
    car_options["fuel_type"].dropna().unique().tolist()
)

if not fuel_choices:
    fuel_choices = sorted(
        options["fuel_type"].dropna().unique().tolist()
    )

fuel_type = st.selectbox("Fuel type", fuel_choices)

transmission = st.selectbox(
    "Transmission",
    sorted(car_options["transmission"].dropna().unique())
)

city = st.selectbox(
    "City",
    sorted(options["city"].dropna().unique())
)

if st.button("Predict Price"):
    car = pd.DataFrame([{
        "make": make,
        "model": car_model,
        "year": year,
        "km_driven": km_driven,
        "engine_cc": engine_cc,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "city": city
    }])

    prediction = float(model.predict(car)[0])

    st.success(f"Estimated asking price: PKR {prediction:,.0f}")
    st.write(f"**In lakh:** {prediction / 100000:,.2f}")

    st.caption(
        "This is a model estimate, not a guaranteed sale price. "
        "Condition and exact variant are not included in this model."
    )
    
    