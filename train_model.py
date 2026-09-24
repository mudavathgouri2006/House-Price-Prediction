
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ==============================
# 1. Create Dataset
# ==============================

rng = np.random.default_rng(42)
n = 500

area = rng.integers(500, 4000, n)
bedrooms = rng.integers(1, 6, n)
bathrooms = rng.integers(1, 5, n)
age = rng.integers(0, 30, n)
parking = rng.integers(0, 3, n)


# ==============================
# 2. All Indian States and UTs
# ==============================

states = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",

    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry"
]

locations = rng.choice(states, n)


# ==============================
# 3. Location Price Values
# ==============================

location_price = {
    state: 500000 for state in states
}

price = (
    area * 4500
    + bedrooms * 350000
    + bathrooms * 250000
    - age * 45000
    + parking * 150000
    + np.array([
        location_price[x] for x in locations
    ])
    + rng.normal(0, 250000, n)
)


# ==============================
# 4. Create DataFrame
# ==============================

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "parking": parking,
    "location": locations
})


X = df
y = price


# ==============================
# 5. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            ["location"]
        )
    ],
    remainder="passthrough"
)


# ==============================
# 6. Create Machine Learning Model
# ==============================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# ==============================
# 7. Train Model
# ==============================

model.fit(X, y)


# ==============================
# 8. Save Model
# ==============================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/house_price_model.pkl"
)


print("Model trained and saved successfully!")