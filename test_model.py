import joblib 
import pandas as pd


model = joblib.load("decision_tree_model.pkl")


player_data = pd.DataFrame(
    [[260, 55, 30]],
    columns=["time", "shots", "accuracy"]
)

prediction = model.predict(player_data)

print(f"Spillerens niveau er: {prediction[0]}")

