import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

data = {
    "time": [
        300, 270, 250, 230, 210,
        200, 180, 165, 150, 140,
        130, 115, 100, 90, 75
    ],

    "shots": [
        60, 55, 50, 48, 45,
        40, 38, 35, 32, 30,
        27, 25, 23, 20, 18
    ],

    "accuracy": [
        28, 31, 34, 35, 38,
        42, 45, 49, 53, 57,
        63, 68, 74, 85, 94
    ],

    "level": [
        "Beginner", "Beginner", "Beginner", "Beginner", "Beginner",
        "Intermediate", "Intermediate", "Intermediate", "Intermediate", "Intermediate",
        "Expert", "Expert", "Expert", "Expert", "Expert"
    ]
}


df = pd.DataFrame(data)

X = df[["time", "shots", "accuracy"]]
y = df["level"]

model = DecisionTreeClassifier(max_depth=4, random_state=42)

model.fit(X, y)

joblib.dump(model, "decision_tree_model.pkl")

print("Modellen er trænet og gemt!")

test_player = pd.DataFrame(
    [[145, 31, 55]],
    columns=["time", "shots", "accuracy"]
)

prediction = model.predict(test_player)

print(f"Testspilleren blev klassificeret som: {prediction[0]}")

