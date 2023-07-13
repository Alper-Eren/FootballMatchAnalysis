import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Dataset
df = pd.read_csv('dataset_results.csv')  # Replace 'Dataset.csv' with the path to your dataset

# Step 2: Get a List of Valid Teams
valid_teams = list(set(df['home_team']).union(set(df['away_team'])))
print(valid_teams)
team_number = len(valid_teams)
print("Dizideki eleman sayisi:", team_number)

# Step 3: Encode Team Labels
team_encoder = LabelEncoder()
df['home_team'] = team_encoder.fit_transform(df['home_team'])
df['away_team'] = team_encoder.transform(df['away_team'])

# Step 4: Encode Result Labels
result_encoder = LabelEncoder()
df['result'] = result_encoder.fit_transform(df['result'])

# Step 5: Train the Model
X_train = df[['home_team', 'away_team']]
y_train = df['result']

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Analyze Matches
while True:
    home_team = input("Enter the home team (or 'Q' to quit): ")
    if home_team.upper() == 'Q':
        break

    away_team = input("Enter the away team: ")

    if home_team not in valid_teams:
        print(f"The team '{home_team}' does not exist. Please try again.")
        continue

    if away_team not in valid_teams:
        print(f"The team '{away_team}' does not exist. Please try again.")
        continue

    home_team_encoded = team_encoder.transform([home_team])[0]
    away_team_encoded = team_encoder.transform([away_team])[0]

    # Analyze matches between the teams
    team_matches = df[((df['home_team'] == home_team_encoded) & (df['away_team'] == away_team_encoded))
                      | ((df['home_team'] == away_team_encoded) & (df['away_team'] == home_team_encoded))]

    if len(team_matches) == 0:
        print(f"No match records found between '{home_team}' and '{away_team}'.")
        print("--------------------")
        continue

    # Step 7: Predict Match Outcome
    test_sample = team_matches[['home_team', 'away_team']]

    outcome_probabilities = model.predict_proba(test_sample)

    # Step 8: Predict Total Number of Goals
    home_goals_model = RandomForestRegressor(n_estimators=100)
    home_goals_model.fit(X_train, df['home_goals'])

    away_goals_model = RandomForestRegressor(n_estimators=100)
    away_goals_model.fit(X_train, df['away_goals'])

    home_goals_prediction = home_goals_model.predict(test_sample)
    away_goals_prediction = away_goals_model.predict(test_sample)

    # Convert outcome probabilities to percentages
    class_labels = result_encoder.classes_
    outcome_probabilities_percentages = {class_labels[i]: prob * 100 for i, prob in enumerate(outcome_probabilities[0])}

    # Calculate average goals per match
    avg_home_goals = team_matches['home_goals'].mean()
    avg_away_goals = team_matches['away_goals'].mean()

    # Display predictions
    # Plotting the Match Outcome Probabilities
    labels = list(outcome_probabilities_percentages.keys())
    probabilities = list(outcome_probabilities_percentages.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, probabilities)
    plt.xlabel('Match Outcome')
    plt.ylabel('Probability (%)')
    plt.title('Match Outcome Probabilities')
    plt.show()

    # Plotting the Total Number of Goals Prediction
    avg_goals = [avg_home_goals, avg_away_goals]
    predicted_goals = [home_goals_prediction[0], away_goals_prediction[0]]

    goal_labels = ['Average Home Goals', 'Average Away Goals', 'Predicted Home Goals', 'Predicted Away Goals']
    x_pos = np.arange(len(goal_labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x_pos, avg_goals + predicted_goals)
    plt.xticks(x_pos, goal_labels, rotation=45)
    plt.ylabel('Goals')
    plt.title('Total Number of Goals Prediction')
    plt.show()


    print(f"Match Analysis: {home_team} vs {away_team}")
    print("--------------------")
    print("Match Outcome Probabilities:")
    for label, percentage in outcome_probabilities_percentages.items():
        print(f"{label}: {percentage:.2f}%")

    print("\nTotal Number of Goals Prediction:")
    print(f"Average Goals per Match: {avg_home_goals:.2f} (Home), {avg_away_goals:.2f} (Away)")
    print(f"Home Goals: {home_goals_prediction[0]:.2f}")
    print(f"Away Goals: {away_goals_prediction[0]:.2f}")
    print("--------------------")

print("Program ended.")
