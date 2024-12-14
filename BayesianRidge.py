from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import pandas as pd
import requests
import re

app = Flask(__name__)

# Global variables
PLAYERS = []

TEAMS = {}  # Global dictionary to store team mappings


def fetch_players():
    global PLAYERS, TEAMS
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        # Extract team names mapping
        TEAMS = {team["id"]: team["name"] for team in data.get("teams", [])}

        # Player position mapping
        positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"} 

        # Fetch fixtures for upcoming matches
        fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
        fixtures = fixtures_response.json() if fixtures_response.status_code == 200 else []
        next_opponents = {}

        for fixture in fixtures:
            if fixture["event"] and not fixture.get("finished", False):
                home_team, away_team = fixture["team_h"], fixture["team_a"]
                next_opponents[home_team] = away_team
                next_opponents[away_team] = home_team

        PLAYERS = [
            {
                **player,
                "team_name": TEAMS.get(player.get("team"), "Unknown Team"),
                "next_opponent": TEAMS.get(next_opponents.get(player.get("team")), "Unknown Team"),
                "position": positions.get(player.get("element_type"), "Unknown"),
            }
            for player in data.get("elements", [])
        ]

    else:
        print(f"Error fetching data: {response.status_code}")

# Pagination helper
def paginate(data, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    return data[start:end]

@app.route('/')
def home():
    search_query = request.args.get('search', '').lower()
    page = int(request.args.get('page', 1))
    per_page = 50

    # Get sort and filter parameters
    sort_by = request.args.get('sort_by', 'points')  # Default sorting by cost
    position_filter = request.args.get('position', '')
    team_filter = request.args.get('team', '')

    filtered_players = PLAYERS

    # Apply search query filter
    if search_query:
        filtered_players = [p for p in filtered_players if search_query in p['web_name'].lower()]

    # Apply position filter
    if position_filter:
        filtered_players = [p for p in filtered_players if p['position'] == position_filter]

    # Apply team filter
    if team_filter:
        filtered_players = [p for p in filtered_players if p['team_name'] == team_filter]

    # Fetch team data to map team ID to team name
    teams_response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    teams = {}
    
    if teams_response.status_code == 200:
        teams_data = teams_response.json().get("teams", [])
        teams = {team["id"]: team["name"] for team in teams_data}

    # Fetch fixtures for upcoming matches and create next_opponent mapping
    fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
    fixtures = []
    
    if fixtures_response.status_code == 200:
        fixtures_data = fixtures_response.json()
        next_opponents = {}

        # Step 1: Create a mapping for the next fixture for each team
        for fixture in fixtures_data:
            # Only consider fixtures that are part of a gameweek and are not finished
            if fixture.get("event") and not fixture.get("finished", False):
                home_team = fixture["team_h"]
                away_team = fixture["team_a"]
                gameweek = fixture["event"]

                # Store the first (earliest) fixture for each team in the next gameweek
                if home_team not in next_opponents or fixture["kickoff_time"] < next_opponents[home_team]["kickoff_time"]:
                    next_opponents[home_team] = fixture
                if away_team not in next_opponents or fixture["kickoff_time"] < next_opponents[away_team]["kickoff_time"]:
                    next_opponents[away_team] = fixture

        # Step 2: Assign the next opponent based on the next fixture
        for player in filtered_players:
            player_team = player['team']
            next_fixture = next_opponents.get(player_team)
            if next_fixture:
                opponent_id = next_fixture["team_a"] if next_fixture["team_h"] == player_team else next_fixture["team_h"]
                player["next_opponent"] = teams.get(opponent_id, "Unknown Team")
                player["next_opponent_gameweek"] = next_fixture["event"]

    # Sorting based on selected criteria
    if sort_by == 'cost':
        filtered_players.sort(key=lambda x: x['now_cost'])
    elif sort_by == 'points':
        filtered_players.sort(key=lambda x: x['total_points'], reverse=True)

    paginated_players = paginate(filtered_players, page, per_page)

    return render_template(
        "index.html",
        players=paginated_players,
        page=page,
        total=len(filtered_players),
        per_page=per_page,
        search_query=search_query,
        position_filter=position_filter,
        team_filter=team_filter,
        sort_by=sort_by,
        teams=sorted(set([p["team_name"] for p in PLAYERS]))  # Pass unique teams
    )


@app.route('/player/<int:player_id>')
def player_details(player_id):
    player = next((p for p in PLAYERS if p["id"] == player_id), None)
    if not player:
        return "Player not found", 404

    # Fetch upcoming fixtures for the player
    fixtures_url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = requests.get(fixtures_url)

    if response.status_code == 200:
        fixtures_data = response.json().get("fixtures", [])
        fixtures = [
            {
                "event_name": f"GW {fixture['event']}",
                "opponent_name": (
                    TEAMS.get(fixture["team_a"], "Unknown Team") if fixture["team_h"] == player["team"]
                    else TEAMS.get(fixture["team_h"], "Unknown Team")
                ),
                "difficulty": fixture["difficulty"],
            }
            for fixture in fixtures_data
        ]
    else:
        fixtures = []

    return render_template("player.html", player=player, fixtures=fixtures)


@app.route('/gameweeks_with_games')
def gameweeks_with_games():
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    
    bootstrap_response = requests.get(bootstrap_url).json()
    fixtures_response = requests.get(fixtures_url).json()

    team_names = {team['id']: team['name'] for team in bootstrap_response['teams']}

    gameweeks = {}
    for event in bootstrap_response['events']:
        gameweeks[event['id']] = {
            'name': event['name'],
            'deadline': event['deadline_time'],
            'is_current': event.get('is_current', False),
            'is_next': event.get('is_next', False),
            'is_finished': event.get('is_finished', False),
            'fixtures': []
        }

    for fixture in fixtures_response:
        gameweek_id = fixture.get('event')
        if gameweek_id and gameweek_id in gameweeks:
            gameweeks[gameweek_id]['fixtures'].append({
                'kickoff_time': fixture['kickoff_time'],
                'home_team': team_names.get(fixture['team_h'], "Unknown Team"),
                'away_team': team_names.get(fixture['team_a'], "Unknown Team"),
                'home_score': fixture.get('team_h_score'),
                'away_score': fixture.get('team_a_score')
            })

    return render_template('gameweeks.html', gameweeks=gameweeks.values())

    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    
    bootstrap_response = requests.get(bootstrap_url).json()
    fixtures_response = requests.get(fixtures_url).json()

    team_names = {team['id']: team['name'] for team in bootstrap_response['teams']}

    gameweeks = {}
    for event in bootstrap_response['events']:
        gameweeks[event['id']] = {
            'name': event['name'],
            'deadline': event['deadline_time'],
            'is_current': event.get('is_current', False),
            'is_next': event.get('is_next', False),
            'is_finished': event.get('is_finished', False),
            'fixtures': []
        }

    for fixture in fixtures_response:
        gameweek_id = fixture.get('event')
        if gameweek_id and gameweek_id in gameweeks:
            gameweeks[gameweek_id]['fixtures'].append({
                'kickoff_time': fixture['kickoff_time'],
                'home_team': team_names.get(fixture['team_h'], "Unknown Team"),
                'away_team': team_names.get(fixture['team_a'], "Unknown Team"),
                'home_score': fixture.get('team_h_score'),
                'away_score': fixture.get('team_a_score')
            })
    return render_template('gameweeks.html', gameweeks=gameweeks.values())
# Initialize player data
fetch_players()

def train_model(players):
    df = pd.DataFrame(players)
    X = df[["now_cost", "total_points", "form", "minutes"]]
    y = df["total_points"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, X.columns)])
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", BayesianRidge())])

    model.fit(X, y)
    return model

# Predict player performance
def predict_player_performance(model, players):
    df = pd.DataFrame(players)
    X = df[["now_cost", "total_points", "form", "minutes"]]
    
    # BayesianRidge: Predict mean and standard deviation
    predicted_mean, predicted_std = model.predict(X, return_std=True)
    
    for idx, player in enumerate(players):
        player["predicted_points"] = predicted_mean[idx] / 10
        player["uncertainty"] = predicted_std[idx]  # Adding uncertainty
    return players

from pulp import LpMaximize, LpProblem, LpVariable, lpSum

def preprocess_players(players):
    for player in players:
        player['predicted_points'] = float(player.get('predicted_points', 0))  # Default to 0 if missing
        player['form'] = float(player.get('form', 0))  # Default to 0 if missing
        player['now_cost'] = float(player.get('now_cost', 0))  # Default to 0 if missing

def optimize_squad(players, budget=100):
    preprocess_players(players)
    num_players = len(players)

    # Create LP problem
    problem = LpProblem("FantasyFootballSquad", LpMaximize)

    # Define decision variables (binary: 0 or 1)
    decision_vars = [LpVariable(f"x{i}", cat="Binary") for i in range(num_players)]

    # Objective function: maximize weighted score
    predicted_points = [player['predicted_points'] for player in players]
    form = [player['form'] for player in players]
    value_for_money = [player['predicted_points'] / (player['now_cost'] / 10) for player in players]

    problem += lpSum(
        decision_vars[i] * (
            0.6 * predicted_points[i] + 0.3 * form[i] + 0.1 * value_for_money[i]
        )
        for i in range(num_players)
    )

    # Constraints: Budget, position limits, squad size
    costs = [player['now_cost'] / 10 for player in players]
    problem += lpSum([decision_vars[i] * costs[i] for i in range(num_players)]) <= budget

    # Position constraints
    position_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    positions = [player['position'] for player in players]
    for pos, limit in position_limits.items():
        problem += lpSum([decision_vars[i] for i in range(num_players) if positions[i] == pos]) == limit

    # Total squad size
    problem += lpSum(decision_vars) == 15

    # Team constraint: No more than 3 players from the same team
    team_names = [player['team_name'] for player in players]
    unique_teams = set(team_names)
    for team in unique_teams:
        problem += lpSum([decision_vars[i] for i in range(num_players) if team_names[i] == team]) <= 3

    # Solve the problem
    problem.solve()

    # Extract selected players
    selected_indices = [i for i in range(num_players) if decision_vars[i].varValue == 1]
    squad = [players[i] for i in selected_indices]

    total_cost = sum(player['now_cost'] / 10 for player in squad)
    total_predicted_points = sum(player['predicted_points'] for player in squad)

    return squad, total_cost, total_predicted_points

@app.route("/predict", methods=["GET"])
def predict_best_squad():
    budget = 100  # Squad budget constraint
    model = train_model(PLAYERS)  # Train Bayesian Ridge model
    players_with_predictions = predict_player_performance(model, PLAYERS)  # Add predictions with uncertainty
    squad, total_cost, predicted_points = optimize_squad(players_with_predictions, budget=budget)  # Optimize
    return render_template("predict.html", squad=squad, total_cost=total_cost, predicted_points=predicted_points)


# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["fantasy_football"]
saved_teams_collection = db["saved_teams"]

# Save a squad to MongoDB
@app.route('/save_squad', methods=['POST'])
def save_squad():
    data = request.get_json()
    squad = data.get("squad")
    squad_name = data.get("squad_name")

    if not squad_name:
        return jsonify({"success": False, "message": "Squad name is required"}), 400

    # Calculate the total cost of the squad
    total_cost = sum(player['now_cost'] for player in squad) / 10  # converting to millions

    # Get the current date and time for the save timestamp
    date_saved = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if a similar team is already stored
    if saved_teams_collection.find_one({"squad_name": squad_name}):
        return jsonify({"success": False, "message": "Squad name already exists"})

    # Save the team in MongoDB with the date, name, and total cost
    saved_teams_collection.insert_one({
        "squad_name": squad_name,
        "squad": squad,
        "date_saved": date_saved,
        "total_cost": total_cost
    })
    return jsonify({"success": True, "message": "Squad saved successfully!"})


# View saved squads page
@app.route('/saved_squads', methods=['GET'])
def saved_squads_page():
    return render_template('saved_squads.html')
@app.route('/api/delete_squad/<squad_id>', methods=['DELETE'])
def delete_squad(squad_id):
    # Try to find the squad by its ID
    squad = saved_teams_collection.find_one({"_id": ObjectId(squad_id)})

    if not squad:
        return jsonify({"success": False, "message": "Squad not found"}), 404

    # Delete the squad from the database
    saved_teams_collection.delete_one({"_id": ObjectId(squad_id)})

    return jsonify({"success": True, "message": "Squad deleted successfully!"})

# API to fetch saved squads
@app.route('/api/view_saved_squads', methods=['GET'])
def api_view_saved_squads():
    saved_squads = list(saved_teams_collection.find({}, {"_id": 1, "squad_name": 1, "squad": 1, "date_saved": 1, "total_cost": 1}))
    for squad in saved_squads:
        squad["_id"] = str(squad["_id"])  # Convert ObjectId to string
    return jsonify(saved_squads)


# API to fetch players of a specific squad
@app.route('/api/squad_players/<squad_id>', methods=['GET'])
def get_squad_players(squad_id):
    try:
        # Find the squad in MongoDB by its ObjectId
        squad = saved_teams_collection.find_one({'_id': ObjectId(squad_id)})
        if not squad:
            return jsonify({"error": "Squad not found"}), 404

        # Extract players from the squad
        players = squad.get('squad', [])
        if not players:
            return jsonify({"error": "No players found in this squad"}), 404

        # Return the full squad data as expected by the frontend
        return jsonify({"squad": players})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == "__main__":
    fetch_players()
    app.run(debug=True, port=8080)
