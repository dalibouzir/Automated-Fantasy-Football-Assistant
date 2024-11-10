from flask import Flask, render_template_string
import requests

app = Flask(__name__)

@app.route('/')
def index():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    categorized_players = {
        'goalkeepers': [],
        'defenders': [],
        'midfielders': [],
        'forwards': []
    }

    for player in data['elements']:
        player_data = {
            'name': player['web_name'],
            'cost': player['now_cost'] / 10,  # Fantasy value in millions
            'points': player['total_points'],
            'goals': player['goals_scored'],
            'assists': player['assists'],
            'selected_percent': player['selected_by_percent']
        }

        if player['element_type'] == 1:
            categorized_players['goalkeepers'].append(player_data)
        elif player['element_type'] == 2:
            categorized_players['defenders'].append(player_data)
        elif player['element_type'] == 3:
            categorized_players['midfielders'].append(player_data)
        elif player['element_type'] == 4:
            categorized_players['forwards'].append(player_data)
    
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Automated Fantasy Football Assistant</title>

    <style>
      body {
        background: linear-gradient(to right, #00a2ff, #8c00ff);
        font-family: Arial, sans-serif;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: flex-start;
        height: 100vh;
        overflow: hidden;
      }
      .logo {
        height: 40px;
      }
      .container {
        width: 30%;
        min-width: 300px;
        height: 100vh;
        padding: 20px;
        margin-left: auto;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 10px 0 0 10px;
        box-shadow: -2px 0 10px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        overflow-y: auto;
      }

      h1 {
        text-align: center;
        color: #ff0073;
        margin-bottom: 20px;
      }

      .category {
        margin-bottom: 40px;
      }

      .category-title {
        font-size: 1.5em;
        color: #ff0073;
        margin-bottom: 10px;
      }

      table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 15px;
        color: #fff;
        font-size: 0.9em;
      }

      th,
      td {
        padding: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        text-align: left;
      }

      th {
        background-color: #ff0073;
      }

      .sort-buttons {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
      }

      .sort-button {
        background-color: #28a745;
        color: white;
        padding: 8px 12px;
        border: none;
        cursor: pointer;
        border-radius: 5px;
        font-size: 1em;
        margin: 0 10px;
      }

      .pagination {
        display: flex;
        justify-content: center;
        margin-top: 10px;
      }

      .pagination button {
        background-color: #ff0073;
        color: white;
        border: none;
        padding: 8px 15px;
        margin: 0 5px;
        cursor: pointer;
        border-radius: 5px;
      }

      .pagination button.disabled {
        background-color: #555;
        cursor: not-allowed;
      }
    </style>

    <script>
      let playersData = {{ players | tojson }};
      let currentPage = 1;
      const itemsPerPage = 5;

      function paginate(array, page) {
          const start = (page - 1) * itemsPerPage;
          const end = start + itemsPerPage;
          return array.slice(start, end);
      }

      function renderTable(players, categoryId) {
          const tableBody = document.getElementById(`table-body-${categoryId}`);
          tableBody.innerHTML = '';
          players.forEach(player => {
              const row = `<tr>
                  <td>${player.name}</td>
                  <td>£${player.cost}m</td>
                  <td>${player.points}</td>
                  <td>${player.goals}</td>
                  <td>${player.assists}</td>
                  <td>${player.selected_percent}%</td>
              </tr>`;
              tableBody.innerHTML += row;
          });
      }

      function updatePaginationButtons(totalItems, categoryId) {
          const totalPages = Math.ceil(totalItems / itemsPerPage);
          document.getElementById(`prev-${categoryId}`).disabled = currentPage === 1;
          document.getElementById(`next-${categoryId}`).disabled = currentPage === totalPages;
      }

      function changePage(page, players, categoryId) {
          currentPage = page;
          const paginatedPlayers = paginate(players, currentPage);
          renderTable(paginatedPlayers, categoryId);
          updatePaginationButtons(players.length, categoryId);
      }

      function sortPlayersBy(property, category, categoryId) {
          category.sort((a, b) => b[property] - a[property]);
          changePage(1, category, categoryId);
      }

      document.addEventListener('DOMContentLoaded', () => {
          const categories = ['goalkeepers', 'defenders', 'midfielders', 'forwards'];

          categories.forEach(categoryId => {
              const players = playersData[categoryId];
              changePage(1, players, categoryId);

              document.getElementById(`sort-points-${categoryId}`).addEventListener('click', () => sortPlayersBy('points', players, categoryId));
              document.getElementById(`sort-cost-${categoryId}`).addEventListener('click', () => sortPlayersBy('cost', players, categoryId));
          });
      });
    </script>
  </head>
  <body>
    <div class="container">
      <h1>Fantasy Premier League Player Stats</h1>

      {% for category, players in players.items() %}
      <div class="category" id="{{ category }}">
        <div class="category-title">{{ category | capitalize }}</div>

        <div class="sort-buttons">
          <button id="sort-points-{{ category }}" class="sort-button">
            Sort by Points
          </button>
          <button id="sort-cost-{{ category }}" class="sort-button">
            Sort by Cost
          </button>
        </div>

        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Cost (£m)</th>
              <th>Points</th>
              <th>Goals</th>
              <th>Assists</th>
              <th>Selected By (%)</th>
            </tr>
          </thead>
          <tbody id="table-body-{{ category }}">
          </tbody>
        </table>

        <div class="pagination">
          <button
            id="prev-{{ category }}"
            onclick="changePage(currentPage - 1, playersData['{{ category }}'], '{{ category }}')"
          >
            Prev
          </button>
          <button
            id="next-{{ category }}"
            onclick="changePage(currentPage + 1, playersData['{{ category }}'], '{{ category }}')"
          >
            Next
          </button>
        </div>
      </div>
      {% endfor %}
    </div>
  </body>
</html>
''', players=categorized_players)

if __name__ == '__main__':
    app.run(debug=True)
