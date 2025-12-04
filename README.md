# EDA Final Project Group 5
This is a final project for Exploratory Data Analysis Course(DS5610).


## Setup and Installation
First setup a Python virtual environment using the following commands, after cloning the repository.

```sh
$ git clone https://github.com:harrisonsrubin/eda_final_project.git
$ cd eda_final_project
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ pip install -r requirements.txt
(.venv) $ jupyter lab
```

This opens the following [notebook](./eda_final_project_code_group_5.ipynb)

To format the code better, we have added pre-commit hooks. As of now, nothing is done for the Jupyter notebooks. To enable `pre-commit`, use the following commands.

```sh
$ pre-commit install
```

# Enhanced Team Metadata
A small helper module called `transfer_viz` is created for the purposes of organizing and getting code for the teams metadata and the plan is to add further visualization utils.

## Progress as of 2025-Nov-15
1. Parsed the teams data to include location. The file is saved [here]('./data/raw/team_info/teams_details.csv').

```sh
$ export CFBD_ACCESS_TOKEN=YOUR_ACCESS_TOKEN
$ python -m transfer_viz teams
```

This rewrites the csv file.

## Visualizing HTML Diagrams
To visualize the html diagrams:

```sh
$ python -m http.server 8000
```

Then open the browser at `http://localhost:8000/data/plots` to see the html files.

# To Get Done

Visualize Transfers out vs Coaching Change
Visualize Transfers in vs Coaching Change
Visualize Transfers out vs losses
Visualize Transfers in vs losses
Visualize Transfers out vs (Difference between Expected Wins and Actual)
Visualize Transfers in vs (Difference between Expected Wins and Actual)
Visualize Transfers out vs FPI
Visualize Transfers in vs FPI
Visualize Transfers out vs (Difference between Past Season FPI and Current Season FPI)
Visualize Transfers in vs (Difference between Past Season FPI and Current Season FPI)

Network Graph between Each conference Transfers (G6, ACC, Big 10, Big 12, SEC)
Map and Network Graph between interconference Transfers (SEC)
Average Recruiting Rankings between each team in the
