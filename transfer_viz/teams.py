import cfbd
import pandas as pd
import folium

from transfer_viz.utils import load_dataset


class TeamsParser:
    def __init__(self, url: str, bearer_token: str):
        configuration = cfbd.Configuration()
        configuration.access_token = bearer_token
        configuration.host = url
        self.api_instance = cfbd.TeamsApi(cfbd.ApiClient(configuration))

    def parse(self) -> pd.DataFrame:
        """Parse and return the list of teams from the API.

        Returns:
            pd.DataFrame: DataFrame containing team information.
        """
        try:
            teams = self.api_instance.get_teams()
            teams_data = []
            for team in teams:
                team_info = {
                    "id": team.id,
                    "school": team.school,
                    "color": team.color,
                    "alternate_color": team.alternate_color,
                    "alternate_names": ",".join(team.alternate_names)
                    if team.alternate_names
                    else None,
                    "abbreviation": team.abbreviation,
                    "conference": team.conference,
                    "division": team.division,
                    "logos": ",".join(team.logos) if team.logos else None,
                    "location_id": team.location.id if team.location else None,
                    "location_city": team.location.city if team.location else None,
                    "location_state": team.location.state if team.location else None,
                    "location_zip": team.location.zip if team.location else None,
                    "location_country": team.location.country_code
                    if team.location
                    else None,
                    "location_latitude": team.location.latitude
                    if team.location
                    else None,
                    "location_longitude": team.location.longitude
                    if team.location
                    else None,
                    "stadium_capacity": team.location.capacity
                    if team.location
                    else None,
                    "location_elevation": team.location.elevation
                    if team.location
                    else None,
                    "location_timezone": team.location.timezone
                    if team.location
                    else None,
                    "grass": team.location.grass if team.location else None,
                    "dome": team.location.dome if team.location else None,
                }
                teams_data.append(team_info)
            return pd.DataFrame(teams_data)
        except cfbd.ApiException as e:
            raise e


def parse_teams(url: str, bearer_token: str) -> pd.DataFrame:
    """Parse and return the list of teams from the API."""
    parser = TeamsParser(url=url, bearer_token=bearer_token)

    return parser.parse()


def filter_teams() -> pd.DataFrame:
    """Filter teams details based on team metadata to save processed teams dataset."""
    teams_details_df = load_dataset("raw_teams_details")
    team_metadata_df = load_dataset("team_metadata")
    filtered_details_df = teams_details_df[
        teams_details_df["id"].isin(team_metadata_df["TeamId"])
    ]

    return filtered_details_df


def teams_map():
    teams_details_df = load_dataset("teams_details")
    map_center = [39.8283, -98.5795]  # Center of the US
    teams_map = folium.Map(location=map_center, zoom_start=4)

    for _, row in teams_details_df.iterrows():
        if pd.notnull(row["location_latitude"]) and pd.notnull(
            row["location_longitude"]
        ):
            folium.Marker(
                location=[row["location_latitude"], row["location_longitude"]],
                popup=f"{row['school']} ({row['abbreviation']})",
                icon=folium.Icon(
                    color="white", icon="info-sign", icon_color=row["color"]
                ),
            ).add_to(teams_map)

    return teams_map


def run(args=None):
    import os
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from pathlib import Path

    parser = ArgumentParser(
        description="Fetch and display FBS teams from the College Football Data API.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="https://api.collegefootballdata.com",
        help="Base URL for the College Football Data API",
    )

    parser.add_argument(
        "--access-token",
        type=str,
        default=os.environ.get("CFBD_ACCESS_TOKEN", ""),
        help="Access token for the College Football Data API",
    )

    parser.add_argument(
        "command",
        metavar="COMMAND",
        help=", ".join(sorted(["visualize", "filter", "parse", "teams_map"])),
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "visualize":
        print("Visualization functionality is not implemented yet.")
    elif parsed_args.command == "filter":
        filtered_teams_df = filter_teams()

        dataset_path = (
            Path(__file__).parent.parent / "data" / "processed" / "teams_details.csv"
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_teams_df.to_csv(dataset_path, index=False)
        print(f"Filtered teams data saved to {dataset_path}")

    elif parsed_args.command == "teams_map":
        teams_map_obj = teams_map()
        map_path = (
            Path(__file__).parent.parent / "data" / "processed" / "teams_map.html"
        )
        map_path.parent.mkdir(parents=True, exist_ok=True)
        teams_map_obj.save(str(map_path))
        print(f"Teams map saved to {map_path}")

    elif parsed_args.command == "parse":
        teams_df = parse_teams(
            url=parsed_args.api_url, bearer_token=parsed_args.access_token
        )

        dataset_path = (
            Path(__file__).parent.parent
            / "data"
            / "raw"
            / "team_info"
            / "raw_teams_details.csv"
        )
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        teams_df.to_csv(dataset_path, index=False)
        print(f"Teams data saved to {dataset_path}")
