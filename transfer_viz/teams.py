import cfbd
import pandas as pd


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
            teams = self.api_instance.get_fbs_teams()
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

    parsed_args = parser.parse_args(args)

    teams_df = parse_teams(
        url=parsed_args.api_url, bearer_token=parsed_args.access_token
    )

    dataset_path = (
        Path(__file__).parent.parent
        / "data"
        / "raw"
        / "team_info"
        / "teams_details.csv"
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    teams_df.to_csv(dataset_path, index=False)
    print(f"Teams data saved to {dataset_path}")
