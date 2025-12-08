from typing import Literal
import pandas as pd
from pathlib import Path
from folium.features import CustomIcon


def load_dataset(
    name: Literal[
        "team_metadata",
        "transfer_data",
        "teams_details",
        "raw_teams_details",
        "2022_FPI",
        "2023_FPI",
        "2024_FPI",
        "2022_team_records",
        "2023_team_records",
        "2024_team_records",
        "coaches",
    ],
) -> pd.DataFrame:
    """Load dataset by name from the data/raw directory."""
    raw_processed_dir = {
        "team_metadata": "processed",
        "transfer_data": "processed",
        "teams_details": "processed",
        "raw_teams_details": "raw/team_info",
        "2022_FPI": "raw/team_info",
        "2023_FPI": "raw/team_info",
        "2024_FPI": "raw/team_info",
        "2022_team_records": "raw/team_info",
        "2023_team_records": "raw/team_info",
        "2024_team_records": "raw/team_info",
        "coaches": "raw/team_info",
        "2022_transfers": "raw/transfer_portal_info",
        "2023_transfers": "raw/transfer_portal_info",
        "2024_transfers": "raw/transfer_portal_info",
    }

    location = raw_processed_dir.get(name, "raw")

    dataset_path = Path(__file__).parent.parent / "data" / location / f"{name}.csv"
    try:
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset {name} not found at {dataset_path}. Please ensure the file exists and run any necessary data preparation scripts."
        )
    return dataset


def custom_icon(
    icon_image: str,
    icon_size: tuple[int, int] = (30, 30),
    icon_anchor: tuple[int, int] = (15, 15),
) -> CustomIcon:
    """Create a custom DivIcon for folium markers."""
    return CustomIcon(
        icon_image=icon_image,
        icon_size=icon_size,
        icon_anchor=icon_anchor,
    )
