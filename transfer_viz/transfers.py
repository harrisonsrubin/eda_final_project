from typing import Optional
from transfer_viz.utils import load_dataset
import pandas as pd


class TransfersExplorer:
    def __init__(self):
        self.transfers_data = load_dataset("transfer_data")
        self.teams_metadata = load_dataset("team_metadata")
        self.teams_details = load_dataset("teams_details")

    def get_transfers_by_year(self, season: int) -> pd.DataFrame:
        """Get transfer data for a specific year.

        Args:
            season (int): The season year to filter data.
        """
        if season not in self.transfers_data["Season"].unique():
            raise ValueError(f"No transfer data available for the Season {season}.")
        return self.transfers_data[self.transfers_data["Season"] == season]

    def summarize_transfers_by_position(
        self, season: Optional[int] = None
    ) -> pd.DataFrame:
        """Summarize transfers by player position.

        Args:
            season (Optional[int], optional): Specific season to filter data. Defaults to None.
        """
        data = self.transfers_data
        if season is not None:
            data = data[data["Season"] == season]
        summary = data.groupby("Position").size().reset_index(name="transfer_count")
        return summary.sort_values(by="transfer_count", ascending=False)

    def summarize_transfers_in_by_team(
        self, season: Optional[int] = None
    ) -> pd.DataFrame:
        """Summarize transfers by team.

        Args:
            season (Optional[int], optional): Specific season to filter data. Defaults to None.
        """
        data = self.transfers_data
        if season is not None:
            data = data[data["Season"] == season]
        summary = data.groupby("Destination").size().reset_index(name="transfer_in")
        return summary.sort_values(by="transfer_in", ascending=False)


def run(args=None):
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Exploration of the transfer dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        type=str,
        metavar="COMMAND",
        help=", ".join(sorted(["plot_top_positions", "plot_top_teams_transfers_in"])),
    )

    parser.add_argument(
        "--season",
        type=int,
        help="Season year to filter data for certain commands",
        default=None,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        help="Number of top positions to plot, only used with plot_top_positions command",
        default=10,
    )

    parsed_args = parser.parse_args(args)

    explorer = TransfersExplorer()
    if parsed_args.command == "plot_top_positions":
        summary_df = explorer.summarize_transfers_by_position(season=parsed_args.season)
        top_n_df = summary_df.head(parsed_args.top_n)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.bar(top_n_df["Position"], top_n_df["transfer_count"], color="skyblue")
        plt.xlabel("Player Position")
        plt.ylabel("Number of Transfers")
        plt.title(
            f"Top {parsed_args.top_n} Transfer Positions"
            + (f" in {parsed_args.season}" if parsed_args.season else "")
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    elif parsed_args.command == "plot_top_teams_transfers_in":
        summary_df = explorer.summarize_transfers_in_by_team(season=parsed_args.season)
        top_n_df = summary_df.head(parsed_args.top_n)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        # Plot number on top of bars
        bars = plt.bar(
            top_n_df["Destination"],
            top_n_df["transfer_in"],
            color="salmon",
        )
        plt.bar_label(bars, fmt="%.0f")
        plt.xlabel("Team")
        plt.ylabel("Number of Transfers")
        plt.title(
            f"Top {parsed_args.top_n} Teams by Transfer In"
            + (f" in {parsed_args.season}" if parsed_args.season else "")
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        parser.print_help()
