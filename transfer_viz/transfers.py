from typing import Optional
from transfer_viz.utils import load_dataset
import pandas as pd

import networkx as nx
from matplotlib import pyplot as plt
import plotly.graph_objects as go


def power_5_shankey_diagram(
    year: int,
):
    """Create a Sankey diagram of transfers between Power 5 conferences for a specific year.

    Args:
        year (int): The season year to create the diagram.
    """
    explorer = TransfersExplorer()
    df = explorer.power_5_transfer_network_df(year)

    all_conferences = list(
        set(df["Origin_Conference"].unique())
        | set(df["Destination_Conference"].unique())
    )
    conf_to_idx = {conf: idx for idx, conf in enumerate(all_conferences)}

    source = [conf_to_idx[row["Origin_Conference"]] for _, row in df.iterrows()]
    target = [conf_to_idx[row["Destination_Conference"]] for _, row in df.iterrows()]
    value = df["Transfer_Count"].tolist()

    conference_colors = {
        "SEC": "rgba(255, 0, 0, 0.8)",
        "Big Ten": "rgba(0, 0, 255, 0.8)",
        "Big 12": "rgba(255, 165, 0, 0.8)",
        "ACC": "rgba(0, 128, 0, 0.8)",
        "Pac-12": "rgba(128, 0, 128, 0.8)",
    }
    labels = all_conferences
    node_colors = [
        conference_colors.get(conf, "rgba(128, 128, 128, 0.8)")
        for conf in all_conferences
    ]

    link_colors = []
    for t in target:
        nc = node_colors[t]
        if isinstance(nc, str) and nc.startswith("rgba"):
            base = nc.rsplit(",", 1)[0]
            link_colors.append(f"{base}, 0.4)")
        else:
            link_colors.append("rgba(192,192,192,0.4)")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=10,
                    thickness=50,
                    line=dict(color="black", width=2),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Power 5 Conference Transfer Flows after the {year} Season",
        font=dict(size=14),
        height=600,
        width=1200,
    )

    fig.write_html(f"data/plots/power_5_sankey_{year}.html")
    return fig


def power_5_transfer_network_graph(year: int):
    """Create a directed graph of transfers between Power 5 conferences for a specific year.

    Args:
        year (int): The season year to create the graph.
    """
    explorer = TransfersExplorer()
    df = explorer.power_5_transfer_network_df(year)

    G = nx.DiGraph()

    for _, row in df.iterrows():
        origin = row["Origin_Conference"]
        destination = row["Destination_Conference"]
        weight = row["Transfer_Count"]
        avg_stars = row["Avg_Stars"]
        avg_rating = row["Avg_Rating"]

        G.add_edge(
            origin,
            destination,
            weight=weight,
            avg_stars=avg_stars,
            avg_rating=avg_rating,
        )

    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    plt.figure(figsize=(16, 12))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#3498db",
        node_size=4000,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
    )

    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]

    for u, v in edges:
        weight = G[u][v]["weight"]

        edge_color = plt.cm.Reds(weight / max(weights))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=weight / 2,
            alpha=0.8,
            arrows=True,
            arrowsize=25,
            arrowstyle="-|>",
            edge_color=[edge_color],
            connectionstyle="arc3,rad=0.1",
            min_source_margin=20,
            min_target_margin=20,
        )

    nx.draw_networkx_labels(
        G, pos, font_size=11, font_weight="bold", font_color="white"
    )

    edge_labels = {(u, v): f"→ {G[u][v]['weight']}" for u, v in edges}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels,
        font_size=9,
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    plt.title(
        f"Power 5 Transfer Network after {year} Season \n(Arrow shows direction: Origin → Destination)",
        fontsize=16,
        fontweight="bold",
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"data/plots/power_5_transfer_network_{year}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: power_5_transfer_network_{year}.png")


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

    def power_5_transfer_network_df(self, year: int) -> pd.DataFrame:
        """Summarize transfers by conference for a specific year.

        Args:
            year (int): The season year to summarize transfers.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - Origin_Conference: Conference player transferred from
                - Destination_Conference: Conference player transferred to
                - Transfer_Count: Number of transfers between these conferences
                - Avg_Stars: Average star rating of transfers (if available)
                - Avg_Rating: Average recruiting rating of transfers (if available)
        """
        data = self.get_transfers_by_year(year + 1)

        team_conf = self.teams_metadata[self.teams_metadata["Year"] == year][
            ["Team", "Conference"]
        ].copy()

        data_with_conf = data.merge(
            team_conf, left_on="Origin", right_on="Team", how="left"
        ).rename(columns={"Conference": "Origin_Conference"})

        data_with_conf = data_with_conf.merge(
            team_conf,
            left_on="Destination",
            right_on="Team",
            how="left",
            suffixes=("_origin", "_destination"),
        ).rename(columns={"Conference": "Destination_Conference"})

        data_with_conf = data_with_conf.dropna(
            subset=["Origin_Conference", "Destination_Conference"]
        )

        summary = (
            data_with_conf.groupby(["Origin_Conference", "Destination_Conference"])
            .agg(
                Transfer_Count=("Origin", "size"),
                Avg_Stars=("Stars", "mean"),
                Avg_Rating=("Rating", "mean"),
            )
            .reset_index()
        )

        summary = summary.sort_values("Transfer_Count", ascending=False)
        power_5 = [
            "SEC",
            "Big Ten",
            "Big 12",
            "Pac-12",
            "ACC",
        ]

        power_5_transfers = summary[
            (summary["Origin_Conference"].isin(power_5))
            & (summary["Destination_Conference"].isin(power_5))
        ]

        return power_5_transfers


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
        help=", ".join(
            sorted(
                [
                    "plot_top_positions",
                    "plot_top_teams_transfers_in",
                    "power_5_transfer_network",
                    "power_5_shankey_diagram",
                ]
            )
        ),
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
    elif parsed_args.command == "power_5_transfer_network":
        if parsed_args.season is None:
            raise ValueError(
                "Please provide a season year using --season for this command."
            )
        power_5_transfer_network_graph(parsed_args.season)
        print(f"Power 5 transfer network graph for {parsed_args.season} saved as PNG.")
    elif parsed_args.command == "power_5_shankey_diagram":
        if parsed_args.season is None:
            raise ValueError(
                "Please provide a season year using --season for this command."
            )
        power_5_shankey_diagram(parsed_args.season)
        print(
            f"Power 5 transfer Sankey diagram for {parsed_args.season} saved as HTML."
        )
    else:
        parser.print_help()
