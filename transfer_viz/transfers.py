from typing import Optional
from transfer_viz.utils import load_dataset
import pandas as pd

import networkx as nx
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import folium
from transfer_viz.utils import custom_icon
from networkx.algorithms.community import louvain_communities
import numpy as np



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

def sec_transfers_map(season: int) -> pd.DataFrame:
    """Create a summary of transfers involving SEC teams."""
    explorer = TransfersExplorer()
    sec_summary_df = explorer.summarize_sec_transfers()
    season_summary = sec_summary_df[sec_summary_df['Season'] == season]

    sec_schools = explorer.teams_metadata[
          (explorer.teams_metadata['Year'] == season) &
          (explorer.teams_metadata['Conference'] == 'SEC')
    ]['Team'].unique()

    sec_to_sec = season_summary[
        (season_summary['Origin'].isin(sec_schools)) &
        (season_summary['Destination'].isin(sec_schools))
    ]


    origin_coords = explorer.teams_details[['school', 'location_latitude', 'location_longitude', 'logos']].copy()
    origin_coords.columns = ['Origin', 'origin_lat', 'origin_lon', 'origin_logos']

    dest_coords = explorer.teams_details[['school', 'location_latitude', 'location_longitude', 'logos']].copy()
    dest_coords.columns = ['Destination', 'dest_lat', 'dest_lon', 'dest_logos']

    flows_with_coords = sec_to_sec.merge(origin_coords, on='Origin', how='left')
    flows_with_coords = flows_with_coords.merge(dest_coords, on='Destination', how='left')
    print(flows_with_coords.head(5))
    return flows_with_coords


def create_sec_transfers_map(season: int, save_path: Optional[str] = None) -> Optional[folium.Map]:
    """Create an interactive folium map showing SEC transfer flows.
    
    Args:
        season (int): The season year to visualize transfers
        save_path (str, optional): Path to save the HTML map file
    
    Returns:
        folium.Map: Interactive map with transfer flows
    """
    flows_df = sec_transfers_map(season)
    
    flows_df = flows_df.dropna(subset=['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'])
    
    if flows_df.empty:
        print(f"No SEC transfer data with coordinates found for season {season}")
        return None
    
    center_lat = flows_df[['origin_lat', 'dest_lat']].values.flatten().mean()
    center_lon = flows_df[['origin_lon', 'dest_lon']].values.flatten().mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    origin_schools = flows_df[['Origin', 'origin_lat', 'origin_lon', 'origin_logos']].drop_duplicates()
    dest_schools = flows_df[['Destination', 'dest_lat', 'dest_lon', 'dest_logos']].drop_duplicates()
    
    all_schools = pd.concat([
        origin_schools.rename(columns={'Origin': 'School', 'origin_lat': 'lat', 'origin_lon': 'lon', 'origin_logos': 'logos'}),
        dest_schools.rename(columns={'Destination': 'School', 'dest_lat': 'lat', 'dest_lon': 'lon', 'dest_logos': 'logos'})
    ]).drop_duplicates()
    
    for _, school in all_schools.iterrows():
        transfers_in = flows_df[flows_df['Destination'] == school['School']]['No_of_Transfers'].sum()
        transfers_out = flows_df[flows_df['Origin'] == school['School']]['No_of_Transfers'].sum()
        
        popup_text = f"""
        <b>{school['School']}</b><br>
        Transfers In: {transfers_in}<br>
        Transfers Out: {transfers_out}<br>
        Net Transfers: {transfers_in - transfers_out}
        """
        
        if pd.notna(school['logos']) and school['logos']:
            logo_url = school['logos'].split(',')[0]  # Use first logo
            icon = folium.CustomIcon(
                icon_image=logo_url,
                icon_size=(50, 50),
                icon_anchor=(15, 15)
            )
        else:
            icon = folium.Icon(color='blue', icon='graduation-cap', prefix='fa')
        
        folium.Marker(
            location=[school['lat'], school['lon']],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=school['School'],
            icon=icon
        ).add_to(m)
    
    flow_pairs = {}
    for _, flow in flows_df.iterrows():
        if flow['No_of_Transfers'] > 0:
            schools = sorted([flow['Origin'], flow['Destination']])
            pair_key = f"{schools[0]}_{schools[1]}"
            
            if pair_key not in flow_pairs:
                flow_pairs[pair_key] = []
            flow_pairs[pair_key].append(flow)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, (pair_key, flows) in enumerate(flow_pairs.items()):
        base_color = colors[i % len(colors)]
        
        for j, flow in enumerate(flows):
            weight = max(2, min(flow['No_of_Transfers'] * 2, 8))
            
            offset = 0.02 * (j - len(flows)/2 + 0.5) if len(flows) > 1 else 0
            
            start_lat, start_lon = flow['origin_lat'], flow['origin_lon']
            end_lat, end_lon = flow['dest_lat'], flow['dest_lon']
            
            mid_lat = (start_lat + end_lat) / 2 + offset
            mid_lon = (start_lon + end_lon) / 2 + offset
            
            arc_points = [
                [start_lat, start_lon],
                [mid_lat, mid_lon],
                [end_lat, end_lon]
            ]
            
            folium.PolyLine(
                locations=arc_points,
                color=base_color,
                weight=weight,
                opacity=0.8,
                popup=f"{flow['Origin']} → {flow['Destination']}: {flow['No_of_Transfers']} transfers",
                tooltip=f"{flow['Origin']} → {flow['Destination']}"
            ).add_to(m)
            
            arrow_lat = start_lat + 0.75 * (end_lat - start_lat) + 0.5 * offset
            arrow_lon = start_lon + 0.75 * (end_lon - start_lon) + 0.5 * offset
            
            folium.CircleMarker(
                location=[arrow_lat, arrow_lon],
                radius=3,
                popup=f"Direction: {flow['Origin']} → {flow['Destination']}",
                color=base_color,
                fill=True,
                fillColor=base_color,
                fillOpacity=0.9
            ).add_to(m)
    
    if save_path:
        m.save(save_path)
        print(f"SEC transfers map saved to {save_path}")
    
    return m

def create_transfer_density_map(season: int, transfer_type: str = "destination", save_path: Optional[str] = None) -> Optional[folium.Map]:
    """Create a density heatmap of transfer destinations or origins.
    
    Args:
        season (int): The season year to visualize transfers
        transfer_type (str): Either "destination", "origin", or "both" for transfer density
        save_path (str, optional): Path to save the HTML map file
    
    Returns:
        folium.Map: Interactive map with transfer density heatmap
    """
    from folium.plugins import HeatMap
    
    explorer = TransfersExplorer()
    transfers = explorer.get_transfers_by_year(season + 1)
    
    if transfers.empty:
        print(f"No transfer data found for season {season}")
        return None
    
    team_coords = explorer.teams_details[['school', 'location_latitude', 'location_longitude', 'logos']].copy()
    team_coords = team_coords.dropna(subset=['location_latitude', 'location_longitude'])
    
    if transfer_type == "both":
        origin_counts = transfers["Origin"].value_counts().reset_index()
        origin_counts.columns = ['school', 'origin_count']
        
        dest_counts = transfers["Destination"].value_counts().reset_index()
        dest_counts.columns = ['school', 'dest_count']
        
        combined_counts = origin_counts.merge(dest_counts, on='school', how='outer').fillna(0)
        combined_counts['total_transfers'] = combined_counts['origin_count'] + combined_counts['dest_count']
        
        density_data = combined_counts.merge(team_coords, on='school', how='inner')
        
        if density_data.empty:
            print(f"No coordinate data available for transfers")
            return None
        
        origin_heat_data = []
        dest_heat_data = []
        
        for _, row in density_data.iterrows():
            lat, lon = row['location_latitude'], row['location_longitude']
            
            origin_weight = int(row['origin_count'])
            for _ in range(origin_weight):
                origin_heat_data.append([lat, lon])
                
            dest_weight = int(row['dest_count'])
            for _ in range(dest_weight):
                dest_heat_data.append([lat, lon])
        
        center_lat = density_data['location_latitude'].mean()
        center_lon = density_data['location_longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='CartoDB dark_matter'
        )
        
        if origin_heat_data:
            HeatMap(
                origin_heat_data,
                name='Transfer Origins',
                radius=25,
                blur=15,
                max_zoom=8,
                gradient={
                    0.0: 'lightblue',
                    0.25: 'yellow', 
                    0.5: 'orange',
                    0.75: 'red',
                    1.0: 'darkred'
                }
            ).add_to(m)
        
        if dest_heat_data:
            HeatMap(
                dest_heat_data,
                name='Transfer Destinations',
                radius=25,
                blur=15,
                max_zoom=8,
                gradient={
                    0.0: 'lightcyan',
                    0.25: 'cyan', 
                    0.5: 'blue',
                    0.75: 'darkblue',
                    1.0: 'navy'
                }
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        top_transfers = density_data.nlargest(10, 'total_transfers')
        remaining_transfers = density_data[~density_data['school'].isin(top_transfers['school'])]
        
        for _, school in top_transfers.iterrows():
            popup_text = f"""
            <b>{school['school']}</b><br>
            Origin Transfers: {int(school['origin_count'])}<br>
            Destination Transfers: {int(school['dest_count'])}<br>
            Total Transfers: {int(school['total_transfers'])}
            """
            
            if pd.notna(school['logos']) and school['logos']:
                logo_url = school['logos'].split(',')[0]  # Use first logo
                icon = custom_icon(
                    icon_image=logo_url,
                    icon_size=(40, 40),
                    icon_anchor=(20, 20)
                )
            else:
                icon = folium.Icon(color='purple', icon='graduation-cap', prefix='fa')
            
            folium.Marker(
                location=[school['location_latitude'], school['location_longitude']],
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{school['school']}: {int(school['total_transfers'])} total transfers",
                icon=icon
            ).add_to(m)
        
        for _, school in remaining_transfers.iterrows():
            popup_text = f"""
            <b>{school['school']}</b><br>
            Origin Transfers: {int(school['origin_count'])}<br>
            Destination Transfers: {int(school['dest_count'])}<br>
            Total Transfers: {int(school['total_transfers'])}
            """
            
            if pd.notna(school['logos']) and school['logos']:
                logo_url = school['logos'].split(',')[0]  # Use first logo
                icon = custom_icon(
                    icon_image=logo_url,
                    icon_size=(25, 25),
                    icon_anchor=(12, 12)
                )
            else:
                icon = folium.Icon(color='lightgray', icon='graduation-cap', prefix='fa')
            
            folium.Marker(
                location=[school['location_latitude'], school['location_longitude']],
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{school['school']}: {int(school['total_transfers'])} total transfers",
                icon=icon
            ).add_to(m)
            
    else:
        location_col = "Destination" if transfer_type == "destination" else "Origin"
        
        transfer_counts = transfers[location_col].value_counts().reset_index()
        transfer_counts.columns = ['school', 'transfer_count']
        
        density_data = transfer_counts.merge(team_coords, on='school', how='inner')
        
        if density_data.empty:
            print(f"No coordinate data available for transfers")
            return None
        
        heat_data = []
        for _, row in density_data.iterrows():
            lat, lon = row['location_latitude'], row['location_longitude']
            weight = row['transfer_count']
            for _ in range(weight):
                heat_data.append([lat, lon])
        
        center_lat = density_data['location_latitude'].mean()
        center_lon = density_data['location_longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='CartoDB dark_matter'
        )
        
        HeatMap(
            heat_data,
            radius=25,
            blur=15,
            max_zoom=8,
            gradient={
                0.0: 'lightblue',
                0.25: 'green', 
                0.5: 'yellow',
                0.75: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
        
        top_transfers = density_data.nlargest(10, 'transfer_count')
        remaining_transfers = density_data[~density_data['school'].isin(top_transfers['school'])]
        
        for _, school in top_transfers.iterrows():
            popup_text = f"""
            <b>{school['school']}</b><br>
            {transfer_type.title()} Transfers: {school['transfer_count']}
            """
            
            if pd.notna(school['logos']) and school['logos']:
                logo_url = school['logos'].split(',')[0]  # Use first logo
                icon = custom_icon(
                    icon_image=logo_url,
                    icon_size=(40, 40),
                    icon_anchor=(20, 20)
                )
            else:
                icon = folium.Icon(color='orange', icon='graduation-cap', prefix='fa')
            
            folium.Marker(
                location=[school['location_latitude'], school['location_longitude']],
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{school['school']}: {school['transfer_count']} transfers",
                icon=icon
            ).add_to(m)
        
        for _, school in remaining_transfers.iterrows():
            popup_text = f"""
            <b>{school['school']}</b><br>
            {transfer_type.title()} Transfers: {school['transfer_count']}
            """
            
            if pd.notna(school['logos']) and school['logos']:
                logo_url = school['logos'].split(',')[0]  # Use first logo
                icon = custom_icon(
                    icon_image=logo_url,
                    icon_size=(25, 25),
                    icon_anchor=(12, 12)
                )
            else:
                icon = folium.Icon(color='lightgray', icon='graduation-cap', prefix='fa')
            
            folium.Marker(
                location=[school['location_latitude'], school['location_longitude']],
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{school['school']}: {school['transfer_count']} transfers",
                icon=icon
            ).add_to(m)
    
    if save_path:
        m.save(save_path)
        print(f"Transfer {transfer_type} density map saved to {save_path}")
    
    return m

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
    
    def summarize_sec_transfers(self) -> pd.DataFrame:
        """Summarize transfers involving SEC teams.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - Origin_Team: Team player transferred from
                - Destination_Team: Team player transferred to
                - No_of_Transfers: Number of transfers between these teams
                - Season: Season of the transfers
        """
        data = self.transfers_data
        
        sec_teams = self.teams_metadata[
            self.teams_metadata["Conference"] == "SEC"
        ]["Team"].unique()

        data_sec = data[
            (data["Origin"].isin(sec_teams)) | (data["Destination"].isin(sec_teams))
        ]

        summary = (
            data_sec.groupby(["Origin", "Destination", "Season"])
            .size()
            .reset_index(name="No_of_Transfers")
        )

        summary = summary.sort_values("No_of_Transfers", ascending=False)
        summary['Season'] = summary['Season'] - 1  # Adjust season to reflect the starting year

        return summary
    
    
    def transfers_community_detection(self, season: int):
        """Perform community detection on transfer network for a specific season between conferences"""
        data = self.get_transfers_by_year(season + 1)

        team_conf = self.teams_metadata[self.teams_metadata["Year"] == season][
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

        G = nx.DiGraph()

        for _, row in data_with_conf.iterrows():
            origin = row["Origin_Conference"]
            destination = row["Destination_Conference"]

            if G.has_edge(origin, destination):
                G[origin][destination]["weight"] += 1
            else:
                G.add_edge(origin, destination, weight=1)


        communities = list(louvain_communities(G.to_undirected(), weight='weight', seed=42))
        partitions = {
            member: idx for idx, community in enumerate(communities) for member in community
        }


        nx.set_node_attributes(G, partitions, "group")

       
        supergraph = nx.cycle_graph(len(communities))
        superpos = nx.spring_layout(supergraph, scale=10, seed=429)
        centers = list(superpos.values())

        pos = {}
        for center, community in zip(centers, communities):
            subgraph = nx.subgraph(G, community)
            subpos = nx.spring_layout(subgraph, center=center, seed=1430, scale=1.5)
            pos.update(subpos)

        from matplotlib import cm
        cmap = cm.get_cmap('tab10')
        
        colors = [G.nodes[node]["group"] for node in G.nodes()]
        node_colors = [cmap(color) for color in colors]
        node_sizes = [G.nodes[node].get("size", 2E6) for node in G.nodes()]
        scaled_sizes = [max(size / 125, 150) for size in node_sizes]

        fig, ax = plt.subplots(figsize=(16, 20), facecolor="#ffffff")  # Increased height for legend
        ax.set_facecolor("#ffffff")

        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        edge_weights_normalized = (
            np.array(edge_weights) / max(edge_weights) if edge_weights else [1]
        )

        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.30,
            width=[w * 50 for w in edge_weights_normalized],
            edge_color="#000000",
            style="solid",
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=scaled_sizes,
            node_color=node_colors,  # Use explicit colors instead of cmap
            alpha=0.95,
            edgecolors="black",
            linewidths=2.5,
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=12,
            font_weight="bold",
            font_color="white",
            font_family="sans-serif",
        )

        plt.title(
            f"Transfer Communities in Conferences after {season} Season",
            fontsize=42,
            fontweight="bold",
            color="black",
            pad=30,
            fontfamily="sans-serif",
        )

        import matplotlib.patches as mpatches
        from matplotlib import cm
        
        cmap = cm.get_cmap('tab10')
        
        legend_patches = []
        for i, community in enumerate(communities):
            community_members = sorted(list(community))
            color = cmap(i)
            
            if len(community_members) <= 3:
                community_name = f"Community {i+1}: {', '.join(community_members)}"
            else:
                community_name = f"Community {i+1}: {', '.join(community_members[:3])} + {len(community_members)-3} more"
            
            patch = mpatches.Patch(color=color, label=community_name)
            legend_patches.append(patch)
        
        plt.legend(
            handles=legend_patches,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=True,
            borderpad=1,
            handlelength=2
        )

        plt.axis("off")
        plt.margins(0.05)
        plt.tight_layout()

        filename = f"data/plots/transfers_community_detection_{season}.png"
        plt.savefig(
            filename, dpi=200, bbox_inches="tight", facecolor="#ffffff", edgecolor="none"
        )
        plt.close()

        return filename


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
                    "sec_transfers_summary",
                    "sec_transfers_map",
                    "transfers_community_detection",
                    "transfer_density_map",
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

    parser.add_argument(
        "--transfer-type",
        type=str,
        choices=["destination", "origin", "both"],
        help="Type of transfer density map: destination, origin, or both",
        default="destination",
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
    elif parsed_args.command == "sec_transfers_summary":
        if parsed_args.season is None:
            raise ValueError(
                "Please provide a season year using --season for this command."
            )
        sec_summary_df = sec_transfers_map(season=parsed_args.season)
        print(sec_summary_df)
    elif parsed_args.command == "sec_transfers_map":
        if parsed_args.season is None:
            raise ValueError(
                "Please provide a season year using --season for this command."
            )
        map_path = f"data/plots/sec_transfers_map_{parsed_args.season}.html"
        sec_map = create_sec_transfers_map(season=parsed_args.season, save_path=map_path)
        if sec_map:
            print(f"SEC transfers map for {parsed_args.season} saved to {map_path}")
        else:
            print(f"No transfer data available for season {parsed_args.season}")
    elif parsed_args.command == "transfers_community_detection":
        if parsed_args.season is None:
            raise ValueError(
                "Please provide a season year using --season for this command."
            )
        communities_file = explorer.transfers_community_detection(season=parsed_args.season)
        print(
            f"Transfer community detection graph for {parsed_args.season} saved as PNG: {communities_file}"
        )
    elif parsed_args.command == "transfer_density_map":
        if parsed_args.season is None:
            raise ValueError(
                "Please provide a season year using --season for this command."
            )
        map_path = f"data/plots/transfer_{parsed_args.transfer_type}_density_{parsed_args.season}.html"
        density_map = create_transfer_density_map(
            season=parsed_args.season, 
            transfer_type=parsed_args.transfer_type,
            save_path=map_path
        )
        if density_map:
            print(f"Transfer {parsed_args.transfer_type} density map for {parsed_args.season} saved to {map_path}")
        else:
            print(f"No transfer data available for season {parsed_args.season}")
    else:
        parser.print_help()
