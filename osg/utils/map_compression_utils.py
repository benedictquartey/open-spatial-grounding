import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import torch

def save_and_show_plot(fig, filename, visualize, tmp_fldr):
    if not os.path.exists(tmp_fldr):
        os.makedirs(tmp_fldr)
    filepath = os.path.join(tmp_fldr, filename)
    fig.savefig(filepath)
    if visualize:
        plt.show()
    else:
        plt.close(fig)

def extract_direction_vector(rotation_matrix):
    return rotation_matrix[:, 2]  # Assuming forward direction is along z-axis

def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def group_nodes_by_direction(nodes, threshold_angle=10):
    groups = []
    for node in nodes:
        direction_vector = extract_direction_vector(node['rep_pose']['rotation_matrix'])
        added_to_group = False
        for group in groups:
            group_direction_vector = extract_direction_vector(group[0]['rep_pose']['rotation_matrix'])
            angle = angle_between_vectors(direction_vector, group_direction_vector)
            if angle <= threshold_angle:
                group.append(node)
                added_to_group = True
                break
        if not added_to_group:
            groups.append([node])
    return groups

def plot_clustering(node_coords, y_kmeans, centers, labels, visualize, tmp_fldr):
    x = [coord[0] for coord in node_coords.values()]
    y = [coord[1] for coord in node_coords.values()]

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(x, y, c=y_kmeans, s=50, cmap='viridis')  # plot embedding points

    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)  # plot centers

    plt.title("Clustering waypoints based on coordinates")

    # Show only the first 4 characters of each key
    short_labels = [label[:4] for label in labels]
    for i, txt in enumerate(short_labels):
        ax.annotate(txt, (x[i], y[i]))

    save_and_show_plot(fig, 'clustering.png', visualize, tmp_fldr)

def plot_direction_groups(groups, visualize, tmp_fldr):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))

    for i, group in enumerate(groups):
        group_x = [node['xy_coordinate'][0] for node in group]
        group_y = [node['xy_coordinate'][1] for node in group]
        color = colors[i]
        ax.scatter(group_x, group_y, label=f'Direction Group {i + 1}', alpha=0.6, color=color)
        for node in group:
            direction_vector = extract_direction_vector(node['rep_pose']['rotation_matrix'])
            ax.quiver(node['xy_coordinate'][0], node['xy_coordinate'][1],
                      direction_vector[0], direction_vector[1],
                      color=color, scale=2, scale_units='xy', angles='xy', width=0.003, headwidth=3, headlength=4)

    ax.set_title("Direction Groups")
    ax.legend()

    # Set axis limits
    all_x = [node['xy_coordinate'][0] for group in groups for node in group]
    all_y = [node['xy_coordinate'][1] for group in groups for node in group]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

    save_and_show_plot(fig, 'direction_groups.png', visualize, tmp_fldr)

def plot_nodes_side_by_side_direction(original_nodes, selected_nodes, direction_groups, visualize, tmp_fldr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(direction_groups)))

    # Plot original nodes
    for i, group in enumerate(direction_groups):
        group_x = [node['xy_coordinate'][0] for node in group]
        group_y = [node['xy_coordinate'][1] for node in group]
        color = colors[i]
        ax1.scatter(group_x, group_y, label=f'Direction Group {i + 1}', alpha=0.6, color=color)
        for node in group:
            direction_vector = extract_direction_vector(node['rep_pose']['rotation_matrix'])
            ax1.quiver(node['xy_coordinate'][0], node['xy_coordinate'][1],
                       direction_vector[0], direction_vector[1],
                       color=color, scale=2, scale_units='xy', angles='xy', width=0.003, headwidth=3, headlength=4)
    ax1.set_title("Original Nodes by Direction Groups")
    ax1.legend()

    # Plot selected nodes
    for i, group in enumerate(direction_groups):
        group_keys = [node['waypoint_key'] for node in group]
        group_nodes = [node for node in selected_nodes if node['waypoint_key'] in group_keys]
        if group_nodes:
            group_x = [node['xy_coordinate'][0] for node in group_nodes]
            group_y = [node['xy_coordinate'][1] for node in group_nodes]
            color = colors[i]
            ax2.scatter(group_x, group_y, label=f'Direction Group {i + 1}', alpha=0.6, color=color)
            for node in group_nodes:
                direction_vector = extract_direction_vector(node['rep_pose']['rotation_matrix'])
                ax2.quiver(node['xy_coordinate'][0], node['xy_coordinate'][1],
                           direction_vector[0], direction_vector[1],
                           color=color, scale=2, scale_units='xy', angles='xy', width=0.003, headwidth=3, headlength=4)
    ax2.set_title("Selected Nodes by Direction Groups")
    ax2.legend()

    # Set axis limits
    all_x = [node['xy_coordinate'][0] for node in original_nodes + selected_nodes]
    all_y = [node['xy_coordinate'][1] for node in original_nodes + selected_nodes]
    ax1.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax1.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax2.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax2.set_ylim(min(all_y) - 1, max(all_y) + 1)

    save_and_show_plot(fig, 'side_by_side_direction.png', visualize, tmp_fldr)

def plot_nodes_side_by_side_position(original_nodes, selected_nodes, clusters, cluster_centers, visualize, tmp_fldr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    num_clusters = len(cluster_centers)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

    color_map = {i: colors[i] for i in range(num_clusters)}

    # Plot original nodes
    for i in range(num_clusters):
        cluster_nodes = [node for node, label in zip(original_nodes, clusters) if label == i]
        cluster_x = [node['xy_coordinate'][0] for node in cluster_nodes]
        cluster_y = [node['xy_coordinate'][1] for node in cluster_nodes]
        color = color_map[i]
        ax1.scatter(cluster_x, cluster_y, label=f'Cluster {i + 1}', alpha=0.6, color=color)
    ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
    ax1.set_title("Original Nodes by Position Clusters")
    ax1.legend()

    # Plot selected nodes
    for i in range(num_clusters):
        cluster_nodes = [node for node, label in zip(selected_nodes, clusters) if label == i]
        if cluster_nodes:
            cluster_x = [node['xy_coordinate'][0] for node in cluster_nodes]
            cluster_y = [node['xy_coordinate'][1] for node in cluster_nodes]
            color = color_map[i]
            ax2.scatter(cluster_x, cluster_y, label=f'Cluster {i + 1}', alpha=0.6, color=color)
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
    ax2.set_title("Selected Nodes by Position Clusters")
    ax2.legend()

    # Set axis limits
    all_x = [node['xy_coordinate'][0] for node in original_nodes + selected_nodes]
    all_y = [node['xy_coordinate'][1] for node in original_nodes + selected_nodes]
    ax1.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax1.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax2.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax2.set_ylim(min(all_y) - 1, max(all_y) + 1)

    save_and_show_plot(fig, 'side_by_side_position.png', visualize, tmp_fldr)

def plot_nodes_side_by_side_position_direction(original_nodes, selected_nodes, clusters, cluster_centers, direction_groups, visualize, tmp_fldr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    num_clusters = len(cluster_centers)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

    color_map = {i: colors[i] for i in range(num_clusters)}

    # Plot original nodes
    for i in range(num_clusters):
        cluster_nodes = [node for node, label in zip(original_nodes, clusters) if label == i]
        cluster_x = [node['xy_coordinate'][0] for node in cluster_nodes]
        cluster_y = [node['xy_coordinate'][1] for node in cluster_nodes]
        color = color_map[i]
        ax1.scatter(cluster_x, cluster_y, label=f'Cluster {i + 1}', alpha=0.6, color=color)
    ax1.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
    ax1.set_title("Original Nodes by Position Clusters")
    ax1.legend()

    # Plot selected nodes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(direction_groups)))
    for i, group in enumerate(direction_groups):
        group_keys = [node['waypoint_key'] for node in group]
        group_nodes = [node for node in selected_nodes if node['waypoint_key'] in group_keys]
        if group_nodes:
            group_x = [node['xy_coordinate'][0] for node in group_nodes]
            group_y = [node['xy_coordinate'][1] for node in group_nodes]
            color = colors[i]
            ax2.scatter(group_x, group_y, label=f'Direction Group {i + 1}', alpha=0.6, color=color)
            for node in group_nodes:
                direction_vector = extract_direction_vector(node['rep_pose']['rotation_matrix'])
                ax2.quiver(node['xy_coordinate'][0], node['xy_coordinate'][1],
                           direction_vector[0], direction_vector[1],
                           color=color, scale=2, scale_units='xy', angles='xy', width=0.003, headwidth=3, headlength=4)
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
    ax2.set_title("Selected Nodes by Position and Direction Groups")
    ax2.legend()

    # Set axis limits
    all_x = [node['xy_coordinate'][0] for node in original_nodes + selected_nodes]
    all_y = [node['xy_coordinate'][1] for node in original_nodes + selected_nodes]
    ax1.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax1.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax2.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax2.set_ylim(min(all_y) - 1, max(all_y) + 1)

    save_and_show_plot(fig, 'side_by_side_position_direction.png', visualize, tmp_fldr)

def compress_observation_graph(percentage_to_keep, observations_graph, group_by="direction", visualize=True, tmp_fldr="plots", plot_initial_graph=False):
    # Extract nodes from the graph
    nodes = [data for _, data in observations_graph.nodes(data=True)]

    if group_by == "direction":
        groups = group_nodes_by_direction(nodes, threshold_angle=10)
        selected_nodes = select_nodes_from_groups(groups, nodes, percentage_to_keep)
        if plot_initial_graph: plot_direction_groups(groups, visualize, tmp_fldr)  # Plot direction groups
        plot_nodes_side_by_side_direction(nodes, selected_nodes, groups, visualize, tmp_fldr)
    elif group_by == "position":
        node_coords, kmeans, y_kmeans, centers = cluster_nodes_by_position(nodes,percentage_to_keep)
        groups = create_groups_from_clusters(nodes, y_kmeans, kmeans)
        selected_nodes = select_nodes_from_groups(groups, nodes, percentage_to_keep)
        if plot_initial_graph: plot_clustering(node_coords, y_kmeans, centers, [node['waypoint_key'] for node in nodes], visualize, tmp_fldr)
        plot_nodes_side_by_side_position(nodes, selected_nodes, y_kmeans, centers, visualize, tmp_fldr)
    elif group_by == "position_direction":
        node_coords, kmeans, y_kmeans, centers = cluster_nodes_by_position(nodes,percentage_to_keep)
        groups = create_groups_from_clusters(nodes, y_kmeans, kmeans)
        selected_nodes = select_nodes_from_clusters_with_direction(groups, nodes, percentage_to_keep)
        if plot_initial_graph: plot_clustering(node_coords, y_kmeans, centers, [node['waypoint_key'] for node in nodes], visualize, tmp_fldr)
        direction_groups = [group_nodes_by_direction(cluster, threshold_angle=10) for cluster in groups]
        flat_direction_groups = [group for subgroup in direction_groups for group in subgroup]  # Flatten the nested list
        plot_nodes_side_by_side_position_direction(nodes, selected_nodes, y_kmeans, centers, flat_direction_groups, visualize, tmp_fldr)

    # Create a new graph with the selected nodes
    new_observations_graph = nx.Graph()
    for node in selected_nodes:
        new_observations_graph.add_node(node['waypoint_key'], **node)

    return new_observations_graph

def select_nodes_from_groups(groups, nodes, percentage_to_keep):
    # Calculate number of nodes to keep
    total_nodes = len(nodes)
    num_nodes_to_keep = max(1, int(total_nodes * (percentage_to_keep / 100)))

    # Select one node from each group
    selected_nodes = []
    selected_node_keys = set()
    for group in groups:
        if group:
            selected_node = random.choice(group)
            selected_nodes.append(selected_node)
            selected_node_keys.add(selected_node['waypoint_key'])

    # Ensure the correct number of nodes are selected
    if len(selected_nodes) < num_nodes_to_keep:
        additional_nodes_needed = num_nodes_to_keep - len(selected_nodes)
        remaining_nodes = [node for node in nodes if node['waypoint_key'] not in selected_node_keys]
        additional_nodes = random.sample(remaining_nodes, additional_nodes_needed)
        selected_nodes.extend(additional_nodes)
        selected_node_keys.update(node['waypoint_key'] for node in additional_nodes)
    elif len(selected_nodes) > num_nodes_to_keep:
        selected_nodes = random.sample(selected_nodes, num_nodes_to_keep)

    return selected_nodes

def select_nodes_from_clusters_with_direction(clusters, nodes, percentage_to_keep):
    selected_nodes = []
    selected_node_keys = set()
    for cluster in clusters:
        direction_groups = group_nodes_by_direction(cluster, threshold_angle=10)
        for group in direction_groups:
            if group:
                selected_node = random.choice(group)
                selected_nodes.append(selected_node)
                selected_node_keys.add(selected_node['waypoint_key'])

    # Ensure the correct number of nodes are selected
    total_nodes = len(nodes)
    num_nodes_to_keep = max(1, int(total_nodes * (percentage_to_keep / 100)))
    if len(selected_nodes) < num_nodes_to_keep:
        additional_nodes_needed = num_nodes_to_keep - len(selected_nodes)
        remaining_nodes = [node for node in nodes if node['waypoint_key'] not in selected_node_keys]
        additional_nodes = random.sample(remaining_nodes, additional_nodes_needed)
        selected_nodes.extend(additional_nodes)
    elif len(selected_nodes) > num_nodes_to_keep:
        selected_nodes = random.sample(selected_nodes, num_nodes_to_keep)

    return selected_nodes

def cluster_nodes_by_position(nodes,percentage_to_keep):
    node_coords = {node['waypoint_key']: node['xy_coordinate'] for node in nodes}
    elements = list(node_coords.values())

    # Calculate number of clusters from percentage to keep
    total_nodes = len(nodes)
    num_nodes_to_keep = int(max(1, int(total_nodes) * (percentage_to_keep / 100)))
    
    # print(f"Number of nodes to keep: {num_nodes_to_keep}")
    # print(f"Total nodes: {total_nodes}")

    # num_centroids = min(10, num_nodes_to_keep)
    num_centroids = num_nodes_to_keep
    kmeans = KMeans(n_clusters=num_centroids, init='k-means++', random_state=1)
    kmeans.fit(elements)
    y_kmeans = kmeans.predict(elements)
    centers = kmeans.cluster_centers_
    return node_coords, kmeans, y_kmeans, centers

def create_groups_from_clusters(nodes, y_kmeans, kmeans):
    num_centroids = len(kmeans.cluster_centers_)
    groups = [[] for _ in range(num_centroids)]
    for node, label in zip(nodes, y_kmeans):
        groups[label].append(node)
    return groups

