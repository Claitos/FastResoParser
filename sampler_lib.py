import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


################################################################
#
# Functions for edge sampling studies
#
################################################################




def get_edges(p_df: pd.DataFrame, identifier: str = "mass", edge_id: str = "+", verbose: bool = False) -> pd.DataFrame:
    """
    Calculate the mass edge and replace the original mass values. Edge is determined by adding or subtracting the error. Only unstable particles are affected.

    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        identifier (str): Type of edge to calculate, either "mass", "width".
        edge_id (str): Identifier for the mass edge, either "+" or "-".
        verbose (bool): If True, print additional information.

    Returns:
        pd.DataFrame: DataFrame with the mass edge value.
    """
    
    if edge_id not in ["+", "-"]:
        raise ValueError("edge_id must be either '+' or '-'")
    
    if identifier not in ["mass", "width"]:
        raise ValueError("identifier must be either 'mass' or 'width'")

    stable_particles_test = p_df[p_df["Width (GeV)"] == 0.0]
    stable_particles = stable_particles_test[stable_particles_test["No. of decay channels"] == 1]["ID"].tolist()
    bool_series = p_df["ID"].isin(stable_particles)

    if identifier == "mass":
        err_p = p_df["Mass Error Pos (GeV)"]
        err_n = p_df["Mass Error Neg (GeV)"]
    elif identifier == "width":
        err_p = p_df["Width Error Pos (GeV)"]
        err_n = p_df["Width Error Neg (GeV)"]

    p_df_edges = p_df.copy()   # create a copy to avoid modifying the original DataFrame

    if identifier == "mass":
        if edge_id == "+":
            p_df_edges.loc[~bool_series, "Mass (GeV)"] += err_p
        else:
            p_df_edges.loc[~bool_series, "Mass (GeV)"] -= err_n
    elif identifier == "width":
        if edge_id == "+":
            p_df_edges.loc[~bool_series, "Width (GeV)"] += err_p
        else:
            p_df_edges.loc[~bool_series, "Width (GeV)"] -= err_n

    return p_df_edges




def br_edge(d_df: pd.DataFrame, edge_id: str = "+") -> pd.DataFrame:
    """
    Calculate the branching ratio edge and replace the original branching ratio values. Edge is determined by adding or subtracting the error.
    
    Parameters:
        d_df (pd.DataFrame): DataFrame containing the branching ratio data.
        edge_id (str): Identifier for the branching ratio edge, either "+" or "-".

    Returns:
        pd.DataFrame: DataFrame with the branching ratio edge value.
    """
    
    if edge_id not in ["+", "-"]:
        raise ValueError("edge_id must be either '+' or '-'")

    err_p = d_df["BR Error Pos"]
    err_n = d_df["BR Error Neg"]

    d_df_br = d_df.copy()   # create a copy to avoid modifying the original DataFrame


    if edge_id == "+":
        d_df_br["BranchingRatio"] = d_df_br["BranchingRatio"] + err_p
    else:
        d_df_br["BranchingRatio"] = d_df_br["BranchingRatio"] - err_n

    return d_df_br