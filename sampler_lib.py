import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


################################################################
#
# Functions for edge sampling studies
#
################################################################




def mass_edge(p_df: pd.DataFrame, edge_id: str = "+") -> pd.DataFrame:
    """
    Calculate the mass edge and replace the original mass values. Edge is determined by adding or subtracting the error.

    Parameters:
    p_df (pd.DataFrame): DataFrame containing the mass data.
    edge_id (str): Identifier for the mass edge, either "+" or "-".

    Returns:
    pd.DataFrame: DataFrame with the mass edge value.
    """
    
    if edge_id not in ["+", "-"]:
        raise ValueError("edge_id must be either '+' or '-'")

    err_p = p_df["Mass Error Pos (GeV)"]
    err_n = p_df["Mass Error Neg (GeV)"]


    if edge_id == "+":
        p_df["Mass (GeV)"] = p_df["Mass (GeV)"] + err_p
    else:
        p_df["Mass (GeV)"] = p_df["Mass (GeV)"] - err_n

    return p_df



def width_edge(p_df: pd.DataFrame, edge_id: str = "+") -> pd.DataFrame:
    """
    Calculate the width edge and replace the original width values. Edge is determined by adding or subtracting the error.

    Parameters:
    p_df (pd.DataFrame): DataFrame containing the width data.
    edge_id (str): Identifier for the width edge, either "+" or "-".

    Returns:
    pd.DataFrame: DataFrame with the width edge value.
    """
    
    if edge_id not in ["+", "-"]:
        raise ValueError("edge_id must be either '+' or '-'")

    err_p = p_df["Width Error Pos (GeV)"]
    err_n = p_df["Width Error Neg (GeV)"]


    if edge_id == "+":
        p_df["Width (GeV)"] = p_df["Width (GeV)"] + err_p
    else:
        p_df["Width (GeV)"] = p_df["Width (GeV)"] - err_n

    return p_df



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


    if edge_id == "+":
        d_df["BranchingRatio"] = d_df["BranchingRatio"] + err_p
    else:
        d_df["BranchingRatio"] = d_df["BranchingRatio"] - err_n

    return d_df