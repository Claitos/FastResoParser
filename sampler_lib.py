import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parser
import reverser
from pathlib import Path
import random as rd


################################################################
#
# Functions for edge studies
#
################################################################



def mass_conservation(p_df: pd.DataFrame, d_df: pd.DataFrame, edge_id: str = "+", verbose: bool = False) -> pd.DataFrame:
    """
    Check mass conservation for each decay channel.
    If the sum of the masses of the decay products exceeds the mass of the parent particle, correct the mass of the mother to the sum of the daughter masses in the case of "-".
    In the case of "+", scale the masses of the decay products to match the parent's mass while preserving their relative ratios. Also handle the cases where scaling leads to negative mass errors (the error on the decay product mass is very small and thus can become negative when scaling).

    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.
        verbose (bool): If True, print additional information.

    Returns:
        pd.DataFrame: DataFrame with updated parent masses.
    """

    counter_corr = 0
    counter_negative = 0

    if edge_id == "+":
        order = 1
    elif edge_id == "-":
        order = -1

    for index, row in d_df.iloc[::order].iterrows():
        parent_id = row["ParentID"]
        decay_products = row["ProductIDs"]
        decay_product_ids = [int(dp) for dp in decay_products]
        decay_product_ids = [dp for dp in decay_product_ids if dp != 0]  # Remove zeros

        parent_mass = p_df.loc[p_df["ID"] == parent_id, "Mass (GeV)"].values[0]
        decay_product_masses = p_df.loc[p_df["ID"].isin(decay_product_ids), "Mass (GeV)"].values

        parent_mass_err_p = p_df.loc[p_df["ID"] == parent_id, "Mass Error Pos (GeV)"].values[0]
        parent_mass_err_n = p_df.loc[p_df["ID"] == parent_id, "Mass Error Neg (GeV)"].values[0]
        decay_product_masses_err_p = p_df.loc[p_df["ID"].isin(decay_product_ids), "Mass Error Pos (GeV)"].values.tolist()
        decay_product_masses_err_n = p_df.loc[p_df["ID"].isin(decay_product_ids), "Mass Error Neg (GeV)"].values.tolist()

        if verbose:
            print(f"\nChecking Parent ID {parent_id} with Decay Products {decay_product_ids}")
            print(f"Parent Mass: {parent_mass}, Decay Product Masses: {decay_product_masses}")

        total_decay_mass = np.sum(decay_product_masses)

        if total_decay_mass > parent_mass:
            if verbose:
                print(f"Mass conservation violated for Parent ID {parent_id}: Parent Mass = {parent_mass}, Sum of Decay Products Mass = {total_decay_mass}. Correcting masses.")

            epsilon = 0.001  # Small factor to avoid exact equality
            if edge_id == "+":
                scale_factor = (parent_mass / total_decay_mass) * (1-epsilon)
                decay_product_masses_err_scaled = decay_product_masses * scale_factor - (decay_product_masses - decay_product_masses_err_p)

                indices_negative = []
                for i, decay_product_mass_err in enumerate(decay_product_masses_err_scaled):
                    if decay_product_mass_err < 0:
                        indices_negative.append(i)
                        if verbose:
                            print(f"Warning: Negative scaled mass error encountered for a decay product of Parent ID {parent_id}. Setting error to zero.")
                        decay_product_masses_err_scaled[i] = 0.0
                        counter_negative += 1

                if indices_negative is not None and len(indices_negative) > 0:
                    for i , decay_product_mass_err in enumerate(decay_product_masses_err_scaled):
                        if i not in indices_negative:
                            scale_factor = (parent_mass - np.sum(np.array(decay_product_masses)[indices_negative])) / np.sum(np.array(decay_product_masses)[[j for j in range(len(decay_product_masses)) if j not in indices_negative]])
                            scale_factor *= (1-epsilon)
                            decay_product_masses_err_scaled[i] = decay_product_masses[i] * scale_factor - (decay_product_masses[i] - decay_product_masses_err_p[i])
                


                decay_product_masses_scaled = decay_product_masses - decay_product_masses_err_p + decay_product_masses_err_scaled
                p_df.loc[p_df["ID"].isin(decay_product_ids), "Mass (GeV)"] = decay_product_masses_scaled

                if verbose:
                    print(f"Scaled Decay Product Masses: {decay_product_masses_scaled} replacing original masses {decay_product_masses}.")

            elif edge_id == "-":
                p_df.loc[p_df["ID"] == parent_id, "Mass (GeV)"] = total_decay_mass * (1+epsilon)
                if verbose:
                    print(f"Corrected Parent Mass to: {total_decay_mass * (1+epsilon)}")

            counter_corr += 1
    
    if verbose:
        print(f"\nTotal corrections made: {counter_corr}")

    return p_df



def get_edges(p_df: pd.DataFrame, d_df: pd.DataFrame, identifier: str = "mass", edge_id: str = "+", verbose: bool = False) -> pd.DataFrame:
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
    
    if verbose:
        print(f"\n\n\n------------Calculating {identifier} edge with edge_id '{edge_id}'--------------\n\n\n")

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


    if identifier == "mass":
        p_df_edges = mass_conservation(p_df_edges, d_df, edge_id=edge_id, verbose=verbose)

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
        d_df_br["BranchingRatio"] = min(d_df_br["BranchingRatio"] + err_p, 1)
    else:
        d_df_br["BranchingRatio"] = max(d_df_br["BranchingRatio"] - err_n, 0)

    return d_df_br




def edge_study(p_df: pd.DataFrame, d_df: pd.DataFrame, cut: int = 0, verbose: bool = False) -> None:
    """
    Perform an edge study by generating data files with mass and width edges applied. Optionally apply a importance score cut to the particles in the dataframes.
    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.
        cut (int): If non-zero, apply a cut to the dataframes before processing.
    """

    if cut == 0:
        dir_name = "Datafiles_sampled/edge_study_final2"
        p_df_cut = p_df
        d_df_cut = d_df
    else:
        dir_name = f"Datafiles_sampled/edge_study_final2_{cut:.0e}"
        p_df_cut, d_df_cut = parser.cutting_dataframes(p_df, d_df, cut=cut, verbose=True)


    mass_p_df = get_edges(p_df_cut, d_df_cut, identifier="mass", edge_id="+", verbose=verbose)
    mass_n_df = get_edges(p_df_cut, d_df_cut, identifier="mass", edge_id="-", verbose=verbose)
    decay_p_df = get_edges(p_df_cut, d_df_cut, identifier="width", edge_id="+", verbose=verbose)
    decay_n_df = get_edges(p_df_cut, d_df_cut, identifier="width", edge_id="-", verbose=verbose)
    br_p_df = br_edge(d_df_cut, edge_id="+")
    br_n_df = br_edge(d_df_cut, edge_id="-")

    p_dfs = [p_df_cut, mass_p_df, mass_n_df, decay_p_df, decay_n_df, p_df_cut, p_df_cut]
    d_dfs = [d_df_cut, d_df_cut, d_df_cut, d_df_cut, d_df_cut, br_p_df, br_n_df]

    no_lists = len(p_dfs)

    Path(dir_name).mkdir(parents=True, exist_ok=True)
    output_paths = [f"{dir_name}/PDG2016Plus_{i}.dat" for i in range(no_lists)]

    for i in range(no_lists):
        parser.parse_to_dat(output_paths[i], p_dfs[i], d_dfs[i])

    reverser.reverser_routine_new(dir_name, no_lists)










################################################################
#
# Functions for sampling studies
#
################################################################




def sampling_study(p_df: pd.DataFrame, d_df: pd.DataFrame, cut: int = 0, verbose: bool = False) -> None:
    """
    Perform a sampling study by generating data files with mass and branching ratio values sampled. Optionally apply a importance score cut to the particles in the dataframes.
    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.
        cut (int): If non-zero, apply a cut to the dataframes before processing.
    """

    if cut == 0:
        dir_name = "Datafiles_sampled/sampling_studys/br_rejection_0"
        p_df_cut = p_df
        d_df_cut = d_df
    else:
        dir_name = f"Datafiles_sampled/sampling_studys/br_rejection_{cut:.0e}"
        p_df_cut, d_df_cut = parser.cutting_dataframes(p_df, d_df, cut=cut, verbose=True)


    n_samples = 20
    p_dfs = [p_df_cut]
    d_dfs = [d_df_cut]

    for _ in range(n_samples):
        if verbose:
            print(f"\n\n\n-------Sampling iteration {_+1}-------\n\n\n")
        else:
            print(f"Sampling iteration {_+1}")

        sampled_p_df = p_df_cut.copy()  # Currently no sampling for masses implemented
        sampled_d_df = sample_branching_ratios(d_df_cut, rej_sampling=True, verbose=verbose)

        p_dfs.append(sampled_p_df)
        d_dfs.append(sampled_d_df)

    no_lists = len(p_dfs)

    Path(dir_name).mkdir(parents=True, exist_ok=True)
    output_paths = [f"{dir_name}/PDG2016Plus_{i}.dat" for i in range(no_lists)]

    for i in range(no_lists):
        parser.parse_to_dat(output_paths[i], p_dfs[i], d_dfs[i])

    reverser.reverser_routine_new(dir_name, no_lists)




def sample_branching_ratios(d_df: pd.DataFrame, rej_sampling: bool = True, verbose: bool = False) -> pd.DataFrame:
    """
    Generate a decay DataFrame with branching ratios sampled according to their uncertainties.
    Optionally implements rejection sampling to ensure physical validity. (Sum of BRs for each parent must be equal to 1).
    Otherwise, samples each BR independently within its error range.

    Parameters:
        d_df (pd.DataFrame): DataFrame containing the decay data.
        verbose (bool): If True, print additional information.

    Returns:
        pd.DataFrame: DataFrame with sampled branching ratios.
    """

    sampled_dfs = d_df.copy() # create a copy to avoid modifying the original DataFrame
    fail_counter = 0

    for id in sampled_dfs["ParentID"].unique():

        decay_channels = sampled_dfs[sampled_dfs["ParentID"] == id]
        brs = decay_channels["BranchingRatio"].values
        br_err_p = decay_channels["BR Error Pos"].values
        br_err_n = decay_channels["BR Error Neg"].values
        br_err_p_corrected = np.minimum(br_err_p, 1.0 - brs)
        br_err_n_corrected = np.minimum(br_err_n, brs)
        no_modes = len(brs)

        if verbose:
            print(f"\nSampling BRs for Parent ID {id} with original BRs: {brs.tolist()} and errors +{br_err_p.tolist()}, -{br_err_n.tolist()}")
            print(f"Corrected errors to avoid unphysical values: +{br_err_p_corrected.tolist()}, -{br_err_n_corrected.tolist()}")


        if rej_sampling:
            if no_modes == 1:
                if verbose:
                    print(f"Only one decay channel for Parent ID {id}. No sampling needed.")
                continue


            biased = rd.randint(0, no_modes - 1)
            if verbose:
                print(f"Using rejection sampling. Biased mode index: {biased}")

            sampled_brs = np.zeros_like(brs)
            n_attempts = 0
            max_attempts = 100000

            while n_attempts < max_attempts:
                n_attempts += 1


                for i in range(no_modes):
                    if i == biased:
                        continue
                    else:
                        sampled_brs[i] = rd.uniform(brs[i] - br_err_n_corrected[i], brs[i] + br_err_p_corrected[i])

                sampled_brs[biased] = 1.0 - np.sum(sampled_brs)

                if sampled_brs[biased] > brs[biased] - br_err_n_corrected[biased] and sampled_brs[biased] < brs[biased] + br_err_p_corrected[biased]:
                    if verbose:
                        print(f"Sampled BRs for Parent ID {id} after {n_attempts} attempts: {sampled_brs.tolist()}")
                    break
            else:
                if verbose:
                    print(f"Warning: Maximum attempts reached for Parent ID {id}. Using original BRs {brs.tolist()}.")
                fail_counter += 1
                sampled_brs = brs.copy()
            
        else:
            sampled_brs = np.zeros_like(brs)
            for i in range(no_modes):
                sampled_brs[i] = rd.uniform(brs[i] - br_err_n[i], brs[i] + br_err_p[i])
            if verbose:
                print(f"Sampled BRs for Parent ID {id}: {sampled_brs.tolist()}")
            

        sampled_dfs.loc[sampled_dfs["ParentID"] == id, "BranchingRatio"] = sampled_brs

        
    
    print("\nSampling complete.")
    print(f"Total failures (rejection sampling): {fail_counter}")
    print(f"Total successful samples: {len(sampled_dfs['ParentID'].unique()) - fail_counter} \n")

    return sampled_dfs


