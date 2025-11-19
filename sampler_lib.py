import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import parser
import reverser
from pathlib import Path
import random as rd
from scipy.sparse import csr_matrix, diags


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

    err_p = d_df["BR Error Pos"].values
    err_n = d_df["BR Error Neg"].values
    brs = d_df["BranchingRatio"].values

    d_df_br = d_df.copy()   # create a copy to avoid modifying the original DataFrame


    if edge_id == "+":
        d_df_br["BranchingRatio"] = np.minimum(d_df_br["BranchingRatio"] + err_p, 1)
    else:
        d_df_br["BranchingRatio"] = np.maximum(d_df_br["BranchingRatio"] - err_n, 0)

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




def sampling_study(p_df: pd.DataFrame, d_df: pd.DataFrame, dir_name: str, case: str = "mass", cut: int = 0,  verbose: bool = False) -> None:
    """
    Perform a sampling study by generating data files with mass and branching ratio values sampled. Optionally apply a importance score cut to the particles in the dataframes.
    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.
        dir_name (str): Directory name to save the sampled data files.
        case (str): Type of sampling to perform, either "mass", "br", or "both".
        cut (int): If non-zero, apply a cut to the dataframes before processing.
        verbose (bool): If True, print detailed information during processing.
    """

    if case.lower() not in ["mass", "br", "both"]:
        raise ValueError("case must be either 'mass', 'br', or 'both'")

    if cut == 0:
        dir_name = f"{dir_name}_0"
        p_df_cut = p_df
        d_df_cut = d_df
    else:
        dir_name = f"{dir_name}_{cut:.0e}"
        p_df_cut, d_df_cut = parser.cutting_dataframes(p_df, d_df, cut=cut, verbose=verbose)


    n_samples = 1001    # includes the last list with mean masses -> 1000 samples + 1 mean mass list
    p_dfs = [p_df_cut]
    d_dfs = [d_df_cut]

    if case.lower() == "br":
        for _ in range(n_samples):
            if verbose:
                print(f"\n\n\n-------Sampling iteration {_+1}-------\n\n\n")
            else:
                print(f"Sampling iteration {_+1}")

            sampled_p_df = p_df_cut.copy()  
            sampled_d_df = sample_branching_ratios(d_df_cut, rej_sampling=True, verbose=False)

            p_dfs.append(sampled_p_df)
            d_dfs.append(sampled_d_df)

    elif case.lower() == "mass":
        sampled_df_list = sample_masses(p_df_cut, d_df_cut, n_samples=n_samples-1, verbose=verbose)

        for i in range(n_samples):

            sampled_p_df = sampled_df_list[i]
            sampled_d_df = d_df_cut.copy()  

            p_dfs.append(sampled_p_df)
            d_dfs.append(sampled_d_df)

    elif case.lower() == "both":
        sampled_df_list = sample_masses(p_df_cut, d_df_cut, n_samples=n_samples-1, verbose=verbose)

        for i in range(n_samples):
            if verbose:
                print(f"\n\n\n-------Sampling iteration {i+1}-------\n\n\n")
            else:
                print(f"Sampling iteration {i+1}")

            sampled_p_df = sampled_df_list[i]
            sampled_d_df = sample_branching_ratios(d_df_cut, rej_sampling=True, verbose=False)

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
    Optionally implements rejection sampling to ensure physical validity. (Sum of BRs for each parent must be equal to 1). If this fails after a set number of attempts, the original BRs are kept.
    Otherwise, samples each BR independently within its error range, but ensures that the sampled BRs are between 0 and 1. Here the sum does not necessarily equal 1.

    Parameters:
        d_df (pd.DataFrame): DataFrame containing the decay data.
        rej_sampling (bool): If True, use rejection sampling to ensure sum of BRs equals 1.
        verbose (bool): If True, print additional information.

    Returns:
        pd.DataFrame: DataFrame with sampled branching ratios.
    """

    sampled_dfs = d_df.copy() # create a copy to avoid modifying the original DataFrame
    fail_counter = 0
    max_no_attempts = 0

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
            lower = brs[biased] - br_err_n_corrected[biased]
            upper = brs[biased] + br_err_p_corrected[biased]
            if verbose:
                print(f"Using rejection sampling. Biased mode index: {biased} with range [{lower}, {upper}]")

            sampled_brs = np.zeros(no_modes)
            n_attempts = 0
            max_attempts = 100000

            while n_attempts < max_attempts:
                n_attempts += 1

                for i in range(no_modes):
                    if i == biased:
                        sampled_brs[i] = 0.0
                    else:
                        sampled_brs[i] = rd.uniform(brs[i] - br_err_n_corrected[i], brs[i] + br_err_p_corrected[i])

                sampled_brs[biased] = 1.0 - np.sum(sampled_brs)

                if sampled_brs[biased] >= lower and sampled_brs[biased] <= upper:
                    if verbose:
                        print(f"Sampled BRs for Parent ID {id} after {n_attempts} attempts: {sampled_brs.tolist()} with sum {round(np.sum(sampled_brs), 6)}")
                    break
            else:
                if verbose:
                    print(f"Warning: Maximum attempts reached for Parent ID {id}. Using original BRs {brs.tolist()}.")
                fail_counter += 1
                sampled_brs = brs.copy()

            if n_attempts > max_no_attempts:
                max_no_attempts = n_attempts
               
        else:
            sampled_brs = np.zeros_like(brs)
            for i in range(no_modes):
                lower = brs[i] - br_err_n[i]
                if lower < 0:
                    lower = 0.0
                upper = brs[i] + br_err_p[i]
                if upper > 1:
                    upper = 1.0
                sampled_brs[i] = rd.uniform(lower, upper)
            if verbose:
                print(f"Sampled BRs for Parent ID {id}: {sampled_brs.tolist()}")
            

        sampled_dfs.loc[sampled_dfs["ParentID"] == id, "BranchingRatio"] = sampled_brs

        
    
    print("\nSampling complete.")
    print("Maximum number of attempts in rejection sampling for a single parent ID:", max_no_attempts)
    print(f"Total failures (rejection sampling): {fail_counter}")
    print(f"Total successful samples: {len(sampled_dfs['ParentID'].unique()) - fail_counter} \n")

    return sampled_dfs




def sample_masses(p_df: pd.DataFrame, d_df: pd.DataFrame, n_samples: int, verbose: bool = False) -> list[pd.DataFrame]:
    """
    Generate a particle DataFrame with masses sampled according to their uncertainties fulfilling all the linear constraints that are given by "mass conservation".
    To sample the masses a hit and run MCMC algorithm is used.

    Parameters:
        p_df (pd.DataFrame): DataFrame containing the particle and mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.
        n_samples (int): Number of samples to generate.
        verbose (bool): If True, print additional information.

    Returns:
        pd.DataFrame: DataFrame with sampled masses.
    """

    stable_particles_test = p_df[p_df["Width (GeV)"] == 0.0]
    stable_particles = stable_particles_test[stable_particles_test["No. of decay channels"] == 1]["ID"].tolist()
    stable_particles.remove(3212) # Remove Sigma0 from list since it decays to Lambda + photon
    stable_particles.remove(-3212) # Remove anti-Sigma0 from list since it decays to anti-Lambda + photon
    stable_particles_ids = p_df[p_df["ID"].isin(stable_particles)].index.tolist()

    if verbose:
        print(f"Stable particles (no sampling applied): {stable_particles}\nwith indices {stable_particles_ids}\n")

    masses = p_df["Mass (GeV)"].values
    mass_err_p = p_df["Mass Error Pos (GeV)"].values
    mass_err_n = p_df["Mass Error Neg (GeV)"].values

    lower_bounds = masses - mass_err_n
    upper_bounds = masses + mass_err_p

    scale = upper_bounds - lower_bounds # scale for each mass bounds to sample direction vector later
    scale = np.clip(scale, a_min=1e-12, a_max=None)  # avoid zero scale for stable particles

    if verbose:
        print(f"Initial masses: {masses.tolist()}")
        print(f"Lower bounds: {lower_bounds.tolist()}")
        print(f"Upper bounds: {upper_bounds.tolist()}")
        #print(f"Mass scales : {(upper_bounds - lower_bounds).tolist()}")
        print(f"Scale: {scale.tolist()} \n")

    csr_constraints = constraint_matrix_constructor(p_df, d_df, stable_particles, verbose=verbose)

    mass_edges_n, mass_edges_p = mass_edges(p_df, d_df)

    initial_point = set_initial_feasible_point(masses, mass_edges_n, mass_edges_p, where="neg_edge", verbose=verbose)

    sampled_masses = hit_and_run_uniform(csr_constraints, 1e-09, lower_bounds, upper_bounds, initial_point, stable_particles_ids, scale, n_samples=n_samples, burn=1000, thin=2000, scale_coordinates=True, verbose=verbose)

    mean_sampled_masses = mean_sample(sampled_masses, scale, lower_bounds)

    if verbose:
        print(f"\n\n\nSampled masses shape: {sampled_masses.shape}")
        if np.any(sampled_masses - lower_bounds < 0):
            print("Error: Some sampled masses are below the lower bounds!")
        if np.any(sampled_masses - upper_bounds > 0):
            print("Error: Some sampled masses are above the upper bounds!")

        plot_sample(sampled_masses, scale, lower_bounds, p_df=p_df, dir_name="Plots/Debug_sampling2")
        plot_sample_2(sampled_masses, scale, lower_bounds, dir_name="Plots/Debug_sampling2", edges=(mass_edges_n, mass_edges_p))

    sampled_df_list = []

    for i in range(n_samples):
        sampled_p_df = p_df.copy()  # create a copy to avoid modifying the original DataFrame
        sampled_p_df["Mass (GeV)"] = sampled_masses[i]
        sampled_df_list.append(sampled_p_df)

    mass_mean_p_df = p_df.copy()    # create a copy to avoid modifying the original DataFrame
    mass_mean_p_df["Mass (GeV)"] = mean_sampled_masses    # set the masses to the mean of the sampled masses
    sampled_df_list.append(mass_mean_p_df)

    return sampled_df_list



def constraint_matrix_constructor(p_df: pd.DataFrame, d_df: pd.DataFrame, stable_particles: list[int], verbose: bool = False) -> csr_matrix:
    """
    Construct the constraint matrix for mass conservation.

    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.
        stable_particles (list[int]): List of stable particle IDs.
        verbose (bool): If True, print additional information.

    Returns:
        csr_matrix: Constraint matrix in CSR format.
    """

    constraints_matrix = np.zeros((len(d_df), len(p_df)))

    for i, row in d_df.iterrows():
        parent_id = row["ParentID"]
        decay_products = row["ProductIDs"]
        decay_product_ids = [int(dp) for dp in decay_products]
        decay_product_ids = [dp for dp in decay_product_ids if dp != 0]  # Remove zeros

        parent_index = p_df.index[p_df["ID"] == parent_id][0]
        constraints_matrix[i, parent_index] = 1

        if parent_id in stable_particles:
            if verbose:
                print(f"Skipping stable parent particle ID {parent_id} in constraints. Just leave 1 at its index and skip the check for decay products.")
            continue

        for dp_id in decay_product_ids:
            dp_index = p_df.index[p_df["ID"] == dp_id][0]
            constraints_matrix[i, dp_index] = -1

    if verbose:
        print(f"\nConstraints matrix with shape {constraints_matrix.shape} :\n", constraints_matrix)
        print(f"first constraint row: {constraints_matrix[0]} \n\n")

    csr_constraints = csr_matrix(constraints_matrix)

    return csr_constraints




def set_initial_feasible_point(org_masses: np.ndarray, edge_neg: np.ndarray, edge_pos: np.ndarray, where: str = "org", verbose: bool = False) -> np.ndarray:
    """
    Set an initial feasible point within the bounds defined by edge_neg and edge_pos.

    Parameters:
        org_masses (np.ndarray): Original masses.
        edge_neg (np.ndarray): Negative mass edges.
        edge_pos (np.ndarray): Positive mass edges.
        where (str): Where to set the initial point, either "org", "neg_edge", or "pos_edge".

    Returns:
        np.ndarray: Initial feasible point.
    """

    if where not in ["org", "neg_edge", "pos_edge"]:
        raise ValueError("where must be either 'org', 'neg_edge', or 'pos_edge'")

    intial_mases = org_masses.copy()
    
    if where == "neg_edge":
        intial_mases = edge_neg.copy()
    elif where == "pos_edge":
        intial_mases = edge_pos.copy()

    if verbose:
        print(f"Initial feasible point set at '{where}': {intial_mases.tolist()}")

    return intial_mases




def hit_and_run_uniform(A, eps, lower, upper, x0, stable_ids, scale, n_samples=20, burn=5, thin=1, scale_coordinates: bool=False, verbose: bool=False) -> np.ndarray:
    """
    Uniform hit-and-run sampler inside convex region:
        A x >= eps, lower <= x <= upper

    Parameters:
        A (csr_matrix): Constraint matrix.
        eps (float): Small positive value for constraints.
        lower (np.ndarray): Lower bounds for each variable.
        upper (np.ndarray): Upper bounds for each variable.
        x0 (np.ndarray): Initial feasible point.
        stable_ids (list): List of indices of variables that should remain fixed.
        scale (np.ndarray): Scale for each variable to sample direction vector.
        n_samples (int): Number of samples to generate.
        burn (int): Number of burn-in iterations.
        thin (int): Thinning factor.
        verbose (bool): If True, print additional information.
    Returns:
        np.ndarray: Array of sampled points.
    """

    if verbose:
        print("Starting hit-and-run sampling...")
        print(f"number of samples: {n_samples}, burn-in: {burn}, thinning: {thin}")
        print(f"total iterations: {n_samples * thin + burn}")

    n = len(x0)
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    x = x0.copy()
    lower_orig = lower.copy()

    if scale_coordinates:
        x = (x - lower) / scale
        eps = eps - A @ lower
        A = A @ diags(scale)
        lower = np.zeros_like(lower)
        upper = np.ones_like(upper)
        if verbose:
            print("\nScaling coordinates to unit cube.")
            print(f"New x0: \n{x}")
            print(f"New lower bounds: \n{lower}")
            print(f"New upper bounds: \n{upper}")
            print(f"New eps with length {len(eps)}: \n{eps}\n")
            print(f"New A matrix:\n{A.toarray()}\n")
            print("Check feasibility in transformed-space:", np.all(A @ x >= eps), np.all((0 <= x) & (x <= 1)))
            violations = (A @ x) - eps
            print("Min constraint residual (should be >=0):", np.min(violations))
            bad = np.where(violations < 0)[0]
            print("Example bad constraints:", bad[:5])
            print("A[bad] @ x =", (A @ x)[bad])
            print("eps[bad]    =", eps[bad], "\n\n")

    samples = []
    fig, ax = plt.subplots(1,1, figsize=(6,6))

    for step in range(n_samples * thin + burn):
        # 1. Choose random direction (normalize to unit length)
        if scale_coordinates:
            d = np.random.normal(size=n)
        else:
            d = np.random.normal(size=n, scale=scale)
        for id in stable_ids:
            d[id] = 0.0
        d /= np.linalg.norm(d)

        # 2. Compute feasible step range [t_min, t_max]
        t_min, t_max = feasible_t_range(A, eps, x, d, lower, upper, verbose=False)

        if t_min > t_max:
            if verbose:
                print("Numerical issue -> skip this step")
            continue

        # 3. Sample t uniformly in [t_min, t_max]
        t = np.random.uniform(t_min, t_max)

        # 4. Move
        x += t * d

        # 5. Collect
        if step >= burn and ((step - burn) % thin == 0):
            if scale_coordinates:
                if verbose and ((step - burn) % (10*thin) == 0):
                    ax.hist(x, bins=10, alpha=0.1, label=f"step {step}")
                x_sample = x * scale + lower_orig
                samples.append(x_sample.copy())
            else:
                samples.append(x.copy())

          
        if verbose and (step + 1) % thin == 0:
            print(f"Iteration {step+1}/{n_samples*thin + burn}")

    ax.legend()
    if verbose:
        fig.savefig("Plots/Debug_sampling/debug_sampling.png")
    plt.close(fig)

    return np.array(samples)




def feasible_t_range(A, eps, x, d, lower, upper, verbose=False) -> tuple[float, float]:
    """
    Compute feasible step range [t_min, t_max] for moving along direction d.
    Ensures A(x + t d) >= eps and lower <= x + t d <= upper.

    Parameters:
        A (csr_matrix): Constraint matrix.
        eps (float): Small positive value for constraints.
        x (np.ndarray): Current point.
        d (np.ndarray): Direction vector.
        lower (np.ndarray): Lower bounds for each variable.
        upper (np.ndarray): Upper bounds for each variable.
        verbose (bool): If True, print additional information.
    Returns:
        tuple[float, float]: Feasible step range (t_min, t_max).
    """
    t_min, t_max = -np.inf, np.inf

    # Linear constraints: A x >= eps
    Ax = A.dot(x)
    Ad = A.dot(d)
    rhs = eps - Ax

    pos = Ad > 1e-12
    neg = Ad < -1e-12
    if np.any(pos):
        t_min = max(t_min, np.max(rhs[pos] / Ad[pos]))
    if np.any(neg):
        t_max = min(t_max, np.min(rhs[neg] / Ad[neg]))

    if verbose:
        print(f"\ncalc steps: \n min: {t_min} \n max: {t_max}")
        if t_min > t_max:
            print("First step issue")

    # Box constraints: lower <= x + t d <= upper
    pos = d > 1e-12
    neg = d < -1e-12
    if np.any(pos):
        t_max = min(t_max, np.min((upper[pos] - x[pos]) / d[pos]))
        t_min = max(t_min, np.max((lower[pos] - x[pos]) / d[pos]))
    if np.any(neg):
        t_max = min(t_max, np.min((lower[neg] - x[neg]) / d[neg]))
        t_min = max(t_min, np.max((upper[neg] - x[neg]) / d[neg]))

    if verbose:
        print(f"\n2calc steps: \n min: {t_min} \n max: {t_max}")

    return t_min, t_max




def mass_edges(p_df: pd.DataFrame, d_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mass edges (upper and lower bounds) for each particle.

    Parameters:
        p_df (pd.DataFrame): DataFrame containing the mass data.
        d_df (pd.DataFrame): DataFrame containing the decay data.

    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bounds for each particle.
    """

    p_df_edge_p = get_edges(p_df, d_df, identifier="mass", edge_id="+", verbose=False)
    p_df_edge_n = get_edges(p_df, d_df, identifier="mass", edge_id="-", verbose=False)

    mass_p = p_df_edge_p["Mass (GeV)"].values
    mass_n = p_df_edge_n["Mass (GeV)"].values

    return mass_n, mass_p




def mean_sample(samples: np.ndarray, scale: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the sampled masses for each particle.

    Parameters:
        samples (np.ndarray): Array of sampled points.
        scale (np.ndarray): Scale values for transformation.
        lower (np.ndarray): Lower bounds for transformation.

    Returns:
        np.ndarray: Mean of the sampled masses for each particle.
    """

    # Shape: (100, 739)
    n_samples, n_particles = samples.shape
    sampled_masses = samples.copy()

    means = []

    for i in range(n_particles):    # len(samples.T[i]) = 739 = n_particles
        print(f"samples: {sampled_masses.T[i][:5]} with lower: {lower[i]:.3f} and scale: {scale[i]:.3f}")
        sampled_masses.T[i] = (sampled_masses.T[i] - lower[i]) / scale[i]
        mean = sampled_masses.T[i].mean()
        means.append(mean)

    
    means_np = samples.mean(axis=0)
    means_rescaled = means * scale + lower

    print(f"\nMean sampled masses (original scale): {means_rescaled.tolist()}")
    print(f"\nMean sampled masses (transformed scale): {means_np.tolist()}")

    return means_rescaled







################################################################
#
# Functions for sampling studies to plot and debug
#
################################################################




def plot_sample(samples: np.ndarray, scale: np.ndarray, lower: np.ndarray, p_df: pd.DataFrame, dir_name: str) -> None:
    """
    Plot histogram of sampled masses for a specific particle.

    Parameters:
        samples (np.ndarray): Array of sampled points.
        particle_id (int): ID of the particle to plot.
        p_df (pd.DataFrame): DataFrame containing the mass data.
        dir_name (str): Directory name to save the plot.
    """

    for i in range(10):
        particle_ids = p_df["ID"].values[(i)*10:(i+1)*10]
        plt.figure(figsize=(8, 6))

        for particle_id in particle_ids:
            particle_index = p_df.index[p_df["ID"] == particle_id][0]
            particle_name = p_df.loc[p_df["ID"] == particle_id, "Name"].values[0]
            sampled_masses = samples[:, particle_index]

            sampled_masses_trans = (sampled_masses - lower[particle_index]) / scale[particle_index]
            mean = sampled_masses_trans.mean()

            plt.hist(sampled_masses_trans, bins=30, alpha=0.1, label=fr"Particle ID {particle_name} with $\mu={mean:.2f}$")

        plt.title(f"Sampled Mass Distribution for Particles")
        plt.xlabel("Mass transformed to [0, 1]")
        plt.xlim(0, 1)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.legend()

        Path(dir_name).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{dir_name}/samples/sampled_mass_particles_{i}.png")
        plt.close()



def plot_sample_2(samples: np.ndarray, scale: np.ndarray, lower: np.ndarray, dir_name: str, edges: tuple[np.ndarray, np.ndarray]) -> None:

    # Shape: (1000, 739)
    n_samples, n_particles = samples.shape
    sampled_masses = samples.copy()

    means = []

    mass_n, mass_p = edges

    mass_p_scaled = (mass_p - lower) / scale
    mass_n_scaled = (mass_n - lower) / scale

    for i in range(n_particles):    # len(samples.T[i]) = 739 = n_particles
        #print(f"samples: {sampled_masses.T[i][:5]} with lower: {lower[i]:.3f} and scale: {scale[i]:.3f}")
        sampled_masses.T[i] = (sampled_masses.T[i] - lower[i]) / scale[i]
        mean = sampled_masses.T[i].mean()
        means.append(mean)
        

    # Simulated data
    x = np.tile(np.arange(n_particles), n_samples)
    y = sampled_masses.flatten()

    # Create 2D histogram
    plt.figure(figsize=(20, 6))
    counts, xedges, yedges, im = plt.hist2d(
        x, y,
        bins=[n_particles, 30],     # 800 bins in x, 30 bins in y
        range=[[0, n_particles], [0, 1]],
        cmap='viridis',
        norm=LogNorm()
    )

    plt.plot(means, color='red', linestyle='--', label='Mean Value', linewidth=1)
    plt.plot(mass_p_scaled, color='purple', linestyle='-', label='Mass Edge +', linewidth=2)
    plt.plot(mass_n_scaled, color='orange', linestyle='-', label='Mass Edge -', linewidth=2)
    plt.legend()

    mean_of_means = np.mean(means)
    plt.text(n_particles * 0.8, 0.9, f'Overall Mean: {mean_of_means:.3f}', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Add color bar for frequency
    cbar = plt.colorbar(im)
    cbar.set_label('Frequency')

    # Labels and title
    plt.xlabel('Particle species')
    plt.ylabel('Mass transformed to [0, 1]')
    plt.title('2D Histogram of Particle Species vs Value')

    plt.tight_layout()
    plt.savefig(f"{dir_name}/sampled_mass_particles_edges2.png")
    plt.close()























































































####################################################
#
# Old versions - ignore - deprecated
#
####################################################


def feasible_t_range_org(A, eps, x, d, lower, upper, verbose=False):
    """
    Compute feasible step range [t_min, t_max] for moving along direction d.
    Ensures A(x + t d) >= eps and lower <= x + t d <= upper.
    """
    t_min, t_max = -np.inf, np.inf

    # (a) constraints A x >= eps  ->  A(x + t d) >= eps
    Ad = A.dot(d)
    Ax = A.dot(x)
    rhs = eps - Ax

    pos = Ad > 1e-12
    neg = Ad < -1e-12
        
    if np.any(pos):
        t_min = max(t_min, np.max(rhs[pos] / Ad[pos]))
    if np.any(neg):
        t_max = min(t_max, np.min(rhs[neg] / Ad[neg]))

    if verbose:
        print(f"\ncalc steps: \n min: {t_min} \n max: {t_max}")
        if t_min > t_max:
            print("First step issue")

    # (b) box bounds lower <= x + t d <= upper
    pos = d > 1e-12
    neg = d < -1e-12
    if np.any(pos):
        t_max = min(t_max, np.min((upper[pos] - x[pos]) / d[pos]))
        if verbose and np.min((upper[pos] - x[pos]) / d[pos]) < 0:
            print(f"pos d: {(upper[pos] - x[pos]) / d[pos]}")
    if np.any(neg):
        t_min = max(t_min, np.min((lower[neg] - x[neg]) / d[neg]))     # pretty sure this should be np.min()
        # if verbose and np.min((lower[neg] - x[neg]) / d[neg]) > 0:     # but still maybe rewrite all to check in a for loop
        #     print(f"neg d: {(lower[neg] - x[neg]) / d[neg]}")

    if verbose:
        print(f"\n2calc steps: \n min: {t_min} \n max: {t_max}")
    
    return t_min, t_max