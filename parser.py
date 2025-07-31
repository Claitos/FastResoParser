import pandas as pd
import math as m
import matplotlib.pyplot as plt
import numpy as np


def parse_to_df(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses the PDG decay data file and returns two DataFrames:
    one for particles and one for decay channels.
    
    :param file_path: Path to the PDG decay data file.
    :return: Tuple of DataFrames (particles_df, decays_df).
    """

    # Containers for the data
    particles = []
    decays = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        # current_particle = None
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 12:
                # This is a parent particle line
                particle = {
                    "ID": int(parts[0]),
                    "Name": parts[1],
                    "Mass (GeV)": float(parts[2]),
                    "Width (GeV)": float(parts[3]),
                    "Degeneracy": int(parts[4]),
                    "Baryon no.": int(parts[5]),
                    "Strangeness no.": int(parts[6]),
                    "Charm no.": int(parts[7]),
                    "Bottom no.": int(parts[8]),
                    "Isospin": float(parts[9]),
                    "Charge": int(parts[10]),
                    "No. of decay channels": int(parts[11])
                }
                particles.append(particle)
                # current_particle = particle["ID"]
            elif len(parts) == 8:
                # This is a decay channel line
                decay = {
                    "ParentID": int(parts[0]),
                    "No. of daughter particles": int(parts[1]),
                    "BranchingRatio": float(parts[2]),
                    "ProductIDs": [int(p) for p in parts[3:] if p.strip() != ""]
                }
                decays.append(decay)

    # Convert to DataFrames
    particles_df = pd.DataFrame(particles)
    decays_df = pd.DataFrame(decays)

    return particles_df, decays_df

def parse_to_dat(output_path: str, particles_df: pd.DataFrame, decays_df: pd.DataFrame):
    """
    Writes the parsed DataFrames to a new file in the original format.
    
    :param output_path: Path to the output file.
    :param particles_df: DataFrame containing particle data.
    :param decays_df: DataFrame containing decay channel data.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for _, particle in particles_df.iterrows():
            # Write particle header line
            particle_line = "\t".join([
                str(particle["ID"]),
                particle["Name"],
                f"{particle['Mass (GeV)']:.9f}",
                f"{particle['Width (GeV)']:.9f}",
                str(particle["Degeneracy"]),
                str(particle["Baryon no."]),
                str(particle["Strangeness no."]),
                str(particle["Charm no."]),
                str(particle["Bottom no."]),
                f"{particle['Isospin']:.1f}",
                str(particle["Charge"]),
                str(particle["No. of decay channels"])
            ])
            f.write(particle_line + "\n")

            # Write associated decay lines
            decay_rows = decays_df[decays_df["ParentID"] == particle["ID"]]
            for _, decay in decay_rows.iterrows():
                decay_line = "\t".join([
                    str(decay["ParentID"]),
                    str(decay["No. of daughter particles"]),
                    f"{decay['BranchingRatio']:.9f}"
                ] + [str(product) for product in decay["ProductIDs"]])
                f.write(decay_line + "\n")
    
    print(f"Data successfully written to {output_path}")





def delete_particle_helper(particle_id: int):
    """
    Deletes a particle and its associated decay channels from the DataFrames. 
    This function assumes that the DataFrames are global variables.
    Be careful with this operation as it modifies the DataFrames directly. 
    So they are not passed as arguments (mind the exact variable name) and no copied dataframes are returned.
    
    :param particle_id: The ID of the particle to delete.
    :raises NameError: If particles_df or decays_df is not defined.
    :raises TypeError: If particles_df or decays_df is not a DataFrame.
    :return: None
    """

    if 'particles_df' in globals():
        if not isinstance(particles_df, pd.DataFrame):
            raise TypeError("particles_df is not a DataFrame.")
    else:
        raise NameError("particles_df is not defined. Make sure to run the parser first.")
    
    if 'decays_df' in globals():
        if not isinstance(decays_df, pd.DataFrame):
            raise TypeError("decays_df is not a DataFrame.")
    else:
        raise NameError("decays_df is not defined. Make sure to run the parser first.")

    # Remove the particle
    particles_df.drop(particles_df[particles_df["ID"] == particle_id].index, inplace=True)
    
    # Remove associated decay channels
    decays_df.drop(decays_df[decays_df["ParentID"] == particle_id].index, inplace=True)


def delete_particle_list_helper(particle_ids: list[int]):
    particle_list = []
    for particle_id in particle_ids:
        decay_list, _ = decay_chain_helper(particle_id)
        decay_paths = get_value_paths(decay_list)
        particle_list.append(particle_id)  # Add the particle ID itself
        for decay_path in decay_paths:
            decay_steps = decay_path.split('_')         # Split the decay path into steps
            decay_steps_int = [int(step) for step in decay_steps]   # Convert steps to integers
            particle_list.extend(decay_steps_int)

    # print(f"Particle steps to keep: {particle_list}")
    particle_list_unique = list(set(particle_list))  # Remove duplicates
    # print(f"Unique particle steps to keep: {particle_list_unique}")


    if 'particles_df' not in globals():
        raise NameError("particles_df is not defined. Make sure to run the parser first.")
    
    if 'decays_df' not in globals():
        raise NameError("decays_df is not defined. Make sure to run the parser first.")
    

    # Filter the particles_df and decays_df to keep only the particles in particle_list_unique
    global particles_df, decays_df  # Declare as global to modify in this function
    particles_df = particles_df[particles_df["ID"].isin(particle_list_unique)].reset_index(drop=True)
    decays_df = decays_df[decays_df["ParentID"].isin(particle_list_unique)].reset_index(drop=True)  





def decay_to_pion_helper(particle_id: int) -> bool:
    """ 
    Checks if a particle decays into a pion (or antipion).
    This function assumes that the particle ID is valid and exists in the decays DataFrame.
    It returns True if the particle decays into a pion, False otherwise.

    :param particle_id: The ID of the particle to check.
    :return: True if the particle decays into a pion, False otherwise.
    :raises NameError: If the particle ID is not found in the decays DataFrame
    """

    pion_id = 211  # PDG ID for pion

    if particle_id == pion_id or particle_id == -pion_id:
        return False  # A pion does not decay into itself

    decays_of_particle = decays_df[decays_df["ParentID"] == particle_id]  # Filter decays for the given particle ID
    if decays_of_particle.empty:
        NameError(f"Particle with ID {particle_id} not found in decays DataFrame.")


    default = False  # Default value to return if no pion decay is found
    for _, decay in decays_of_particle.iterrows():
        # Check if the pion ID is in the decay products
        if pion_id in decay["ProductIDs"] or -pion_id in decay["ProductIDs"]:
            default = True

    return default


def decay_to_pion_chain_helper(particle_id: int) -> bool:

    default = False  # Default value to return if no pion decay is found

    if decay_to_pion_helper(particle_id):
        default = True
    else:
        # Check if the particle decays into other particles that eventually decay into a pion
        decays_of_particle = decays_df[decays_df["ParentID"] == particle_id]
        for _, decay in decays_of_particle.iterrows():
            for product_id in decay["ProductIDs"]:
                if product_id in stable_particles:
                    continue
                if decay_to_pion_chain_helper(product_id):
                    default = True
                    break
            if default:
                break


    return default





def list_depth(lst: list) -> int:
    if not isinstance(lst, list):
        return 0
    return 1 + max((list_depth(item) for item in lst), default=0)


def get_value_paths(data:list, prefix='', sep="_") -> list:
    """
    Recursively retrieves paths to all integer values in a nested list structure.
    Each path is represented as a string with dot notation.

    :param data: The nested list structure to search.
    :param prefix: The current path prefix (used for recursion).
    :param sep: The separator to use between path elements (default is underscore).
    :return: A list of paths to integer values.
    """
    
    paths = []
    i = 0
    while i < len(data):
        item = data[i]

        if isinstance(item, (int, float)):
            # If followed by a list → descend
            if i + 1 < len(data) and isinstance(data[i + 1], list):
                new_prefix = f"{prefix}{sep}{item}" if prefix else str(item)
                paths.extend(get_value_paths(data[i + 1], new_prefix, sep))
                i += 2
            else:
                # Standalone int → treat as leaf
                full_path = f"{prefix}{sep}{item}" if prefix else str(item)
                paths.append(full_path)
                i += 1

        elif isinstance(item, list):
            # List not preceded by int (rare in your case), descend flat
            paths.extend(get_value_paths(item, prefix, sep))
            i += 1

        else:
            i += 1

    return paths


def decay_chain_helper(particle_id: int) -> tuple[list, list]:
    """ 
    Checks the branching ratios of a particle's decay chain to see if it decays into stable particles.
    This function assumes that the particle ID is valid and exists in the decays DataFrame.
    It returns two lists of stable particle IDs and their branching ratios if they are found in the decay chain.

    :param particle_id: The ID of the particle to check.
    :return: A tuple containing two lists:
             - decay_chain: List of stable particle IDs in the decay chain.
             - branching_ratios: List of branching ratios corresponding to the decay chain.
    """

    decays_of_particle = decays_df[decays_df["ParentID"] == particle_id]  # Filter decays for the given particle ID
    decay_chain = [] # List to store branching ratios of decay
    branching_ratios = []  # List to store branching ratios of decay

    for _, decay in decays_of_particle.iterrows():
        for product_id in decay["ProductIDs"]:
            if product_id == 0:
                continue
            elif product_id in stable_particles:
                decay_chain.append(product_id)  # Add stable particle ID to the decay chain
                branching_ratios.append(decay["BranchingRatio"])
                continue
            else:
                decay_chain.append(product_id)  # Add the product ID to the decay chain
                branching_ratios.append(decay["BranchingRatio"])
                child_decay_chain, child_branching_ratios = decay_chain_helper(product_id)
                if child_decay_chain:
                    decay_chain.append(child_decay_chain)
                    branching_ratios.append(child_branching_ratios)

    return decay_chain, branching_ratios


def branchratio_of_particle_to_pions(particle_id: int) -> tuple[list, list]:
    """
    Checks the decay chain of a particle to see if it decays into pions (or antipions).
    This function assumes that the particle ID is valid and exists in the decays DataFrame.
    It returns two lists:
    - decays_to_pions: List of decay paths that lead to pions.
    - branching_ratios_to_pions: List of branching ratios corresponding to the decay paths to pions.    
    """
    test_ids = (211,)  # PDG ID for pion
    #test_ids = (211, -211)  # PDG IDs for pion and antipion
    test_ids_str = tuple(str(x) for x in test_ids)  # Convert to string
    test_ids_str_prefix = tuple("_" + str(x) for x in test_ids)  # Convert to string with prefix
    decay_list, branching_ratios = decay_chain_helper(particle_id)
    decay_paths = get_value_paths(decay_list)
    branching_ratios_paths = get_value_paths(branching_ratios)

    if len(decay_paths) != len(branching_ratios_paths):
        raise ValueError("Mismatch between the lengths of decay paths and branching ratios paths.")

    decays_to_pions = [] # List to store decay paths that lead to pions
    branching_ratios_to_pions = []  # List to store branching ratios corresponding to decay paths to pions
    for i in range(len(decay_paths)):
        if decay_paths[i].endswith(test_ids_str_prefix) or decay_paths[i] in test_ids_str:
            br_steps = branching_ratios_paths[i].split('_')
            br_steps_float = [float(x) for x in br_steps]
            product_br = m.prod(br_steps_float)
            # print(f"Particle ID {particle_id} decays into a pion through path: {decay_paths[i]} with branching ratios {branching_ratios_paths[i]} and product branching ratio {product_br:.9f}")
            # print(f"Particle ID {particle_id} decays into a pion through path: {decay_paths[i]} with branching ratio {product_br:.9f}")

            decays_to_pions.append(decay_paths[i])
            branching_ratios_to_pions.append(product_br)

    return decays_to_pions, branching_ratios_to_pions


def importance_score(particle_id: int) -> float:
    decay_chain_to_pion, branching_ratio_to_pion = branchratio_of_particle_to_pions(particle_id)
    total_br = sum(branching_ratio_to_pion)
    mass = particles_df.loc[particles_df["ID"] == particle_id, "Mass (GeV)"].values[0]
    degeneracy = particles_df.loc[particles_df["ID"] == particle_id, "Degeneracy"].values[0]
    #print(f"Particle ID {particle_id} has mass {mass:.9f} GeV, degeneracy {degeneracy}, and total branching ratio to pions {total_br:.9f}")
    temperature = 0.140  # This can be adjusted based on the system's temperature
    exponential_factor = np.exp(-mass/temperature)  # This can be adjusted based on the importance of the particle
    # print(f"Exponential factor for particle ID {particle_id}: {exponential_factor:.9f}")
    importance = total_br * exponential_factor * degeneracy
    return importance








def main():


    # Path to your file
    file_path = "decays_PDG2016Plus_massorder_original.dat"
    # file_path = "decays_QM2016Plus_massorder.dat"

    # Parse the file
    global particles_df, decays_df  # Declare as global to modify in helper functions
    particles_df, decays_df = parse_to_df(file_path)

    global stable_particles  # Declare as global to use in helper functions
    stable_particles_test = particles_df[particles_df["Width (GeV)"] == 0.0]
    stable_particles = stable_particles_test[stable_particles_test["No. of decay channels"] == 1]["ID"].tolist()
    # print(f"Number of stable particles: {len(stable_particles)}")
    # print(f"Stable particles IDs: {stable_particles}")



    # # View the data
    # print("Particles DataFrame:")
    # print(particles_df.head(n=10))
    # print(f"total number of particles : {len(particles_df)}")

    # print("\nDecays DataFrame:")
    # print(decays_df.head(n=10))
    # print()



    # # Example usage of the helper functions
    # counter = 0
    # counter_2 = 0
    # for particle_id in particles_df["ID"]:
    #     counter_2 += 1
    #     if decay_to_pion_chain_helper(particle_id):
    #         # print(f"Particle ID {particle_id} decays into a pion.")
    #         counter += 1
    #     else:
    #         print(f"Particle ID {particle_id} does not decay into a pion.")

    # print(f"\nTotal particles checked: {counter_2}")
    # print(f"\nTotal particles that decay into a pion: {counter}")




    # Example usage of the list_depth function to check the depth of decay chains
    # list_depth_values = []
    # for particle_id in particles_df["ID"]:
    #     decay_chain, branching_ratios = decay_chain_helper(particle_id)
    #     paths = get_value_paths(decay_chain)
    #     paths_BR = get_value_paths(branching_ratios)
    #     dict_depth_value = list_depth(decay_chain)
    #     list_depth_values.append(dict_depth_value)
    #     print(f"Depth of decay chain for particle ID {particle_id}: {dict_depth_value}")
    # print(f"Maximum depth of decay chains: {max(list_depth_values)}")




    # Example usage of the decay_chain_helper function
    test_id = 331   #eta prime id = 331
    # print()
    # decay, br = decay_chain_helper(test_id)
    # print(f"Decay chain for particle ID {test_id}: {decay}")
    # print(f"Branching ratios for particle ID {test_id}: {br}")
    # paths = get_value_paths(decay)
    # br_paths = get_value_paths(br)
    # print(f"Paths in decay chain for particle ID {test_id}: {paths}")
    # print(f"Branching ratios paths for particle ID {test_id}: {br_paths}")
    # print(f"length of paths: {len(paths)} and length of branching ratios: {len(br_paths)}")
    # dict_depth_value = list_depth(decay)
    # print(f"Depth of decay chain for particle ID {test_id}: {dict_depth_value}")




    # Example usage of the branchratio_of_particle_to_pions function
    decay_chain_to_pion, branching_ratio_to_pion = branchratio_of_particle_to_pions(test_id)

    combined = list(zip(decay_chain_to_pion, branching_ratio_to_pion))
    combined.sort(key=lambda x: x[1], reverse=True)  # Sort by branching ratio in descending order
    for decay_path, br in combined:
        print(f"Decay chain to pions for particle ID {test_id}:  Path: {decay_path}, Branching Ratio: {br:.9f}")   
    
    total_br = sum(branching_ratio_to_pion)
    print(f"Total branching ratio for particle ID {test_id} decaying into pions: {total_br:.9f}")
    print(f"Number of decay paths to pions for particle ID {test_id}: {len(decay_chain_to_pion)}")




    # Loop to check total branching ratios for all particles to pions
    # total_brs = []
    # for particle_id in particles_df["ID"]:
    #     decay_chain_to_pion, branching_ratio_to_pion = branchratio_of_particle_to_pions(particle_id)
    #     total_br = sum(branching_ratio_to_pion)
    #     if total_br > 0:
    #         print(f"Particle ID {particle_id} decays into pions with total branching ratio: {total_br:.9f}")
    #         total_brs.append(total_br)
    #     else:
    #         print(f"Particle ID {particle_id} does not decay into pions.")
    #         total_brs.append(0.0)

    # plt.plot(total_brs, marker='.', linestyle='None')
    # plt.xlabel("Particle Number in the list")
    # plt.ylabel(r"Total Branching Ratio to $\pi^+$")
    # plt.title(r"Total Branching Ratio to $\pi^+$ for all Particles")
    #plt.show()
    #plt.savefig("Plots/total_branching_ratio_to_pionplu.png", dpi = 300)




    # Example usage of the importance_score function
    # particle_id = 331  # Example particle ID
    # importance = importance_score(particle_id)
    # print(f"Importance score for particle ID {particle_id}: {importance:.9f}")

    # # # Loop to calculate importance scores for all particles
    importance_scores = []
    for particle_id in particles_df["ID"]:
        importance = importance_score(particle_id)
        importance_scores.append(importance)

    plt.plot(importance_scores, marker='.', linestyle='None')
    plt.xlabel("Particle Number in the list")
    plt.ylabel("Importance Score")
    plt.title("Importance Score = BR * exp(-mass/T) * degeneracy")
    plt.yscale('symlog', linthresh=1e-12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    #plt.savefig("Plots/importance_score_all_particles.png", dpi = 300)




    # Example usage of the delete_particle_list_helper function

    cut = 0.0001  # Threshold for importance score
    important_particles = []
    for particle_id in particles_df["ID"]:
        importance = importance_score(particle_id)
        if importance > cut:  # Threshold for importance
            important_particles.append(particle_id)

    print(f"Important particles (importance > {cut}): {important_particles}")

    #delete_particle_list_helper(important_particles)
        
    # print("Particles DataFrame:")
    # print(particles_df.head(n=10))
    # print(f"total number of particles of deletion : {len(particles_df)}")
    # print("\nDecays DataFrame:")
    # print(decays_df.head(n=10))
    # print()



    # Example of deleting a particle
    # delete_particle_helper(2001034)  
    # delete_particle_helper(2001033) 
    # print("Particles DataFrame:")
    # print(particles_df.head(n=10))
    # print("\nDecays DataFrame:")
    # print(decays_df.head(n=10))


    # print("\n")
    # print(particles_df["Mass (GeV)"])

    # plt.plot(particles_df["Mass (GeV)"], marker='o', linestyle='None')
    # plt.show()



    # Output path
    #output_path = f"decays_PDG2016Plus_massorder_{cut}.dat"

    #parse_to_dat(output_path, particles_df, decays_df)

    return 0



if __name__ == "__main__":
    main()






