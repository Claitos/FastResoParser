import pandas as pd
import matplotlib.pyplot as plt


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








# Path to your file
file_path = "decays_PDG2016Plus_massorder_original.dat"

# Parse the file
particles_df, decays_df = parse_to_df(file_path)

stable_particles = particles_df[particles_df["Width (GeV)"] == 0.0]["ID"].tolist()
print(f"Number of stable particles: {len(stable_particles)}")
print(f"Stable particles IDs: {stable_particles}")



# # View the data
# print("Particles DataFrame:")
# print(particles_df.head(n=10))
# print(len(particles_df))

# print("\nDecays DataFrame:")
# print(decays_df.head(n=10))


# Example usage of the helper functions
counter = 0
for particle_id in particles_df["ID"]:
    if decay_to_pion_chain_helper(particle_id):
        # print(f"Particle ID {particle_id} decays into a pion.")
        counter += 1
    else:
        print(f"Particle ID {particle_id} does not decay into a pion.")

print(f"\nTotal particles that decay into a pion: {counter}")


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
# output_path = "decays_PDG2016Plus_massorder_new.dat"

# parse_to_dat(output_path, particles_df, decays_df)







