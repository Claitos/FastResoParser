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
        current_particle = None
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
                current_particle = particle["ID"]
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
                f"{particle['Mass (GeV)']:.5f}",
                f"{particle['Width (GeV)']:.5f}",
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
                    f"{decay['BranchingRatio']:.5f}"
                ] + [str(product) for product in decay["ProductIDs"]])
                f.write(decay_line + "\n")
    
    print(f"Data successfully written to {output_path}")



# Path to your file
file_path = "decays_PDG2016Plus_massorder_original.dat"

# Parse the file
particles_df, decays_df = parse_to_df(file_path)


# View the data
print("Particles DataFrame:")
print(particles_df.head())

print("\nDecays DataFrame:")
print(decays_df.head())

# print("\n")
# print(particles_df["Mass (GeV)"])

plt.plot(particles_df["Mass (GeV)"], marker='o', linestyle='None')
plt.show()



# Output path
output_path = "decays_PDG2016Plus_massorder_new.dat"

parse_to_dat(output_path, particles_df, decays_df)







