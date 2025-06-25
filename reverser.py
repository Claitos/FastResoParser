# Read the file and reverse the lines
with open("decays_PDG2016Plus_massorder_new.dat", "r") as file:
    lines = file.readlines()

# Reverse the list of lines
lines.reverse()

# Write the reversed lines to a new file
with open("decays_PDG2016Plus_massorder_reversed.dat", "w") as file:
    file.writelines(lines)