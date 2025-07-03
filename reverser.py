inputfile = "decays_PDG2016Plus_massorder_new.dat"
outputfile = "decays_PDG2016Plus_massorder_new_reversed.dat"

# Read the file and reverse the lines
with open(inputfile, "r") as file:
    lines = file.readlines()

# Reverse the list of lines
lines.reverse()

# Write the reversed lines to a new file
with open(outputfile, "w") as file:
    file.writelines(lines)