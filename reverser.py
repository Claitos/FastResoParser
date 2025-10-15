
def reverser_routine(cuts: list[float] = [1e-02], dir_name: str = "cuts_test", cut_name: bool = True):
    """
    Reverses the lines of the specified file for a given cut and directory name.
    
    Args:
        cut (list[float]): The cut value to be used in the file name.
        dir_name (str): The directory name where the input file is located.
    """
    for i, cut in enumerate(cuts):

        if cut_name == True:
            inputfile = f"{dir_name}/decays_PDG2016Plus_massorder_{cut}.dat"
            outputfile = f"{dir_name}/decays_PDG2016Plus_massorder_{cut}_reversed.dat"
        else:
            inputfile = f"{dir_name}/decays_PDG2016Plus_massorder_{i+1}.dat"
            outputfile = f"{dir_name}/decays_PDG2016Plus_massorder_{i+1}_reversed.dat"

        with open(inputfile, "r") as file:
            lines = file.readlines()

        lines.reverse()

        with open(outputfile, "w") as file:
            file.writelines(lines)

        print(f"Reversed file created: {outputfile}")


def reverser_routine_new(dir_name: str = "cuts_test", no_lists: int = 1):
    """
    Reverses the lines of the specified files in the given directory.
    
    Args:
        dir_name (str): The directory name where the input files are located.
        no_lists (int): The number of files to be processed.
    """
    for i in range(no_lists):
        
        inputfile = f"{dir_name}/PDG2016Plus_{i}.dat"
        outputfile = f"{dir_name}/PDG2016Plus_{i}_reversed.dat"

        with open(inputfile, "r") as file:
            lines = file.readlines()

        lines.reverse()

        with open(outputfile, "w") as file:
            file.writelines(lines)

        print(f"Reversed file created: {outputfile}")


def main():

    cut = 1e-09
    dir_name = "cuts_pi+stable"
    inputfile = f"{dir_name}/decays_PDG2016Plus_massorder_{cut}.dat"
    outputfile = f"{dir_name}/decays_PDG2016Plus_massorder_{cut}_reversed.dat"

    # Read the file and reverse the lines
    with open(inputfile, "r") as file:
        lines = file.readlines()

    # Reverse the list of lines
    lines.reverse()

    # Write the reversed lines to a new file
    with open(outputfile, "w") as file:
        file.writelines(lines)


if __name__ == "__main__":
    main()