import pandas as pd
import numpy as np
import pdg




def get_mass_errors(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
    unit = "GeV"
    mass_errors_pos = []
    mass_errors_neg = []
    counter = 0

    for mcid in p_df["ID"]:
        try:
            particle = api.get_particle_by_mcid(mcid)
            counter += 1
            mass = list(particle.masses())[0]
            svm = mass.summary_values()[0]
            mass_errors_pos.append(svm.get_error_positive(units = unit))
            mass_errors_neg.append(svm.get_error_negative(units = unit))
        except:
            mass_errors_pos.append(np.nan)
            mass_errors_neg.append(np.nan)

    print(f"Processed {counter} particles for mass errors.")


    counter_name = 0
    for i, name in enumerate(p_df["Name"]):
        if np.isnan(mass_errors_pos[i]) or np.isnan(mass_errors_neg[i]):
            try:
                particle = api.get_particle_by_name(name)
                counter_name += 1
                mass = list(particle.masses())[0]
                svm = mass.summary_values()[0]
                mass_errors_pos[i] = svm.get_error_positive(units = unit)
                mass_errors_neg[i] = svm.get_error_negative(units = unit)
            except:
                continue

    print(f"Processed {counter_name} particles for mass errors by name.")

    return mass_errors_pos, mass_errors_neg



def get_width_errors(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
    unit = "GeV"
    width_errors_pos = []
    width_errors_neg = []
    counter = 0

    hbar = 6.582119569*10**(-25)
    #print(f"Reduced Planck's constant (hbar): {hbar} GeV·s")

    for mcid in p_df["ID"]:

        try:
            particle = api.get_particle_by_mcid(mcid)
            counter += 1
            width = list(particle.widths())[0]
            svw = width.summary_values()[0]
            w_err_p = svw.get_error_positive(units=unit)
            w_err_n = svw.get_error_negative(units=unit)
            width_errors_pos.append(w_err_p)
            width_errors_neg.append(w_err_n)

        except:

            try:
                particle = api.get_particle_by_mcid(mcid)
                lifetime = list(particle.lifetimes())[0]
                svl = lifetime.summary_values()[0]
                tau = svl.get_value(units="s")
                tau_err_p = svl.get_error_positive(units="s")
                tau_err_n = svl.get_error_negative(units="s")
                width_errors_pos.append((hbar/(tau**2)) * tau_err_p)
                width_errors_neg.append((hbar/(tau**2)) * tau_err_n)
            except:
                width_errors_pos.append(np.nan)
                width_errors_neg.append(np.nan)

            
    print(f"Processed {counter} particles for width errors.")


    counter_name = 0
    for i, name in enumerate(p_df["Name"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            try:
                particle = api.get_particle_by_name(name)
                counter_name += 1
                width = list(particle.widths())[0]
                svw = width.summary_values()[0]
                w_err_p = svw.get_error_positive(units=unit)
                w_err_n = svw.get_error_negative(units=unit)
                width_errors_pos[i] = w_err_p
                width_errors_neg[i] = w_err_n

            except:

                try:
                    particle = api.get_particle_by_name(name)
                    lifetime = list(particle.lifetimes())[0]
                    svl = lifetime.summary_values()[0]
                    tau = svl.get_value(units="s")
                    tau_err_p = svl.get_error_positive(units="s")
                    tau_err_n = svl.get_error_negative(units="s")
                    width_errors_pos[i] = (hbar/(tau**2)) * tau_err_p
                    width_errors_neg[i] = (hbar/(tau**2)) * tau_err_n
                except:
                    continue


    print(f"Processed {counter_name} particles for width errors by name.")

    return width_errors_pos, width_errors_neg




def replace_nan_none(input_list: list[float], default_value=0.0) -> list[float]:
    """
    Replaces all None and np.nan values in a list with default_value.
    
    Parameters:
        input_list (list): List containing numbers, None, or np.nan.

    Returns:
        list: A new list with None and np.nan replaced by default_value.
    """
    counter_None = 0
    counter_nan = 0
    for i in range(len(input_list)):
        if np.isnan(input_list[i]):
            counter_nan += 1
        if input_list[i] is None:
            counter_None += 1
    
    #print(f"Replaced {counter_None} None values with {default_value}.")
    print(f"Replaced {counter_nan} np.nan values with {default_value}.")

    return [default_value if x is None or np.isnan(x) else x for x in input_list]





def post_process(p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process the particle DataFrame by checking for and handling particles/antiparticles.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.

    Returns:
        pd.DataFrame: The post-processed particle DataFrame.
    """
    counter_mass = 0
    counter_width = 0

    for _, particle in p_df.iterrows():

        id = particle["ID"]
        m_err_p = particle["Mass Error Pos (GeV)"]
        m_err_n = particle["Mass Error Neg (GeV)"]
        w_err_p = particle["Width Error Pos (GeV)"]
        w_err_n = particle["Width Error Neg (GeV)"]

        # print(id, m_err_p, m_err_n, w_err_p, w_err_n)

        try:
            m_err_p_anti = p_df.loc[p_df["ID"] == -id, "Mass Error Pos (GeV)"].values[0]
            m_err_n_anti = p_df.loc[p_df["ID"] == -id, "Mass Error Neg (GeV)"].values[0]
            w_err_p_anti = p_df.loc[p_df["ID"] == -id, "Width Error Pos (GeV)"].values[0]
            w_err_n_anti = p_df.loc[p_df["ID"] == -id, "Width Error Neg (GeV)"].values[0]
        except: 
            continue


        if m_err_p_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Mass Error Pos (GeV)"] = m_err_p
            counter_mass += 1
        if m_err_n_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Mass Error Neg (GeV)"] = m_err_n
        if w_err_p_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Width Error Pos (GeV)"] = w_err_p
            counter_width += 1
        if w_err_n_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Width Error Neg (GeV)"] = w_err_n

    # print(counter_mass)
    # print(counter_width)

    return p_df





def get_particle_errors(p_df: pd.DataFrame, api: pdg.api.PdgApi) -> pd.DataFrame:
    """
    Get mass and width errors for particles in the DataFrame.
    Do post-processing on the DataFrame to handle particles and antiparticles and handle None and NaN values.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.
        api (pdg.api.PdgApi): The PDG API instance.

    Returns:
        pd.DataFrame: The DataFrame with mass and width errors added.
    """
    mass_errors_pos, mass_errors_neg = get_mass_errors(p_df, api)
    width_errors_pos, width_errors_neg = get_width_errors(p_df, api)

    mass_errors_pos = replace_nan_none(mass_errors_pos)
    mass_errors_neg = replace_nan_none(mass_errors_neg)
    width_errors_pos = replace_nan_none(width_errors_pos)
    width_errors_neg = replace_nan_none(width_errors_neg)

    p_df_errors = p_df.copy()

    # Add the errors to the DataFrame
    p_df_errors["Mass Error Pos (GeV)"] = mass_errors_pos
    p_df_errors["Mass Error Neg (GeV)"] = mass_errors_neg
    p_df_errors["Width Error Pos (GeV)"] = width_errors_pos
    p_df_errors["Width Error Neg (GeV)"] = width_errors_neg

    # Post-process the DataFrame
    p_df_errors = post_process(p_df_errors)

    return p_df_errors
