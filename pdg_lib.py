import pandas as pd
import numpy as np
import pdg


def get_mass_errors(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
    unit = "GeV"
    mass_errors_pos = []
    mass_errors_neg = []

    for mcid in p_df["ID"]:
        try:
            particle = api.get_particle_by_mcid(mcid)
            mass = list(particle.masses())[0]
            svm = mass.summary_values()[0]
            mass_errors_pos.append(svm.get_error_positive(units = unit))
            mass_errors_neg.append(svm.get_error_negative(units = unit))
        except:
            mass_errors_pos.append(np.nan)
            mass_errors_neg.append(np.nan)

    return mass_errors_pos, mass_errors_neg



def get_width_errors(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
    unit = "GeV"
    width_errors_pos = []
    width_errors_neg = []

    hbar = 6.582119569*10**(-25)
    #print(f"Reduced Planck's constant (hbar): {hbar} GeV·s")

    for mcid in p_df["ID"]:
        none_bool = False

        try:
            particle = api.get_particle_by_mcid(mcid)
            width = list(particle.widths())[0]
            svw = width.summary_values()[0]
            w_err_p = svw.get_error_positive(units=unit)
            w_err_n = svw.get_error_negative(units=unit)
            
            if w_err_p is None or w_err_n is None:
                none_bool = True
            else:
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


        if none_bool:
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
            

    return width_errors_pos, width_errors_neg




def replace_nan_none(input_list: list[float], default_value=0.0) -> list[float]:
    """
    Replaces all None and np.nan values in a list with -1.
    
    Parameters:
        input_list (list): List containing numbers, None, or np.nan.

    Returns:
        list: A new list with None and np.nan replaced by -1.
    """
    counter = 0
    for i in range(len(input_list)):
        if input_list[i] is None or np.isnan(input_list[i]):
            counter += 1
    
    print(f"Replaced {counter} None or np.nan values with {default_value}.")

    return [default_value if x is None or np.isnan(x) else x for x in input_list]