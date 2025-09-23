import pandas as pd
import numpy as np
import pdg
import re
from collections import Counter
from pathlib import Path




#################################################################
#
#
#           Functions to extract mass and width errors of particles
#
#
#################################################################





def get_values_text(summary_value, verbose: bool = False):
    unit_dict = {"MeV": 1e-3}
    text = summary_value.value_text
    unit = summary_value.units

    text_numbers = re.findall(r"\d+", text)
    numbers_converted = [float(num) * unit_dict[unit] for num in text_numbers]

    if len(text_numbers) == 0:
        if verbose:
            print("No valid numbers found.")
        return np.nan, np.nan
    
    if len(text_numbers) == 2:
        error_pos = (numbers_converted[1] - numbers_converted[0]) / 2
        error_neg = (numbers_converted[1] - numbers_converted[0]) / 2
        if verbose:
            print(f"text_error positive: {round(error_pos, 10)}, text_error negative: {round(error_neg, 10)}")
        return round(error_pos, 10), round(error_neg, 10)

    if len(text_numbers) == 3:
        error_pos = numbers_converted[2] - numbers_converted[1]
        error_neg = numbers_converted[1] - numbers_converted[0]
        if verbose:
            print(f"text_error positive: {round(error_pos, 10)}, text_error negative: {round(error_neg, 10)}")
        return round(error_pos, 10), round(error_neg, 10)

    return np.nan, np.nan




def get_error_helper(identifier, api: pdg.api.PdgApi, property: str = "mass", call_type: str = "id", verbose: bool = False) -> tuple[float, float]:

    try:
        if call_type == "id":
            particle = api.get_particle_by_mcid(identifier)
        elif call_type == "name":
            particle = api.get_particle_by_name(identifier)
        else:
            raise ValueError(f"Unknown call_type: {call_type}")
    except:
        if verbose:
            print(f"Particle not found for identifier: {identifier} with call_type: {call_type}")
        return np.nan, np.nan

    

    if property == "mass":
        property_infos = list(particle.masses(require_summary_data=False))
    elif property == "width":
        property_infos = list(particle.widths(require_summary_data=False))
    elif property == "lifetime":
        property_infos = list(particle.lifetimes(require_summary_data=False))
    else:
        raise ValueError(f"Unknown property: {property}")

    if not property_infos:
        if verbose:
            print(f"No property information found for particle: {identifier}")
        return np.nan, np.nan

    if verbose:
        print(f"len of property_infos: {len(property_infos)}")


    error_pos, error_neg = None, None

    for info in property_infos[::-1]:   # Reverse order to end on the possible best info
        summary_values = info.summary_values(summary_table_only=False)
        if not summary_values:
            continue
        if verbose:
            print(f"len of summary values: {len(summary_values)}")

        for summary_value in summary_values[::-1]:   # Reverse order to end on the possible best value
            if property == "mass" or property == "width":
                unit = "GeV"
                try:
                    error_pos = summary_value.get_error_positive(units=unit)
                    error_neg = summary_value.get_error_negative(units=unit)
                except:
                    if verbose:
                        print(f"Error retrieval failed for property: {property} of particle: {identifier}")
                        print(f"Value text: {summary_value.value_text} {summary_value.units}     display: {summary_value.display_value_text}")
                        if property == "width":
                            error_pos, error_neg = get_values_text(summary_value, verbose=verbose)
                    continue

            elif property == "lifetime":
                hbar = 6.582119569*10**(-25)
                unit = "s"
                try:
                    tau_value = summary_value.get_value(units=unit)
                    tau_error_pos = summary_value.get_error_positive(units=unit)
                    tau_error_neg = summary_value.get_error_negative(units=unit)
                except:
                    if verbose:
                        print(f"Error retrieval failed for property: {property} of particle: {identifier}")
                        print(f"Value text: {summary_value.value_text} {summary_value.units}     display: {summary_value.display_value_text}")
                    continue
                error_pos = (hbar / (tau_value**2)) * tau_error_pos
                error_neg = (hbar / (tau_value**2)) * tau_error_neg

            if not error_pos and not error_neg:
                continue


    if not error_pos and not error_neg:
        return np.nan, np.nan

    return (error_pos, error_neg)




def get_mass_errors(p_df :pd.DataFrame, api: pdg.api.PdgApi, verbose: bool = False) -> tuple[list[float], list[float]]:
    mass_errors_pos = [np.nan] * len(p_df)
    mass_errors_neg = [np.nan] * len(p_df)

    formatted_names = format_names(p_df, verbose=verbose)

    if verbose:
        print("\n\n\n-------------Getting mass errors by id----------------\n\n\n")

    for i, mcid in enumerate(p_df["ID"]):
        if np.isnan(mass_errors_pos[i]) or np.isnan(mass_errors_neg[i]):
            if verbose:
                print(f"\nGetting mass errors for MCID {mcid}")
            error_pos, error_neg = get_error_helper(mcid, api, property="mass", call_type="id", verbose=verbose)
            if verbose:
                print(f"Got mass errors for MCID {mcid}: +{error_pos}, -{error_neg}")
            mass_errors_pos[i] = error_pos
            mass_errors_neg[i] = error_neg

    if verbose:
        print("\n\n\n-------------Getting mass errors by name----------------\n\n\n")

    for i, name in enumerate(formatted_names):
        if np.isnan(mass_errors_pos[i]) or np.isnan(mass_errors_neg[i]):
            if verbose:
                print(f"\nGetting mass errors for name {name}")
            error_pos, error_neg = get_error_helper(name, api, property="mass", call_type="name", verbose=verbose)
            if verbose:
                print(f"Got mass errors for name {name}: +{error_pos}, -{error_neg}")
            mass_errors_pos[i] = error_pos
            mass_errors_neg[i] = error_neg

    return mass_errors_pos, mass_errors_neg


def get_width_errors(p_df :pd.DataFrame, api: pdg.api.PdgApi, verbose: bool = False) -> tuple[list[float], list[float]]:
    width_errors_pos = [np.nan] * len(p_df)
    width_errors_neg = [np.nan] * len(p_df)

    formatted_names = format_names(p_df, verbose=False)

    if verbose:
        print("\n\n\n-------------Getting width errors by id----------------\n\n\n")

    for i, mcid in enumerate(p_df["ID"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            if verbose:
                print(f"\nGetting width errors for MCID {mcid}")
            error_pos, error_neg = get_error_helper(mcid, api, property="width", call_type="id", verbose=verbose)
            if verbose:
                print(f"Got width errors for MCID {mcid}: +{error_pos}, -{error_neg}")
            width_errors_pos[i] = error_pos
            width_errors_neg[i] = error_neg

    if verbose:
        print("\n\n\n-------------Getting width errors by name----------------\n\n\n")

    for i, name in enumerate(formatted_names):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            if verbose:
                print(f"\nGetting width errors for name {name}")
            error_pos, error_neg = get_error_helper(name, api, property="width", call_type="name", verbose=verbose)
            if verbose:
                print(f"Got width errors for name {name}: +{error_pos}, -{error_neg}")
            width_errors_pos[i] = error_pos
            width_errors_neg[i] = error_neg

    if verbose:
        print("\n\n\n-------------Getting lifetime errors by id----------------\n\n\n")

    for i, mcid in enumerate(p_df["ID"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            if verbose:
                print(f"\nGetting lifetime errors for MCID {mcid}")
            error_pos, error_neg = get_error_helper(mcid, api, property="lifetime", call_type="id", verbose=verbose)
            if verbose:
                print(f"Got lifetime errors for MCID {mcid}: +{error_pos}, -{error_neg}")
            width_errors_pos[i] = error_pos
            width_errors_neg[i] = error_neg

    if verbose:
        print("\n\n\n-------------Getting lifetime errors by name----------------\n\n\n")

    for i, name in enumerate(formatted_names):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            if verbose:
                print(f"\nGetting lifetime errors for name {name}")
            error_pos, error_neg = get_error_helper(name, api, property="lifetime", call_type="name", verbose=verbose)
            if verbose:
                print(f"Got lifetime errors for name {name}: +{error_pos}, -{error_neg}")
            width_errors_pos[i] = error_pos
            width_errors_neg[i] = error_neg


    return width_errors_pos, width_errors_neg






def replace_nan_none(input_list: list[float], default_value=0.0) -> list[float]:
    """
    Replaces all None and np.nan values in a list with default_value.
    
    Parameters:
        input_list (list): List containing numbers, None, or np.nan.

    Returns:
        list: A new list with None and np.nan replaced by default_value.
    """
    # counter_None = 0
    # counter_nan = 0
    # for i in range(len(input_list)):
    #     #print(f"input_list[{i}] = {input_list[i]}")
    #     if np.isnan(input_list[i]):
    #         counter_nan += 1
    #     if input_list[i] is None:
    #         counter_None += 1
    
    # #print(f"Replaced {counter_None} None values with {default_value}.")
    # print(f"Replaced {counter_nan} np.nan values with {default_value}.")

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
            if m_err_p != 0.0:
                counter_mass += 1
        if m_err_n_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Mass Error Neg (GeV)"] = m_err_n
        if w_err_p_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Width Error Pos (GeV)"] = w_err_p
            if w_err_p != 0.0:
                counter_width += 1
        if w_err_n_anti == 0.0:
            p_df.loc[p_df["ID"] == -id, "Width Error Neg (GeV)"] = w_err_n

    print("\nProcessed anti-particle mass errors:", counter_mass)
    print("Processed anti-particle width errors:", counter_width)

    return p_df


def format_names(p_df: pd.DataFrame, verbose: bool = False) -> list[str]:
    """
    Format the names of the particles in the DataFrame.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.

    Returns:
        list[str]: A list of formatted particle names.
    """

    lookup_dict = {
        "pi2(1880)+": "pi_2(1880)0",
        "Omega(2250)": "Omega(2250)-",
        "Lambda(2350)": "Lambda(2350)0",
        "Ksi": "Xi",
        "rho3": "rho_3",
        "rho5": "rho_5",
        "eta2": "eta_2",
        "f0": "f_0",
        "f1": "f_1",
        "f2": "f_2",
        "f4": "f_4",
        "f6": "f_6",
        "a0": "a_0",
        "a1": "a_1",
        "a2": "a_2",
        "a4": "a_4",
        "h1": "h_1",
        "pi1": "pi_1",
        "pi2": "pi_2",
        "K0*": "K_0^*",
        "K0": "K_0",
        "K1": "K_1",
        "K2*": "K_2^*",
        "K2": "K_2",
        "K3": "K_3",
        "K4": "K_4",
        "K5": "K_5"
    }

    formatted_names = []
    for _, particle in p_df.iterrows():
        name = particle["Name"]
        #print(type(name))  ->  str
        for key, value in lookup_dict.items():
            name = name.replace(key, value)
        formatted_names.append(name)

    if verbose:
        print(f"Formatted particle names: {formatted_names}")
        print(f"Number of formatted names: {len(formatted_names)} \n")

    return formatted_names



def get_particle_errors(p_df: pd.DataFrame, api: pdg.api.PdgApi, apply_corrections: bool = False) -> pd.DataFrame:
    """
    Get mass and width errors for particles in the DataFrame.
    Do post-processing on the DataFrame to handle particles and antiparticles and handle None and NaN values.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.
        api (pdg.api.PdgApi): The PDG API instance.

    Returns:
        pd.DataFrame: The DataFrame with mass and width errors added.
    """
    mass_errors_pos, mass_errors_neg = get_mass_errors(p_df, api, verbose=True)
    width_errors_pos, width_errors_neg = get_width_errors(p_df, api, verbose=True)

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

    if apply_corrections:
        correction_factor = 0.5
        p_df_errors = bad_evidence_correction(p_df_errors, api, correction_factor=correction_factor, all=True, verbose=True)

        p_df_errors = post_process(p_df_errors)

    return p_df_errors






def bad_evidence_helper(id:str, name:str, api: pdg.api.PdgApi, property: str = "mass", verbose: bool = False) -> bool:

    try:
        particle = api.get_particle_by_mcid(id)
    except:
        try: 
            particle = api.get_particle_by_name(name)
        except:
            if verbose:
                print(f"Particle not found for ID: {id} and Name: {name}")
            return False


    if property == "mass":
        property_infos = list(particle.masses(require_summary_data=False))
    elif property == "width":
        property_infos = list(particle.widths(require_summary_data=False))
    else:
        raise ValueError(f"Unknown property: {property}")

    if not property_infos:
        if verbose:
            print(f"No property information found for particle: {id} / {name}")
        return True

    if verbose:
        print(f"len of property_infos: {len(property_infos)}")


    default_return = False

    for info in property_infos:
        if not info.has_best_summary():
            default_return = True

    return default_return



def bad_evidence_correction(p_df: pd.DataFrame, api: pdg.api.PdgApi, correction_factor: float = 1.0, all: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Apply bad evidence correction to the particle DataFrame.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.
        api (pdg.api.PdgApi): The PDG API instance.

    Returns:
        pd.DataFrame: The DataFrame with bad evidence correction applied.
    """

    formatted_names = format_names(p_df, verbose=verbose)
    counter_mass = 0
    counter_width = 0
    stable_particles_test = p_df[p_df["Width (GeV)"] == 0.0]
    stable_particles = stable_particles_test[stable_particles_test["No. of decay channels"] == 1]["ID"].tolist()

    for i, particle in p_df.iterrows():

        id = particle["ID"]
        name = formatted_names[i]
        mass = particle["Mass (GeV)"]
        width = particle["Width (GeV)"]
        mass_err_p = particle["Mass Error Pos (GeV)"]
        mass_err_n = particle["Mass Error Neg (GeV)"]
        width_err_p = particle["Width Error Pos (GeV)"]
        width_err_n = particle["Width Error Neg (GeV)"]

        if id in stable_particles:
            continue

        if mass_err_p == 0.0 and mass_err_n == 0.0:
            if verbose:
                print(f"\n Processing particle for mass: {name} / {id}")

            has_bad_evidence = bad_evidence_helper(id, name, api, property="mass", verbose=verbose)

            if all:
                has_bad_evidence = True

            if has_bad_evidence:
                counter_mass += 1
                if verbose:
                    print(f"Applying bad evidence correction for mass of particle {name}")
                p_df.loc[p_df["ID"] == id, "Mass Error Pos (GeV)"] = correction_factor * mass
                p_df.loc[p_df["ID"] == id, "Mass Error Neg (GeV)"] = correction_factor * mass


        if width_err_p == 0.0 and width_err_n == 0.0:
            if verbose:
                print(f"Processing particle for width: {name} / {id}")

            has_bad_evidence = bad_evidence_helper(id, name, api, property="width", verbose=verbose)

            if all:
                has_bad_evidence = True

            if has_bad_evidence:
                counter_width += 1
                if verbose:
                    print(f"Applying bad evidence correction for width of particle {name}")
                p_df.loc[p_df["ID"] == id, "Width Error Pos (GeV)"] = correction_factor * width
                p_df.loc[p_df["ID"] == id, "Width Error Neg (GeV)"] = correction_factor * width

    if verbose:
        print(f"Bad evidence correction applied to {counter_mass} particles (mass) and {counter_width} particles (width) \n")

    return p_df




#################################################################
#
#
#           Functions to extract branching ratio errors of the decay modes on particles
#
#
#################################################################




def get_product_lists_helper(branching_fraction: pdg.decay.PdgBranchingFraction, verbose: bool = False) -> tuple[list[int], list[str]]: 
    product_list = []
    product_list_names = []
    for product in branching_fraction.decay_products:
        #print(product)
        item = product.item
        multi = product.multiplier
        if product.subdecay is None and item.has_particle:
            mcid_prod = item.particle.mcid
            name_prod = item.particle.name  
            for _ in range(multi):
                product_list.append(mcid_prod)
                product_list_names.append(name_prod)
        else:
            # if verbose:
            #     print(f"Decay item has no particle or the product is a subdecay")
            name_prod = item.name       # access name directly from item if no particle available
            for _ in range(multi):
                product_list_names.append(name_prod)

    return product_list, product_list_names




def get_br_text_helper(branching_fraction: pdg.decay.PdgBranchingFraction, verbose: bool = False) -> tuple[float, float]: 
    text = branching_fraction.value_text
    unit = branching_fraction.units

    text_numbers = re.findall(r"\d*\.\d+|\d+", text)
    numbers_converted = [float(num) for num in text_numbers]

    error_pos, error_neg = np.nan, np.nan

    if verbose:
        print(f"Extracted numbers: {text_numbers}, converted: {numbers_converted}")

    if len(text_numbers) == 0:
        if verbose:
            print("No valid numbers found.")
        return np.nan, np.nan
    
    if len(text_numbers) == 2:
        if "to" in text or "TO" in text:
            error_pos = (numbers_converted[1] - numbers_converted[0]) / 2
            error_neg = (numbers_converted[1] - numbers_converted[0]) / 2
        if verbose:
            print(f"text2_error positive: {round(error_pos, 10)}, text2_error negative: {round(error_neg, 10)}")
        return round(error_pos, 10), round(error_neg, 10)

    if len(text_numbers) == 3:
        if "E" in text or "e" in text:
            diff = (numbers_converted[1] - numbers_converted[0]) / 2
            error_pos = diff * 10 ** (-numbers_converted[2])
            error_neg = diff * 10 ** (-numbers_converted[2])
        if "to" in text or "TO" in text:
            error_pos = numbers_converted[2] - numbers_converted[1]
            error_neg = numbers_converted[1] - numbers_converted[0]
        if verbose:
            print(f"text3_error positive: {round(error_pos, 10)}, text3_error negative: {round(error_neg, 10)}")
        return round(error_pos, 10), round(error_neg, 10)

    return np.nan, np.nan




def get_br_errors_helper(identifier, products_ids: list[int], products_names: list[list[str]], api: pdg.api.PdgApi, call_type="id", verbose: bool = False) -> tuple[float, float]:

    try:
        if call_type == "id":
            particle = api.get_particle_by_mcid(identifier)
        elif call_type == "name":
            particle = api.get_particle_by_name(identifier)
        else:
            raise ValueError(f"Unknown call_type: {call_type}")
    except:
        if verbose:
            print(f"Particle not found for identifier: {identifier} with call_type: {call_type}")
            if call_type == "name":
                debug_file("part_found.txt", f"Particle not found for identifier: {identifier} with call_type: {call_type}\n")
        return np.nan, np.nan


    err_pos, err_neg = np.nan, np.nan

    for bf in particle.branching_fractions():

        product_list, product_list_names = get_product_lists_helper(bf, verbose=verbose)

        summary_values = bf.summary_values()
        
        if verbose:
            print('\n%-60s    %s' % (bf.description, bf.value_text))
            # print(bf.units)
            print(f"retrieved product_list: {product_list} or  {product_list_names}")
            print(f"length of summary values: {len(summary_values)}")
            debug_file("products.txt", str(product_list) + "  :  " + str(product_list_names) + "\n")

        for sv in summary_values[::-1]:
            if Counter(products_ids) == Counter(product_list):
                print(f"Match found for {identifier} decay: {products_ids}")
                err_pos = sv.error_positive
                err_neg = sv.error_negative
            elif product_checker(products_names, product_list_names):           
                print(f"Match found for {identifier} decay (names): {products_names}")
                err_pos = sv.error_positive 
                err_neg = sv.error_negative
            else:
                continue

        if err_neg is None and err_pos is None:     # probably because the decay is just "seen" -> should have large errors but still need to check this
            if verbose:
                print(f"No errors could be retrieved, try handling via text. Else nan")
            err_pos, err_neg = get_br_text_helper(bf, verbose=verbose)

    return err_pos, err_neg






def get_br_errors(p_df: pd.DataFrame, d_df: pd.DataFrame, api: pdg.api.PdgApi, verbose: bool = False) -> tuple[list[float], list[float]]:
    br_errors_pos = [np.nan] * len(d_df)
    br_errors_neg = [np.nan] * len(d_df)

    formatted_names_dict = formatted_names(p_df, verbose=verbose)
    formatted_products_dict = formatted_product_names(p_df, verbose=verbose)

    if verbose:
        print("\n\n\n-------------Getting branching ratio errors by id----------------\n\n\n")

    for i, decay in d_df.iterrows():
        if np.isnan(br_errors_pos[i]) or np.isnan(br_errors_neg[i]):
            mcid = int(decay["ParentID"])
            products = decay["ProductIDs"]
            products_ids = [x for x in products if x != 0]             # no dict of ints and strsbecause of possible duplicates which are deleted by dict
            products_names = [formatted_products_dict[x] for x in products_ids]
            if verbose:
                print(f"\nGetting branching ratio errors for MCID {mcid} with products {products_ids} or names {products_names}")
                print(decay)
            error_pos, error_neg = get_br_errors_helper(mcid, products_ids, products_names, api, call_type="id", verbose=verbose)
            if verbose:
                print(f"\nGot branching ratio errors for MCID {mcid}: +{error_pos}, -{error_neg}")
            br_errors_pos[i] = error_pos
            br_errors_neg[i] = error_neg

    if verbose:
        print("\n\n\n-------------Getting branching ratio errors by name----------------\n\n\n")

    for i, decay in d_df.iterrows():
        if np.isnan(br_errors_pos[i]) or np.isnan(br_errors_neg[i]):
            mcid = int(decay["ParentID"])
            name = formatted_names_dict[mcid]
            products = decay["ProductIDs"]
            products_ids = [x for x in products if x != 0]
            products_names = [formatted_products_dict[x] for x in products_ids]
            if verbose:
                print(f"\nGetting branching ratio errors for name {name} with products {products_ids} or names {products_names}")
                print(decay)
            error_pos, error_neg = get_br_errors_helper(name, products_ids, products_names, api, call_type="name", verbose=verbose)
            if verbose:
                print(f"\nGot branching ratio errors for name {name}: +{error_pos}, -{error_neg}")
            br_errors_pos[i] = error_pos
            br_errors_neg[i] = error_neg

    return br_errors_pos, br_errors_neg

















def get_decay_errors(p_df: pd.DataFrame, d_df: pd.DataFrame, api: pdg.api.PdgApi) -> pd.DataFrame:
    """
    Get branching ratio errors for decay modes in the decay DataFrame.
    Do post-processing on the DataFrame to handle particles and antiparticles and handle None and NaN values.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.
        d_df (pd.DataFrame): The input decay DataFrame.
        api (pdg.api.PdgApi): The PDG API instance.

    Returns:
        pd.DataFrame: The decay DataFrame with branching ratio errors added.
    """

    logfolder = Path.cwd() / "logs"          # clear log folder of old files
    files_to_delete = ["part_found.txt", "products.txt"]
    for file in files_to_delete:
        (logfolder / file).unlink()

    br_errors_pos, br_errors_neg = get_br_errors(p_df, d_df, api, verbose=True)

    br_errors_pos = replace_nan_none(br_errors_pos)
    br_errors_neg = replace_nan_none(br_errors_neg)

    d_df_errors = d_df.copy()
    d_df_errors["BR Error Pos"] = br_errors_pos
    d_df_errors["BR Error Neg"] = br_errors_neg

    return d_df_errors









def formatted_names(p_df: pd.DataFrame, verbose: bool = False) -> dict[int, str]:
    """
    Format the names of the particles in the DataFrame.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.

    Returns:
        formatted_names_dict (dict[int, str]): A dictionary mapping particle IDs to formatted names.
    """

    lookup_dict = {
        "pi2(1880)+": "pi_2(1880)0",
        "Omega(2250)": "Omega(2250)-",
        "Lambda(2350)": "Lambda(2350)0",
        "Ksi": "Xi",
        "rho3": "rho_3",
        "rho5": "rho_5",
        "eta2": "eta_2",
        "f0": "f_0",
        "f1": "f_1",
        "f2": "f_2",
        "f4": "f_4",
        "f6": "f_6",
        "a0": "a_0",
        "a1": "a_1",
        "a2": "a_2",
        "a4": "a_4",
        "h1": "h_1",
        "pi1": "pi_1",
        "pi2": "pi_2",
        "K0*": "K_0^*",
        "K0": "K_0",
        "K1": "K_1",
        "K2*": "K_2^*",
        "K2": "K_2",
        "K3": "K_3",
        "K4": "K_4",
        "K5": "K_5"
        # "Anti-Delta": "Deltabar",
        # "Anti-Sigma": "Sigmabar",
        # "Anti-Xi": "Xibar",
        # "Anti-Omega": "Omegabar"
        # "Anti-Proton": "pbar",
        # "Anti-Neutron": "nbar"
        # "Anti-Lambda": "Lambdabar"
        # "Anti-K0": "K0bar",
        # "Anti-K0*": "K0bar*"
        # "Anti-K+": "K-",
        # "Anti-K-": "K+"
    }

    formatted_names = {}
    for _, particle in p_df.iterrows():
        name = particle["Name"]
        id = particle["ID"]
        #print(type(name))  ->  str
        for key, value in lookup_dict.items():
            name = name.replace(key, value)
        formatted_names[id] = name

    if verbose:
        print(f"Formatted particle names: {formatted_names}")
        print(f"Number of formatted names: {len(formatted_names)} \n")

    return formatted_names





def formatted_product_names(p_df: pd.DataFrame, verbose: bool = False) -> dict[int, list[str]]:
    """
    Format the names of the particles in the DataFrame.

    Parameters:
        p_df (pd.DataFrame): The input particle DataFrame.

    Returns:
        formatted_names_dict (dict[int, list[str]]): A dictionary mapping particle IDs to formatted names.
    """

    lookup_dict = {
        "pi2(1880)+": "pi_2(1880)0",
        "Omega(2250)": "Omega(2250)-",
        "Lambda(2350)": "Lambda(2350)0",
        "Ksi": "Xi",
        "rho3": "rho_3",
        "rho5": "rho_5",
        "eta2": "eta_2",
        "f0": "f_0",
        "f1": "f_1",
        "f2": "f_2",
        "f4": "f_4",
        "f6": "f_6",
        "a0": "a_0",
        "a1": "a_1",
        "a2": "a_2",
        "a4": "a_4",
        "h1": "h_1",
        "pi1": "pi_1",
        "pi2": "pi_2",
        "K0": "K_0",
        "K1": "K_1",
        "K2": "K_2",
        "K3": "K_3",
        "K4": "K_4",
        "K5": "K_5"
    }

    def list_appender_helper(lst, item):
        lst.append(item)
        lst.append(item.lower())
        lst.append(item.capitalize())
        if item.endswith("++") or item.endswith("--"):
            lst.append(item[:-2])   # remove charge at the end
        elif item.endswith("0") or item.endswith("+") or item.endswith("-"):
            lst.append(item[:-1])   # remove charge at the end
        return lst

    formatted_names_dict = {}
    for _, particle in p_df.iterrows():
        names_form = []
        name = particle["Name"]
        id = particle["ID"]
        #print(type(name))  ->  str
        for key, value in lookup_dict.items():
            name = name.replace(key, value)

        names_form = list_appender_helper(names_form, name)

        if "Anti-" in name:
            name_anti = name.replace("Anti-", "")
            if name_anti.endswith("+") or name_anti.endswith("++"):
                name_anti = name_anti.replace("+", "-")
            elif name_anti.endswith("-") or name_anti.endswith("--"):
                name_anti = name_anti.replace("-", "+")
            names_form = list_appender_helper(names_form, name_anti)

        if "*" in name:
            name_star = name.replace("*", "^*")
            names_form = list_appender_helper(names_form, name_star)
        
        if "Anti-K" in name:
            name_k = name.replace("Anti-K", "Kbar")
            names_form = list_appender_helper(names_form, name_k)

        if name == "Anti-p":
            name_p = name.replace("Anti-p", "pbar")
            names_form = list_appender_helper(names_form, name_p)

        if name == "n" or name == "p" or name == "Anti-n" or name == "Anti-p":
            names_form.append("N")
            names_form.append("Nbar")


        if verbose:
            print(f"Formatted product names for {id}: {names_form}")
            print(f"Number of formatted names: {len(names_form)} \n")

        formatted_names_dict[id] = names_form


    return formatted_names_dict





def product_checker(products_possibilities: list[list[str]], products_to_check: list[str]) -> bool:
    # lokale Kopie, um nicht das Original zu zerstören
    possibilities = [set(p) for p in products_possibilities]

    for prod in products_to_check:
        # finde Index einer Liste, die das Produkt enthält
        for i, group in enumerate(possibilities):
            if prod in group:
                del possibilities[i]
                break
        else:
            # falls kein break -> Produkt nicht gefunden
            return False

    # true nur, wenn alles gefunden und nichts übrig
    return len(possibilities) == 0

            



def debug_file(name: str, content: str):
    with open("logs/" + name, "a") as f:
        f.write(content)






















































#################################################################
#
#
#           Old and deprecated functions for error extraction
#
#
#################################################################

def get_mass_errors_deprecated(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
    unit = "GeV"
    mass_errors_pos = [np.nan] * len(p_df)
    mass_errors_neg = [np.nan] * len(p_df)

    counter = 0
    for i, mcid in enumerate(p_df["ID"]):
        if np.isnan(mass_errors_pos[i]) or np.isnan(mass_errors_neg[i]):
            try:
                particle = api.get_particle_by_mcid(mcid)
                #print(type(particle))
                counter += 1
                masses = list(particle.masses(require_summary_data=False))
                #print(f"len of masses: {len(masses)}")
                for mass in masses:
                    try:
                        #print(f"len of summary values: {len(mass.summary_values(summary_table_only=False))}")
                        svm = mass.summary_values(summary_table_only=False)[0]
                        mass_errors_pos[i] = svm.get_error_positive(units = unit)
                        mass_errors_neg[i] = svm.get_error_negative(units = unit)
                        #print(f"Mass errors for MCID {mcid}: +{mass_errors_pos[i]}, -{mass_errors_neg[i]}")
                    except:
                        continue
            except:
                continue

    print(f"Processed {counter} particles for mass errors.")


    counter_name = 0
    for i, name in enumerate(p_df["Name"]):
        if np.isnan(mass_errors_pos[i]) or np.isnan(mass_errors_neg[i]):
            try:
                particle = api.get_particle_by_name(name)
                counter_name += 1
                masses = list(particle.masses(require_summary_data=False))
                for mass in masses:
                    try:
                        svm = mass.summary_values(summary_table_only=False)[0]
                        mass_errors_pos[i] = svm.get_error_positive(units = unit)
                        mass_errors_neg[i] = svm.get_error_negative(units = unit)
                        #print(f"Mass errors for name {name}: +{mass_errors_pos[i]}, -{mass_errors_neg[i]}")
                    except:
                        continue
            except:
                continue

    print(f"Processed {counter_name} particles for mass errors by name.")

    return mass_errors_pos, mass_errors_neg


def get_width_errors_deprecated(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
    unit = "GeV"
    width_errors_pos = [np.nan] * len(p_df)
    width_errors_neg = [np.nan] * len(p_df)

    counter = 0
    for i, mcid in enumerate(p_df["ID"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            try:
                particle = api.get_particle_by_mcid(mcid)
                #print(type(particle))
                counter += 1
                widths = list(particle.widths(require_summary_data=False))
                #print(f"len of widths: {len(widths)}")
                for width in widths:
                    try:
                        #print(f"len of summary values: {len(width.summary_values(summary_table_only=False))}")
                        svw = width.summary_values(summary_table_only=False)[0]
                        width_errors_pos[i] = svw.get_error_positive(units = unit)
                        width_errors_neg[i] = svw.get_error_negative(units = unit)
                        #print(f"Width errors for MCID {mcid}: +{width_errors_pos[i]}, -{width_errors_neg[i]}")
                    except:
                        continue
            except:
                continue

    print(f"Processed {counter} particles for mass errors.")


    counter_name = 0
    for i, name in enumerate(p_df["Name"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            try:
                particle = api.get_particle_by_name(name)
                counter_name += 1
                widths = list(particle.widths(require_summary_data=False))
                for width in widths:
                    try:
                        svw = width.summary_values(summary_table_only=False)[0]
                        width_errors_pos[i] = svw.get_error_positive(units = unit)
                        width_errors_neg[i] = svw.get_error_negative(units = unit)
                        #print(f"Width errors for name {name}: +{width_errors_pos[i]}, -{width_errors_neg[i]}")
                    except:
                        continue
            except:
                continue

    print(f"Processed {counter_name} particles for width errors by name.")





    counter = 0
    hbar = 6.582119569*10**(-25)
    for i, mcid in enumerate(p_df["ID"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            try:
                particle = api.get_particle_by_mcid(mcid)
                #print(type(particle))
                counter += 1
                lifetimes = list(particle.lifetimes(require_summary_data=False))
                #print(f"len of lifetimes: {len(lifetimes)}")
                for lifetime in lifetimes:
                    try:
                        #print(f"len of summary values: {len(lifetime.summary_values(summary_table_only=False))}")
                        svl = lifetime.summary_values(summary_table_only=False)[0]
                        tau = svl.get_value(units="s")
                        tau_err_p = svl.get_error_positive(units="s")
                        tau_err_n = svl.get_error_negative(units="s")
                        width_errors_pos[i] = (hbar/(tau**2)) * tau_err_p
                        width_errors_neg[i] = (hbar/(tau**2)) * tau_err_n
                        #print(f"Width errors for MCID {mcid}: +{width_errors_pos[i]}, -{width_errors_neg[i]}")
                    except:
                        continue
            except:
                continue

    print(f"Processed {counter} particles for lifetime errors.")


    counter_name = 0
    for i, name in enumerate(p_df["Name"]):
        if np.isnan(width_errors_pos[i]) or np.isnan(width_errors_neg[i]):
            try:
                particle = api.get_particle_by_name(name)
                #print(type(particle))
                counter_name += 1
                lifetimes = list(particle.lifetimes(require_summary_data=False))
                #print(f"len of lifetimes: {len(lifetimes)}")
                for lifetime in lifetimes:
                    try:
                        #print(f"len of summary values: {len(lifetime.summary_values(summary_table_only=False))}")
                        svl = lifetime.summary_values(summary_table_only=False)[0]
                        tau = svl.get_value(units="s")
                        tau_err_p = svl.get_error_positive(units="s")
                        tau_err_n = svl.get_error_negative(units="s")
                        width_errors_pos[i] = (hbar/(tau**2)) * tau_err_p
                        width_errors_neg[i] = (hbar/(tau**2)) * tau_err_n
                        #print(f"Width errors for MCID {mcid}: +{width_errors_pos[i]}, -{width_errors_neg[i]}")
                    except:
                        continue
            except:
                continue

    print(f"Processed {counter_name} particles for lifetime errors by name.")

    

    return width_errors_pos, width_errors_neg


def get_width_errors_deprecated_old(p_df :pd.DataFrame, api: pdg.api.PdgApi) -> tuple[list[float], list[float]]:
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


































