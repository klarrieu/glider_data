import numpy as np
import gsw


def convert_latlong(val):
    """Converts lat/long val from DDMM.MMM format to decimal degrees"""
    if np.isnan(val):
        return np.nan
    else:
        degs = int(val/100)
        mins = val - degs*100
        return degs + mins/60

def calculate_density(T, p, C, lat, long):
    """Calculates density from temp, pressure, conductivity, and lat/long.
    All parameters and output are float or array-like.
    :param T: temperature (deg C)
    :param p: pressure (bar)
    :param C: conductivity (S/m)
    :param lat: latitude (decimal deg)
    :param long: longitude (decimal deg)
    :return: density (kg/m^3)
    """
    # pressure in dbars = pressure * 10
    if type(p) == float:
        p_dbar = p * 10
    else:
        p_dbar = [pi * 10 for pi in p]

    # conductivity in mS/cm = conductivity * 10
    if type(C) == float:
        C_mScm = C * 10
    else:
        C_mScm = [Ci * 10 for Ci in C]
    # calculate SP from conductivity (mS/cm), in-situ temperature (deg C), and gauge pressure (dbar)
    SP = gsw.SP_from_C(C_mScm, T, p_dbar)
    # calculate SA from SP (unitless), gauge pressure (dbar), longitude and latitude (decimal degrees)
    SA = gsw.SA_from_SP(SP, p_dbar, long, lat)
    # calculate CT from SA (g/kg), in-situ temperature (deg C), and gauge pressure (dbar)
    CT = gsw.CT_from_t(SA, T, p_dbar)
    # calculate density
    return gsw.rho(SA, CT, p_dbar)


def calculate_n2(T, p, C, lat, long):
    """Calculates the buoyancy frequency N^2
    All parameters and output are float or array-like.
    :param T: temperature (deg C)
    :param p: pressure (bar)
    :param C: conductivity (S/m)
    :param lat: latitude (decimal deg)
    :param long: longitude (decimal deg)
    :return: buoyancy (i.e. brunt-vaisala) frequency N^2 (s^-2)
    """
    # pressure in dbars = pressure * 10
    if type(p) == float:
        p_dbar = p * 10
    else:
        p_dbar = [pi * 10 for pi in p]

    # conductivity in mS/cm = conductivity * 10
    if type(C) == float:
        C_mScm = C * 10
    else:
        C_mScm = [Ci * 10 for Ci in C]
    # calculate SP from conductivity (mS/cm), in-situ temperature (deg C), and gauge pressure (dbar)
    SP = gsw.SP_from_C(C_mScm, T, p_dbar)
    # calculate SA from SP (unitless), gauge pressure (dbar), longitude and latitude (decimal degrees)
    SA = gsw.SA_from_SP(SP, p_dbar, long, lat)
    # calculate CT from SA (g/kg), in-situ temperature (deg C), and gauge pressure (dbar)
    CT = gsw.CT_from_t(SA, T, p_dbar)
    # calculate N^2
    return gsw.Nsquared(SA, CT, p_dbar, lat=None)
