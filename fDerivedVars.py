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


def calculate_yo_num(inflections):
    """Assigns yo number based on total number of inflections"""
    yo_nums = []
    current_yo = 0
    for i in inflections:
        if not np.isnan(i):
            current_yo = i
        yo_nums.append(current_yo)
    return yo_nums


def calculate_dist_traveled(pitches, depths):
    """Calculates horizontal distance traveled (starting from 0), given pitch and depth values as Series objects."""
    x = []
    for pitch1, pitch2, depth1, depth2 in zip(pitches,
                                              pitches.shift(1, fill_value=pitches.iloc[0]),
                                              depths,
                                              depths.shift(1, fill_value=depths.iloc[0])):
        dz = depth2 - depth1
        avg_pitch = (pitch1 + pitch2) / 2
        dx = dz / np.tan(avg_pitch)
        if x:
            x.append(x[-1] + dx)
        else:
            x.append(dx)
    return x


def calculate_depth(p, lat):
    """Calculates depth from pressure and latitude.

    :param p: pressure (bar)
    :param lat: latitude (decimal degrees)
    :return: depth (m)
    """
    p_dbar = p * 10
    return abs(gsw.z_from_p(p_dbar, lat))


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
    p_dbar = p * 10

    # conductivity in mS/cm = conductivity * 10
    C_mScm = C * 10

    print(T.count())
    print(p.count())
    print(C.count())
    print(lat.count())
    print(long.count())
    # calculate SP from conductivity (mS/cm), in-situ temperature (deg C), and gauge pressure (dbar)
    SP = gsw.SP_from_C(C_mScm, T, p_dbar)
    # calculate SA from SP (unitless), gauge pressure (dbar), longitude and latitude (decimal degrees)
    SA = gsw.SA_from_SP(SP, p_dbar, long, lat)
    # calculate CT from SA (g/kg), in-situ temperature (deg C), and gauge pressure (dbar)
    CT = gsw.CT_from_t(SA, T, p_dbar)
    # calculate N^2
    nsquared, p_mid = gsw.Nsquared(SA, CT, p_dbar, lat=lat)
    return nsquared
