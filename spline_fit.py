"""
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from utils import Spectrum
import astropy.constants as c
import astropy.units as u


def spline_fit(spec, spec_type):
    """
    Function to fit the continuum with a spline using the user-defined
    pivots.

    Inputs
    ------
    spec = Spectrum object that contains the flux and wavelength of the
           spectrum to be fit
    spec_type = Spectral type. Either "P" (PAH dominated), "A" (absorption
                dominated), "C" (continuum dominated). The continuum estimation
                method depends on the spectral type.

    Outputs
    -------
    cont_obj= Dictionary containing:
              cont = Spectrum object of the estimated continuum at each of the
                     input spectral wavelengths
              pivots = The pivots used in the spline interpolation
              pivot_flux = The pivot fluxes for the spline interpolation
              cont_spline = The output spline object

    """

    # Check to make sure spec is a Spectrum
    if not isinstance(spec, Spectrum):
        raise TypeError('"spec" must be a Spectrum object!')

    # Pull out the wavelength and flux of the spectrum
    spec_waves = spec.waves
    spec_flux = spec.flux
    spec_error = spec.error

    if spec_type == "C":
        ind_short = ((spec_waves >= 5.0*u.micron) &
                     (spec_waves <= 7.2*u.micron))
        ind_long = ((spec_waves >= 27.0*u.micron) &
                    (spec_waves <= 31.5*u.micron))

#        short_fit = np.polyfit(np.log10(spec_waves.value[ind_short]),
#                               np.log10(spec_flux.value[ind_short]), deg=1)
#        long_fit = np.polyfit(np.log10(spec_waves.value[ind_long]),
#                              np.log10(spec_flux.value[ind_long]), deg=1)
#        pivot_short_flux = 10**(short_fit[0]*np.log10(7.2) + short_fit[1])
#        pivot_long_flux = 10**(long_fit[0]*np.log10(27.0) + long_fit[1])
#        pivot_middle_flux = np.interp(13.5, spec_waves.value,
#                                      spec_flux.value)
#        pivots = np.array([7.2, 13.5, 27.0])
#        pivot_flux = np.array([pivot_short_flux, pivot_middle_flux,
#                               pivot_long_flux])
        pivots = spec_waves[ind_short | ind_long].value
        pivot_flux = spec_flux[ind_short | ind_long].value
        pivot_weight = spec_error[ind_short | ind_long].value
        cont_spline = UnivariateSpline(pivots, pivot_flux, w=pivot_weight)

    elif spec_type == "P":
        pivots = np.array([5.5, 14.5, 27.0, 31.5])
        pivot_flux = np.array([np.interp(p, spec_waves.value, spec_flux.value)
                               for p in pivots])
        cont_spline = UnivariateSpline(pivots, pivot_flux, s=0)

    elif spec_type == "A":
        pivots = np.array([5.2, 5.6, 14.0, 27.0, 31.5])
        pivot_flux = np.array([np.interp(p, spec_waves.value, spec_flux.value)
                               for p in pivots])
        cont_spline = UnivariateSpline(pivots, pivot_flux, s=0)

    cont_waves = spec.waves
#    if spec_type == "C":
#        cont_flux = np.zeros(len(cont_waves))
#        short = cont_waves < 7.2*u.micron
#        cont_flux[short] = 10**(short_fit[0]*np.log10(cont_waves.value[short])
#                                + short_fit[1])
#        mid = (cont_waves >= 7.2*u.micron) & (cont_waves <= 27.0*u.micron)
#        cont_flux[mid] = cont_spline(cont_waves.value[mid])
#        llong = cont_waves > 27.0*u.micron
#        cont_flux[llong] = 10**(long_fit[0]*np.log10(cont_waves.value[llong])
#                                + long_fit[1])
#
#    else:
    cont_flux = cont_spline(cont_waves.value)

    continuum = Spectrum(cont_waves, cont_flux*spec_flux.unit)

    cont_object = {'cont': continuum, 'pivots': pivots,
                   'pivot_flux': pivot_flux, 'cont_spline': cont_spline}

    return cont_object


def calc_eqw(spec, cont, wi, wf):
    """
    Calculates the equivalent width of a spectral element (absorption or
    emission)
    between two wavelengths, wi and wf.
    Use the equation: W = Integral(1 - F_l/F_c)dl
    F_l is observed spectrum
    F_c is continuum
    The integral is performed using "trapz" function of numpy.

    Inputs
    ------
    spec = observed spectrum as a Spectrum object
    cont = estimated continuum as a UnivariateSpline object
    wi   = lower wavelength
    wf   = upper wavelength (wi < wf)

    Outputs
    -------
    eqw = the equivalent width of the spectral element between wi and wf

    """

    # Force wi < wf
    if (wi >= wf):
        raise ValueError('wi must be less than wf!')

    # Check to make sure spec is a Spectrum
    if not isinstance(spec, Spectrum):
        raise TypeError('"spec" must be a Spectrum object!')

    # Check to make sure spec is a Spectrum
    if not isinstance(cont, UnivariateSpline):
        raise TypeError('"cont" must be a UnivariateSpline object!')

    # Pull out the wavelengths and flux of the spectrum
    spec_waves = spec.waves
    spec_flux = spec.flux

    # Slice the spectrum between wi and wf
    ind_slice = (spec.waves.value >= wi) & (spec.waves.value <= wf)
    spec_waves_slice = spec_waves[ind_slice]
    spec_flux_slice = spec_flux[ind_slice]

    # Calculate the continuum at the same wavelengths
    cont_slice = cont(spec_waves_slice.value)

    # Integrate (1-F_l/F_c) between wi and wf
    eqw = np.trapz((1-spec_flux_slice.value/cont_slice),
                   spec_waves_slice.value)

    # Return the EQW
    return eqw*spec.waves.unit


def calc_flux(spec, cont, wi, wf):
    """
    Function to calculate the continuum-subtracted flux between two wavelengths

    Inputs
    -----
    spec = observed spectrum as a Spectrum object
    cont = estimated continuum as a UnivariateSpline object
    wi   = lower wavelength
    wf   = upper wavelength (wi < wf)

    Outputs
    -------
    int_flux = Integrated continuum subtracted flux between wi and wf. Astropy
               Quantity object with units

    """

    # Force wi < wf
    if (wi >= wf):
        raise ValueError('wi must be less than wf!')

    # Pull out the wavelengths and flux of the spectrum
    spec_waves = spec.waves.cgs
    spec_flux = spec.flux.cgs

    # Convert wavelengths to frequencies
    spec_freq = c.c.cgs/spec_waves

    ff = c.c.cgs/wi.cgs
    fi = c.c.cgs/wf.cgs

    # Subtract the continuum
    spec_cont_sub_flux = spec_flux.value - cont(spec_waves.value)

    # Pull out frequencies and flux
    ind_slice = (spec_freq >= fi) & (spec_freq <= ff)
    f_slice = spec_freq[ind_slice].value[::-1]
    flux_slice = spec_cont_sub_flux[ind_slice][::-1]

    # Integrate
    int_flux = np.trapz(flux_slice, f_slice)*spec_flux.unit*spec_freq.unit

    # Convert to ergs/s/cm^2
    int_flux = int_flux.to(u.erg/u.s/u.cm**2)
    return int_flux


def calc_strength(spec, cont, w):
    """
    Calculate the strength of a spectral feature using the formula from
    Spoon et al 2007.
    S = ln( F_obs(w) / F_cont(w))

    Inputs
    ------
    spec = observed spectrum as a Spectrum object
    cont = estimated continuum as a UnivariateSpline object
    w = wavelength at which to calculate the strength

    Output
    ------
    strength = scalar, unitless strength of the feature

    """

    fcont = cont(w)
    fobs = np.interp(w, spec.waves.value, spec.flux.value)

    return np.log(fobs/fcont)
