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
	
    # Continuum dominated sources use wavelength intervales 5-7,
    # 13.8-14.2, and 27-31.5 um
    if spec_type == "C":
        ind_short = ((spec_waves >= 5.0*u.micron) &
                     (spec_waves <= 7.0*u.micron))
        ind_int = ((spec_waves >= 13.8*u.micron) &
                   (spec_waves <= 14.2*u.micron))
        ind_long = ((spec_waves >= 27.0*u.micron) &
                    (spec_waves <= 31.5*u.micron))
        
        # Fit each interval with a power-law which is a line in log-space
        p_short = np.polyfit(np.log10(spec_waves.value[ind_short]),
                             np.log10(spec_flux.value[ind_short]),
                             deg=1)
        p_int = np.polyfit(np.log10(spec_waves.value[ind_int]),
                           np.log10(spec_flux.value[ind_int]),
                           deg=1)
        p_long = np.polyfit(np.log10(spec_waves.value[ind_long]),
                            np.log10(spec_flux.value[ind_long]),
                            deg=1)
        
        pivot_short = spec_waves.value[ind_short]
        pivot_int = spec_waves.value[ind_int]
        pivot_long = spec_waves.value[ind_long]
        pivot_flux_short = 10**(p_short[0]*np.log10(pivot_short) +
                                p_short[1])
        pivot_flux_int = 10**(p_int[0]*np.log10(pivot_int) +
                              p_int[1])
        pivot_flux_long = 10**(p_long[0]*np.log10(pivot_long) +
                               p_long[1])
        
        
        pivots = np.hstack([pivot_short, pivot_int, pivot_long])
        pivot_flux = np.hstack([pivot_flux_short, pivot_flux_int, pivot_flux_long])
        cont_spline = UnivariateSpline(pivots, pivot_flux, s=0)
        continuum = Spectrum(spec_waves,
		                     cont_spline(spec_waves.value)*spec_flux.unit)
		                    
		
    elif spec_type == "P":
        ind_short = ((spec_waves >= 5.3*u.micron) &
                     (spec_waves <= 5.7*u.micron))
        ind_int = ((spec_waves >= 14.0*u.micron) &
                   (spec_waves <= 15.0*u.micron))
        ind_long = ((spec_waves >= 27.0*u.micron) &
                    (spec_waves <= 31.5*u.micron))
		
        flux_short = np.mean(spec_flux.value[ind_short])
        flux_int = np.mean(spec_flux.value[ind_int])
        wave_short = np.mean(spec_waves.value[ind_short])
        wave_int = np.mean(spec_waves.value[ind_int])
        p_short_int = np.polyfit(np.log10([wave_short, wave_int]),
                                 np.log10([flux_short, flux_int]),
                                 deg=1)
        pivot_long = spec_waves.value[ind_long]
        pivot_flux_long = spec_flux.value[ind_long]
        pivots = np.hstack([wave_int, pivot_long])
        pivot_flux = np.hstack([flux_int, pivot_flux_long])

        cont_spline = UnivariateSpline(pivots, pivot_flux, s=0)
        continuum_short = 10**(p_short_int[0]*np.log10(spec_waves.value[spec_waves.value < 15.0]) +
		                       p_short_int[1])
        continuum_long = cont_spline(spec_waves.value[spec_waves.value >= 15.0])
        continuum = Spectrum(spec_waves,
                             np.hstack([continuum_short, continuum_long])*spec_flux.unit)
        
    elif spec_type == "A":
        ind_short = ((spec_waves >= 5.2*u.micron) &
                     (spec_waves <= 5.6*u.micron))
        ind_int = ((spec_waves >= 13.2*u.micron) &
                   (spec_waves <= 14.5*u.micron))
        ind_long = ((spec_waves >= 27.0*u.micron) &
                    (spec_waves <= 31.5*u.micron))

        # Fit each interval with a power-law which is a line in log-space
        p_short = np.polyfit(np.log10(spec_waves.value[ind_short]),
                             np.log10(spec_flux.value[ind_short]),
                             deg=1)
        p_int = np.polyfit(np.log10(spec_waves.value[ind_int]),
                           np.log10(spec_flux.value[ind_int]),
                           deg=1)
        p_long = np.polyfit(np.log10(spec_waves.value[ind_long]),
                            np.log10(spec_flux.value[ind_long]),
                            deg=1)

        pivot_short = spec_waves.value[ind_short]
        pivot_anchor = 7.8
        pivot_int = spec_waves.value[ind_int]
        pivot_long = spec_waves.value[ind_long]
        pivot_flux_short = 10**(p_short[0]*np.log10(pivot_short) +
                                p_short[1])
        pivot_flux_anchor = np.interp(7.8, spec_waves.value,
                                      spec_flux.value)
        pivot_flux_int = 10**(p_int[0]*np.log10(pivot_int) +
                              p_int[1])
        pivot_flux_long = 10**(p_long[0]*np.log10(pivot_long) +
                               p_long[1])
                               
        pivots = np.hstack([pivot_short,
                            pivot_anchor,
                            pivot_int,
                            pivot_long])
        pivot_flux = np.hstack([pivot_flux_short,
                                pivot_flux_anchor,
                                pivot_flux_int,
                                pivot_flux_long])
        
        cont_spline = UnivariateSpline(pivots, pivot_flux, s=0)
        continuum = Spectrum(spec_waves,
                             cont_spline(spec_waves.value)*spec_flux.unit)

    return continuum


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
