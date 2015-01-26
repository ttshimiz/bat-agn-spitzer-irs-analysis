"""
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from utils import Spectrum

def spline_fit(spec, pivots):
	"""
	Function to fit the continuum with a spline using the user-defined
	pivots.
	
	Inputs
	------
	spec = Spectrum object that contains the flux and wavelength of the spectrum
	       to be fit
	pivots = Array containing the wavelengths of the spectrum to use to fit the
	         spline
	         
	Outputs
	-------
	cont_spline = UnivariateSpline object defining the fitted continuum flux
	pivot_flux  = Array of fluxes used to fit the spline with same size as pivots
	
	"""
	
	#Check to make sure spec is a Spectrum
	if not isinstance(spec, Spectrum):
		raise TypeError('"spec" must be a Spectrum object!')
		
	#Pull out the wavelength and flux of the spectrum
	spec_waves = spec.waves
	spec_flux = spec.flux
	
	#Interpolate the spectrum to get the flux at the pivot points
	pivot_flux = np.array([np.interp(p, spec_waves.value, spec_flux.value) 
	                       for p in pivots])
	                       
	#Construct the spline
	cont_spline = UnivariateSpline(pivots, pivot_flux)
	
	return cont_spline, pivot_flux
	
def calc_eqw(spec, cont, wi, wf):
	"""
	Calculates the equivalent width of a spectral element (absorption or emission)
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
	
	#Force wi < wf
	if (wi >= wf):
		raise ValueError('wi must be less than wf!')
		
	#Check to make sure spec is a Spectrum
	if not isinstance(spec, Spectrum):
		raise TypeError('"spec" must be a Spectrum object!')
	
	#Check to make sure spec is a Spectrum
	if not isinstance(cont, UnivariateSpline):
		raise TypeError('"cont" must be a UnivariateSpline object!')
	 
	#Pull out the wavelengths and flux of the spectrum
	spec_waves = spec.waves
	spec_flux = spec.flux
	
	#Slice the spectrum between wi and wf
	ind_slice = (spec.waves.value >= wi) & (spec.waves.value <= wf)
	spec_waves_slice = spec_waves[ind_slice]
	spec_flux_slice = spec_flux[ind_slice]
	
	#Calculate the continuum at the same wavelengths
	cont_slice = cont(spec_waves_slice.value)
	
	#Integrate (1-F_l/F_c) between wi and wf 
	eqw = np.trapz((1-spec_flux_slice.value/cont_slice), spec_waves_slice.value)
	
	#Return the EQW
	return eqw*spec.waves.unit
	
	
	
	