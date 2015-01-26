"""
"""

import numpy as np
import astropy.constants as c
from astropy.units.quantity import Quantity
import astropy.units as u

class Spectrum(object):
	"""
	An object to store spectral data. Key components are
	the flux and wavelength with an optional error component. Flux and wavelength
	should be Astropy Quantity objects with units attached
	"""
	
	def __init__(self, waves, flux, error='None'):
		self.set_wavelength(waves)
		self.set_flux(flux)
		if (error == 'None'):
			self._error = np.ones(len(self.flux)) * self.flux.unit
		else:
			self.set_error(error)
				
	def set_wavelength(self, waves):
		if isinstance(waves, Quantity):
			self.waves = waves
		else:
			raise TypeError('Wavelengths must have units attached to it!')		
	
	def set_flux(self, flux):
		if isinstance(flux, Quantity):
			self.flux = flux
		else:
			raise TypeError('Fluxes must have units attached to it!')
	
	def set_error(self, error):
		if isinstance(error, Quantity):
			if (error.unit == self.flux.unit):
				self.error = error
			else:
				raise ValueError('Units on error must be same as units on flux!')
		else:
			raise TypeError('Errors must have units attached to it!')
	
	def waves_to(self, unit):
		if isinstance(unit, u.core.Unit):
			self.waves = self.waves.to(unit)
		else:
			raise TypeError('"unit" must be an Astropy Unit instance')
	
	def flux_to(self, unit):
		if isinstance(unit, u.core.Unit):
			self.flux = self.flux.to(unit)
			self.error = self.error.to(unit)
		else:
			raise TypeError('"unit" must be an Astropy Unit instance')		
