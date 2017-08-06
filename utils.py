"""
"""

import numpy as np
from astropy.units.quantity import Quantity
import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Gaussian1D
import emcee
from scipy.stats import norm

class Spectrum(object):
	"""
	An object to store spectral data. Key components are
	the flux and wavelength with an optional error component.
	Flux and wavelength should be Astropy Quantity objects with units attached
	"""

	def __init__(self, waves, flux, error='None', z=0):
		self.z = z
		self.set_wavelength(waves, z)
		self.set_flux(flux)
		if (error == 'None'):
			self._error = np.ones(len(self.flux)) * self.flux.unit
		else:
			self.set_error(error)
		

	def set_wavelength(self, waves, z):
		if isinstance(waves, Quantity):
			self.waves_obs = waves
			self.waves_rest = waves/(1+z)
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
				raise ValueError('Units on error must be same as'
								 'units on flux!')
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
	
	def set_redshift(self, z):
		self.z = z
		self.waves_rest = self.waves_obs/(1+z)


class Drude(Fittable1DModel):
	"""
	Drude profile to use for dust features in IRS spectra.
	"""
	
	amplitude = Parameter(min=0)
	fwhm = Parameter(min=0)
	x_0 = Parameter(min=0)
		
	def evaluate(self, x, amplitude, fwhm, x_0):
		num = amplitude * (fwhm/x_0)**2
		denom = (x/x_0 - x_0/x)**2 + (fwhm/x_0)**2
		return num/denom


	def luminosity(self, ldist):
		"""
		Calculate luminosity of Drude profile for a specific luminosity distance.
		ldist = luminosity distance in Mpc
		"""
		a = np.pi*3e14/2.*self.amplitude*(self.fwhm/self.x_0)/self.x_0/10**23
		b = 4*np.pi*(ldist*10**6*3.086e18)**2
		return a*b


class GaussianLine(Fittable1DModel):
	"""
	A model for line emission using a Gaussian
	"""
	
	amplitude = Parameter(min=0)
	fwhm = Parameter(min=0)
	x_0 = Parameter(min=0)
	
	def evaluate(self, x, amplitude, fwhm, x_0):
		sig = fwhm/2.3548
		return amplitude * np.exp(-(x - x_0)**2/(2*sig**2))
		
	def luminosity(self, ldist):
		lam = np.arange(-1000, 1000, 0.001)
		nu = 3e14/(lam)
		f = self(lam)
		f_int = -np.trapz(f, nu) / 10**23
		b = 4*np.pi*(ldist*10**6*3.086e18)**2
		return f_int*b


class Continuum(Fittable1DModel):
	
	c0 = Parameter()
	c1 = Parameter()
	c2 = Parameter()
	c3 = Parameter()
	#si_tau = Parameter(min=0)
	#si_fwhm = Parameter(min=0)

	def evaluate(self, x, c0, c1, c2, c3):
		f = c0 + c1*x + c2*x**2 + c3*x**3
		#num = si_tau * (si_fwhm/9.7)**2
		#denom = (x/9.7 - 9.7/x)**2 + (si_fwhm/9.7)**2
		#f = f*np.exp(-num/denom)
		#f[f < 0] = 0
		return f

		
class SpectraBayesFitter(object):

	def __init__(self, nwalkers=50, nsteps=1000, nburn=200, threads=4):

		self.set_nwalkers(nwalkers)
		self.set_nsteps(nsteps)
		self.set_nburn(nburn)
		self.set_nthreads(threads)

	def set_nwalkers(self, nw):
		self.nwalkers = nw

	def set_nsteps(self, ns):
		self.nsteps = ns

	def set_nburn(self, nb):
		self.nburn = nb

	def set_nthreads(self, nt):
		self.threads = nt

	def fit(self, model, x, y, yerr=None, best=np.median, errs=(16, 84)):

		mod = model.copy()
		fixed = np.array([mod.fixed[n] for n in mod.param_names])
		self.ndims = np.sum(~fixed)
		
		# Give equal weight to all data points if yerr is None
		if yerr is None:
			yerr = np.ones(len(x))
		elif np.isscalar(yerr):
			yerr = y*yerr

		# Use the current model parameters as the initial values
		init = mod.parameters[~fixed]
		init_walkers = [init + 1e-4*np.random.randn(self.ndims)
						for k in range(self.nwalkers)]

		# Setup the MCMC sampler
		mcmc = emcee.EnsembleSampler(self.nwalkers, self.ndims, log_post,
									 args=(x, y, yerr, mod, fixed),
									 threads=self.threads)

		mcmc.run_mcmc(init_walkers, self.nsteps)

		mod.chain = mcmc.chain[:, :, :].reshape(-1, self.ndims)
		mod.chain_nb = mcmc.chain[:, self.nburn:, :].reshape(-1, self.ndims)

		if self.threads > 1:
			mcmc.pool.close()

		mod.parameters[~fixed] = best(mod.chain_nb, axis=0)
		mod.param_errs = np.zeros((len(mod.parameters), 2))
		mod.param_errs[~fixed] = np.percentile(mod.chain_nb, q=errs, axis=0).T

		return mod
		

def log_like(x, y, yerr, model):
	y_model = model(x)
	llike = -0.5*(np.sum((y-y_model)**2/yerr**2 + np.log(2*np.pi*yerr**2)))

	return llike

	
def log_prior(params, model, fixed):
	pnames = np.array(model.param_names)
	bounds = np.array([model.bounds[n] for n in model.param_names])
	bounds = bounds[~fixed]
	lp = np.array(map(uniform_prior, params, bounds))
	return sum(lp)


def log_post(params, x, y, yerr, model, fixed):

	lprior = log_prior(params, model, fixed)
	dummy = model.copy()
	if not np.isfinite(lprior):
		return -np.inf
	else:
		dummy.parameters[~fixed] = params
		llike = log_like(x, y, yerr, dummy)
		if not np.isfinite(llike):
			return -np.inf
		else:
			return lprior + llike


def uniform_prior(x, bounds):
	if bounds[0] is None:
		bounds[0] = -np.inf
	if bounds[1] is None:
		bounds[1] = np.inf

	if (x >= bounds[0]) & (x <= bounds[1]):
		if np.isinf(bounds[0]) | np.isinf(bounds[1]):
			return 0.0
		else:
			return np.log(1.0/(bounds[1] - bounds[0]))
	else:
		return -np.inf

# Negative log-likelihood for a truncated normal distribution with a = 0 and b = inf.
def lnlike_truncnorm(params, x):
    return -np.sum(np.log(norm.pdf(x, loc=params[0], scale=params[1])) - np.log(1.0 - norm.cdf(0, loc=params[0], scale=params[1])))
