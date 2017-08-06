# My package of tools to fit the line and dust features in the Spitzer/IRS
# low resolution spectra of the BAT AGN.

# Standard scientific and astronomy modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import astropy.units as u
from scipy.optimize import minimize

# Modeling modules and classes
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy.modeling.polynomial import Polynomial1D
from utils import Spectrum, Drude, GaussianLine, Continuum, SpectraBayesFitter, lnlike_truncnorm
from astropy.stats import sigma_clip

def create_bat_spec(name):
	"""
	Function that uploads a specific BAT AGN IRS spectrum.
	"""
	bat_info = pd.read_csv('/ricci9nb/tshimizu/Github/bat-data/bat_info.csv', index_col=0)
	spec_dir = '/home/tshimizu/Thesis/IRS_stitched_spectra/'
	
	data = np.loadtxt(spec_dir+name+'_SpitzerIRS_stiched.txt')
	
	waves = data[:,0]*u.micron
	flux = data[:, 1]*u.Jy
	err = data[:, 2]*u.Jy
	redshift = bat_info.loc[name, 'Redshift']
	
	spec = Spectrum(waves, flux, error=err, z=redshift)
	
	return spec
	

def plot_fit(spectrum, model, lmin, lmax, plot_spread=True):
	"""
	Function that will plot the best fit model on top of the spectrum
	"""
	
	sn.set(context='notebook', color_codes=True, style='ticks')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlim([lmin, lmax])
	w = spectrum.waves_rest.value
	f = spectrum.flux.value
	e = spectrum.error.value
	ind = (w >= lmin) & (w <= lmax)
	ax.plot(w[ind], f[ind], 'b-', label='Observed')
	ax.fill_between(w[ind], f[ind]-e[ind], f[ind]+e[ind],
	                color='b', alpha=0.3)
	lam = np.arange(lmin, lmax, 0.001)
	# Calculate the 95% confidence interval and median for the model spectrum
	if plot_spread:
		fitter = LevMarLSQFitter()
		model_spec = np.zeros((1000, len(lam)))
		model_cont = np.zeros((1000, len(lam)))
		dummy = model.copy()
		dummy_cont = Polynomial1D(degree=3)
		fixed = np.array([dummy.fixed[n] for n in dummy.param_names])
		ri = np.random.randint(low=0, high=np.shape(dummy.chain_nb)[0], size=1000)
		for i in range(1000):
			dummy.parameters[~fixed] = dummy.chain_nb[ri[i], :]
			dummy_cont.parameters = dummy.chain_nb[ri[i], 0:4]
			model_cont[i,:] = dummy_cont(lam)
			model_spec[i,:] = dummy(lam)
		model_err_up = np.percentile(model_spec, 97.5, axis=0)
		model_err_down = np.percentile(model_spec, 2.5, axis=0)
		model_median = np.percentile(model_spec, 50.0, axis=0)
		model_cont_median = np.percentile(model_cont, 50.0, axis=0)
		model['Continuum'].parameters = fitter(dummy_cont, lam, model_cont_median).parameters
		ax.plot(lam, model_median, 'r--', label='Total Model')
		ax.fill_between(lam, model_err_down, model_err_up, color='r', alpha=0.3)
	else:
		ax.plot(lam, model(lam), 'r--', label='Total Model')
	
	# Plot each component of the model
	for comp in model.submodel_names:
		if comp == 'Continuum':
			ax.plot(lam, model[comp](lam), ls='dotted', label=comp)
		else:
			ax.plot(lam, model[comp](lam) + model['Continuum'](lam), ls='dotted', label=comp)

	ax.legend(loc='upper left')
	ax.set_xlabel('Wavelength [micron]')
	ax.set_ylabel('Flux [Jy]')
	sn.despine()
	
	return fig


def plot_triangle(model, quantiles=[0.16, 0.5, 0.84]):

    import corner
    sn.set_style('ticks')
    fixed  = np.array([model.fixed[n] for n in model.param_names])
    labels = np.array(model.param_names)[~fixed][4:]

    fig = corner.corner(model.chain_nb[:, 4:], quantiles=quantiles,
                        labels=labels, show_titles=True, title_fmt='.2g', verbose=False)
    nfeatures = len(model.submodel_names) - 1
    for i in range(nfeatures):
    	plt.text(0.95, 0.95-i*0.025, 'Feature '+str(i+1)+': '+model.submodel_names[i+1],
    	         transform=fig.transFigure, va='top', ha='right')
    sn.despine()
    return fig

	
def tie_amps(model):
	"""
	Function that ties the 11.23 amplitude to the 11.33 amplitude for
	the PAH 11.3 micron complex
	"""
	return 1.25*model['Drude11_33'].amplitude

def fit_pah11_3(spectrum=None, lmin=10.0, lmax=12.5, poly_deg=3,
                include_siv=True, include_h2s2=True,
                include_pah10_68=True, include_pah12=True, model_only=False):
	"""
	Fit the 11.3 micron PAH complex with the combined model of two Drude profiles
	and a cubic polynomial for the continuum
	"""
	
	# Cubic polynomial to model the continuum
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')
	
	# Two Drude profiles to model the 11.3 micron PAH complex
	pah11_23 = Drude(x_0=11.23, fwhm=0.135, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 11.23')
	pah11_33 = Drude(x_0=11.33, fwhm=0.363, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 11.33')
	
	# Gaussian lines to model the [SIV] and H2 S(2) emission
	siv = GaussianLine(x_0=10.511, fwhm=0.1, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[SIV]')
	siv.amplitude.min = 0
	h2s2 = GaussianLine(x_0=12.278, fwhm=0.1, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='H2 S(2)')
	h2s2.amplitude.min = 0
	pah12 = Drude(x_0=11.99, fwhm=0.540, amplitude=0.1,
	              fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 12')
	pah10_68 = Drude(x_0=10.68, fwhm=0.214, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 10.68')

	# Full model
	model = cont_mod + pah11_23 + pah11_33
	if include_siv:
		model += siv
	if include_h2s2:
		model += h2s2
	if include_pah12:
		model += pah12
	if include_pah10_68:
		model += pah10_68

	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=5000, nburn=1000)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
	else:
		result = model

	return result
	

def fit_pah7_7(spectrum=None, lmin=6.5, lmax=9.5, poly_deg=3,
               include_arII=True, include_h2s5=True,
               include_h2s4=True, include_pah8_6=True,
               include_neVI=True, include_pah8_3=True, 
               model_only=False):
	"""
	Fit the 7.7 micron PAH complex with the combined model of three Drude profiles
	and a cubic polynomial for the continuum
	"""
	
	# Cubic polynomial to model the continuum
	#cont_mod = Continuum(c0=0.01, c1=0.01, c2=0.01, c3=0.01, name='Continuum')
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')

	# Two Drude profiles to model the 11.3 micron PAH complex
	pah7_42 = Drude(x_0=7.42, fwhm=0.935, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 7.42')
	pah7_60 = Drude(x_0=7.60, fwhm=0.334, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 7.60')
	pah7_85 = Drude(x_0=7.85, fwhm=0.416, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 7.85')
	
	# Gaussian lines to model the [SIV] and H2 S(2) emission
	arII = GaussianLine(x_0=6.985, fwhm=0.053, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[ArII]')
	h2s5 = GaussianLine(x_0=6.909, fwhm=0.053, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='H2 S(5)')
	h2s4 = GaussianLine(x_0=8.026, fwhm=0.1, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='H2 S(4)')
	neVI = GaussianLine(x_0=7.652, fwhm=0.053, amplitude=0.3,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[NeVI]')
	pah8_6 = Drude(x_0=8.61, fwhm=0.336, amplitude=0.1,
	              fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 8.60')
	pah8_3 = Drude(x_0=8.33, fwhm=0.417, amplitude=0.1,
	              fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 8.33')
	#pah6_22 = Drude(x_0=6.22, fwhm=0.187, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 6.22')
	# Full model
	model = cont_mod + pah7_42 + pah7_60 + pah7_85
 	if include_arII:
 		model += arII
 	if include_h2s5:
 		model += h2s5
	if include_h2s4:
 		model += h2s4
 	if include_pah8_6:
 		model += pah8_6
	if include_neVI:
		model += neVI
	if include_pah8_3:
		model += pah8_3
	#if include_pah6_22:
	#	model += pah6_22

	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=500, nburn=100)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
		#result = init_mod
	else:
		result = model

	return result
	


def fit_pah6_2(spectrum=None, lmin=5.2, lmax=6.7, poly_deg=3,
               include_pah5_27=True, include_pah5_70=True, include_pah6_70=True,
               include_h2s6=True, include_h2s7=True, include_feII=True, model_only=False):
	"""
	Fit the 6.2 micron PAH complex with the combined model of three Drude profiles
	and a cubic polynomial for the continuum
	"""
	
	# Cubic polynomial to model the continuum
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')
	
	# One Drude profile to model the 6.22 micron PAH feature
	pah6_22 = Drude(x_0=6.22, fwhm=0.187, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 6.22')
	
	# Gaussian lines to model the [FeII] and H2 S(5) and S(6) emission
	# Drude profiles to model the 5.27, 5.70, and 6.7 PAH features
	feII = GaussianLine(x_0=5.34, fwhm=0.053, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[FeII]')
	feII.amplitude.min = 0

	h2s7 = GaussianLine(x_0=5.511, fwhm=0.053, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='H2 S(7)')
	h2s7.amplitude.min = 0
	h2s6 = GaussianLine(x_0=6.109, fwhm=0.053, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='H2 S(6)')
	h2s6.amplitude.min = 0
	pah5_27 = Drude(x_0=5.27, fwhm=0.179, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 5.27')
	pah5_70 = Drude(x_0=5.70, fwhm=0.416, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 5.70')
	pah6_70 = Drude(x_0=6.69, fwhm=0.468, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 6.70')


	# Full model
	model = cont_mod + pah6_22
	if include_h2s6:
 		model += h2s6
 	if include_h2s7:
 		model += h2s7
 	if include_pah5_27:
 		model += pah5_27
 	if include_pah5_70:
 		model += pah5_70
	if include_pah6_70:
 		model += pah6_70
	if include_feII:
		model += feII
	
	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=5000, nburn=1000)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
	else:
		result = model

	return result

	
def fit_neII(spectrum=None, lmin=12.0, lmax=14.0, poly_deg=3,
             include_h2s2=True, include_pah12_7=True,
             include_pah13_5=True, model_only=False):
	"""
	Function to fit the [NeII] line.
	"""
	
	# Cubic polynomial to model the continuum
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')
	
	# Use a Gaussian line to model the [NeII] emission
	neII = GaussianLine(x_0=12.813, fwhm=0.1, amplitude=0.1,
	                    fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[NeII]')
	h2s2 = GaussianLine(x_0=12.278, fwhm=0.1, amplitude=0.1,
	                    fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='H2 S(2)')
	pah12_7 = (Drude(x_0=12.62, fwhm=0.530, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 12.62') + 
	          Drude(x_0=12.69, fwhm=0.165, amplitude=0.01, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 12.69')) 
	pah13_5 = Drude(x_0=13.48, fwhm=0.539, amplitude=0.1, fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 13.5')
	
	model = cont_mod + neII
	
	if include_h2s2:
		model += h2s2
		
	if include_pah12_7:
		model += pah12_7
		
	if include_pah13_5:
		model += pah13_5
	
	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=5000, nburn=1000)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
	else:
		result = model

	return result

def fit_neIII(spectrum=None, lmin=14.5, lmax=16.5, poly_deg=3,
              include_pah15_90=True, include_pah16_45=True, model_only=False):
	"""
	Function to fit the [NeIII] line.
	"""
	
	# Cubic polynomial to model the continuum
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')
	
	# Use a Gaussian line to model the [NeII] emission
	neIII = GaussianLine(x_0=15.555, fwhm=0.14, amplitude=0.1,
	                     fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[NeIII]')
	pah15_90 = Drude(x_0=15.90, fwhm = 0.318, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 15.90')
	pah16_45 = Drude(x_0=16.45, fwhm = 0.230, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 16.45')
	
	model = cont_mod + neIII
	if include_pah15_90:
		model += pah15_90
	if include_pah16_45:
		model += pah16_45

	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=5000, nburn=1000)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
	else:
		result = model

	return result

def fit_neV(spectrum=None, lmin=13.3, lmax=15.3, poly_deg=3,
            include_ciii=True, include_pah13_48=True,
            include_pah14_04=True, include_pah14_19=True, model_only=False):
	"""
	Function to fit the [NeV] line.
	"""
	
	# Cubic polynomial to model the continuum
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')
	
	# Use a Gaussian line to model the [NeV] emission
	neV = GaussianLine(x_0=14.322, fwhm=0.14, amplitude=0.1,
	                   fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[NeV]')
	cIII = GaussianLine(x_0=14.368, fwhm=0.14, amplitude=0.1,
	                    fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[CIII]')
	pah13_48 = Drude(x_0=13.48, fwhm = 0.539, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 13.48')
	pah14_04 = Drude(x_0=14.04, fwhm = 0.225, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 14.04')
	pah14_19 = Drude(x_0=14.19, fwhm = 0.355, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='PAH 14.19')
	model = cont_mod + neV
	
	if include_ciii:
		model += cIII
	if include_pah13_48:
		model += pah13_48
	if include_pah14_04:
		model += pah14_04
	if include_pah14_19:
		model += pah14_19
	
	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=5000, nburn=1000)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
	else:
		result = model

	return result
	
def fit_oiv(spectrum=None, lmin=24.5, lmax=27.5, poly_deg=3, include_feii=True, model_only=False):
	"""
	Function to fit the [OIV] line.
	"""
	
	# Cubic polynomial to model the continuum
	cont_mod = Polynomial1D(degree=poly_deg, name='Continuum')
	
	# Use a Gaussian line to model the [NeII] emission
	oiv = GaussianLine(x_0=25.910, fwhm=0.34, amplitude=0.1,
	                 fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[OIV]')
	feii = GaussianLine(x_0=25.989, fwhm=0.34, amplitude=0.1,
	                  fixed={'x_0':True, 'fwhm':True, 'amplitude':False}, name='[FeII]')
	
	model = cont_mod + oiv
	
	if include_feii:
		model += feii
	
	if not model_only:
		lmFitter = LevMarLSQFitter()
		bayesFitter = SpectraBayesFitter(threads=1, nsteps=5000, nburn=1000)
	 
		lam = spectrum.waves_rest.value
		flux = spectrum.flux.value
		err = spectrum.error.value
		ind = (lam >= lmin) & (lam <= lmax)
		x = lam[ind]
		y = flux[ind]
		e = err[ind]
		w = 1/e
	
		init_mod = lmFitter(model, x, y, maxiter=2000, weights=w)
		result = bayesFitter.fit(init_mod, x, y, yerr=e)
	else:
		result = model

	return result
	

def calc_eqw(model, feature_name):

	cont_mod = model['Continuum']
	feature_mod = model[feature_name]
	
	lmin = feature_mod.x_0 - 6*feature_mod.fwhm
	lmax = feature_mod.x_0 + 6*feature_mod.fwhm	
	lam = np.arange(lmin, lmax, 0.001)
	
	eqw = np.trapz(feature_mod(lam)/cont_mod(lam), lam)
	
	return eqw

# Function that reads the MCMC chain for the fit of each feature and calculates
# the amplitude, luminosity, and equivalent width by randomly pulling out 1000 samples from the MCMC chain.
def get_feature_strengths(model, ldist, nsamp=1000, nburn=1000, nsteps=5000, nwalkers=50):

	# Remove the burn-in steps for each walker
	dummy = model.copy()
	fixed = np.array([dummy.fixed[n] for n in dummy.param_names])
	chain = model.chain
	chain_length = nwalkers*nsteps
	ndims = np.shape(chain)[1]
	chain_burned = np.zeros((nwalkers*(nsteps-nburn), ndims))
	for i in range(nwalkers):
		chain_burned[i*(nsteps-nburn):(i+1)*(nsteps-nburn), :] = chain[i*nsteps+nburn:(i+1)*nsteps]

	# Get random indices to use to sample the posterior distributions
	ri = np.random.choice(nwalkers*(nsteps-nburn), 1000, replace=False)

	features = np.array(model.submodel_names)[np.array(model.submodel_names) != 'Continuum']
	feature_amp = {f:np.zeros(nsamp) for f in features}
	feature_lum = {f:np.zeros(nsamp) for f in features}
	feature_eqw = {f:np.zeros(nsamp) for f in features}
	
	for j in range(nsamp):
		
		dummy.parameters[~fixed] = chain_burned[ri[j], :]

		for f in features:
			
			feature_amp[f][j] = dummy[f].amplitude.value
			feature_lum[f][j] = dummy[f].luminosity(ldist)/10**40.
			feature_eqw[f][j] = calc_eqw(dummy, f)
	

	results = {}
	for f in features:
		# Fit the random sample of amplitudes, luminosities, and equivalent widths
		# with a truncated normal distribution

		# Use sigma-clipping to get rid of outliers in the distributions
		fa = sigma_clip(feature_amp[f], sig=5.0, iters=None)
		fl = sigma_clip(feature_lum[f], sig=5.0, iters=None)
		fe = sigma_clip(feature_eqw[f], sig=5.0, iters=None)
		fa = fa[~fa.mask]
		fe = fe[~fe.mask]
		fl = fl[~fl.mask]

		pamp = minimize(lnlike_truncnorm, [np.mean(fa), np.std(fa)],
                                method='nelder-mead', args=(fa))
		plum = minimize(lnlike_truncnorm, [np.mean(fl), np.std(fl)],
                                method='nelder-mead', args=(fl))
		peqw = minimize(lnlike_truncnorm, [np.mean(fe), np.std(fe)],
                                method='nelder-mead', args=(fe))
		dummy[f].amplitude.value = pamp.x[0]
		mean_lum = dummy[f].luminosity(ldist)
		dummy[f].amplitude.value = pamp.x[1]
		std_lum = dummy[f].luminosity(ldist)
		results[f] = {'amplitude': pamp.x, 'luminosity':plum.x*10**40, 'eqw':peqw.x}

	if np.any(features == 'PAH 12.62'):
		lum_complex = feature_lum['PAH 12.62'] + feature_lum['PAH 12.69']
		eqw_complex = feature_eqw['PAH 12.62'] + feature_eqw['PAH 12.69']

		lc = sigma_clip(lum_complex, sig=5.0, iters=None)
		ec = sigma_clip(eqw_complex, sig=5.0, iters=None)

		lc = lc[~lc.mask]
		ec = ec[~ec.mask]

		plc = minimize(lnlike_truncnorm, [np.mean(lc), np.std(lc)],
                               method='nelder-mead', args=(lc))
                pec = minimize(lnlike_truncnorm, [np.mean(ec), np.std(ec)],
                               method='nelder-mead', args=(ec))

                results['PAH 12.6 Complex'] = {'luminosity':plc.x*10**40, 'eqw':pec.x}

	if np.any(features == 'PAH 11.23'):
		lum_complex = feature_lum['PAH 11.23'] + feature_lum['PAH 11.33']
		eqw_complex = feature_eqw['PAH 11.23'] + feature_eqw['PAH 11.33']

		lc = sigma_clip(lum_complex, sig=5.0, iters=None)
		ec = sigma_clip(eqw_complex, sig=5.0, iters=None)

		lc = lc[~lc.mask]
		ec = ec[~ec.mask]

		plc = minimize(lnlike_truncnorm, [np.mean(lc), np.std(lc)],
                               method='nelder-mead', args=(lc))
                pec = minimize(lnlike_truncnorm, [np.mean(ec), np.std(ec)],
                               method='nelder-mead', args=(ec))

                results['PAH 11.3 Complex'] = {'luminosity':plc.x*10**40, 'eqw':pec.x}

	if np.any(features == 'PAH 7.60'):
		lum_complex = feature_lum['PAH 7.42'] + feature_lum['PAH 7.60'] + feature_lum['PAH 7.85']
		eqw_complex = feature_eqw['PAH 7.42'] + feature_eqw['PAH 7.60'] + feature_eqw['PAH 7.85']

		lc = sigma_clip(lum_complex, sig=5.0, iters=None)
		ec = sigma_clip(eqw_complex, sig=5.0, iters=None)

		lc = lc[~lc.mask]
		ec = ec[~ec.mask]

		plc = minimize(lnlike_truncnorm, [np.mean(lc), np.std(lc)],
                               method='nelder-mead', args=(lc))
                pec = minimize(lnlike_truncnorm, [np.mean(ec), np.std(ec)],
                               method='nelder-mead', args=(ec))

                results['PAH 7.7 Complex'] = {'luminosity':plc.x*10**40, 'eqw':pec.x}

		
	return results
