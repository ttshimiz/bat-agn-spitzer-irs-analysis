# Script to fit all of the BAT AGN IRS spectral for prominent PAH features
# and mid-IR emission lines
# PAH features include the 6.2, 7.7, and 11.3 micron complex
# Mid-IR emission lines include [NeII], [NeIII], [NeV], and [OIV]

import bat_agn_irs_tools as bait
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

def fit_all_features(name, ldist, fig_save_dir=None,
                     chain_save_dir=None):
	
	spec = bait.create_bat_spec(name)
	
	if fig_save_dir is None:
		fig_save_dir = ''
	if chain_save_dir is None:
		chain_save_dir = ''
	
	result_pah11_3 = bait.fit_pah11_3(spec)
	np.savetxt(chain_save_dir+name+'_pah11_3_chain.txt', result_pah11_3.chain)
	fig_pah11_3 = bait.plot_fit(spec, result_pah11_3, 10.0, 12.5)
	fig_pah11_3.savefig(fig_save_dir+name+'_pah11_3_fit.png', bbox_inches='tight')
	plt.close(fig_pah11_3)
	fig_triangle_pah11_3 = bait.plot_triangle(result_pah11_3)
	fig_triangle_pah11_3.savefig(fig_save_dir+name+'_pah11_3_corner.png', bbox_inches='tight')
	plt.close(fig_triangle_pah11_3)
	
	result_pah7_7 = bait.fit_pah7_7(spec)
	np.savetxt(chain_save_dir+name+'_pah7_7_chain.txt', result_pah7_7.chain)
	fig_pah7_7 = bait.plot_fit(spec, result_pah7_7, 6.7, 8.7)
	fig_pah7_7.savefig(fig_save_dir+name+'_pah7_7_fit.png', bbox_inches='tight')
	fig_pah7_7.close()
	fig_triangle_pah7_7 = bait.plot_triangle(result_pah7_7)
	fig_triangle_pah7_7.savefig(fig_save_dir+name+'_pah7_7_corner.png', bbox_inches='tight')
	fig_triangle_pah7_7.close()

	result_pah6_2 = bait.fit_pah6_2(spec)
	np.savetxt(chain_save_dir+name+'_pah6_2_chain.txt', result_pah6_2.chain)
	fig_pah6_2 = bait.plot_fit(spec, result_pah6_2, 5.2, 6.7)
	fig_pah6_2.savefig(fig_save_dir+name+'_pah6_2_fit.png', bbox_inches='tight')
	fig_pah6_2.close()
	fig_triangle_pah6_2 = bait.plot_triangle(result_pah6_2)
	fig_triangle_pah6_2.savefig(fig_save_dir+name+'_pah6_2_corner.png', bbox_inches='tight')
	fig_triangle_pah6_2.close()


	result_neII = bait.fit_neII(spec)
	np.savetxt(chain_save_dir+name+'_neII_chain.txt', result_neII.chain)
	fig_neII = bait.plot_fit(spec, result_neII, 12.0, 14.0)
	fig_neII.savefig(fig_save_dir+name+'_neII_fit.png', bbox_inches='tight')
	fig_neII.close()
	fig_triangle_neII = bait.plot_triangle(result_neII)
	fig_triangle_neII.savefig(fig_save_dir+name+'_neII_corner.png', bbox_inches='tight')
	fig_triangle_neII.close()

	result_neIII = bait.fit_neIII(spec)
	np.savetxt(chain_save_dir+name+'_neIII_chain.txt', result_neIII.chain)
	fig_neIII = bait.plot_fit(spec, result_neIII, 14.5, 16.5)
	fig_neIII.savefig(fig_save_dir+name+'_neIII_fit.png', bbox_inches='tight')
	fig_neIII.close()
	fig_triangle_neIII = bait.plot_triangle(result_neIII)
	fig_triangle_neIII.savefig(fig_save_dir+name+'_neIII_corner.png', bbox_inches='tight')
	fig_triangle_neIII.close()

	result_neV = bait.fit_neV(spec)
	np.savetxt(chain_save_dir+name+'_neV_chain.txt', result_neV.chain)
	fig_neV = bait.plot_fit(spec, result_neV, 13.3, 15.3)
	fig_neV.savefig(fig_save_dir+name+'_neV_fit.png', bbox_inches='tight')
	fig_neV.close()
	fig_triangle_neV = bait.plot_triangle(result_neV)
	fig_triangle_neV.savefig(fig_save_dir+name+'_neV_corner.png', bbox_inches='tight')
	fig_triangle_neV.close()

	result_oiv = bait.fit_oiv(spec)
	np.savetxt(chain_save_dir+name+'_oiv_chain.txt', result_oiv.chain)
	fig_oiv = bait.plot_fit(spec, result_oiv, 24.5, 27.5)
	fig_oiv.savefig(fig_save_dir+name+'_oiv_fit.png', bbox_inches='tight')
	fig_oiv.close()
	fig_triangle_oiv = bait.plot_triangle(result_oiv)
	fig_triangle_oiv.savefig(fig_save_dir+name+'_oiv_corner.png', bbox_inches='tight')
	fig_triangle_oiv.close()

	return 0

def get_lum_eqw(model, nsamp=1000, nburn=1000, nsteps=5000, nwalkers=50):
	
	# Remove the burn-in steps for each walker
	dummy = model.copy()
	chain = model.chain
	chain_length = nwalkers*nsteps
	ndims = np.shape(chain)[1]
	chain_burned = np.zeros((nwalkers*(nsteps-nburn), ndims)
	for i in range(nwalkers):
		chain_burned[i*(nsteps-nburn):(i+1)*(nsteps-nburn), :] = chain[i*nsteps+nburn:(i+1)*nsteps]
	
	# Get random indices to use to sample the posterior distributions
	ri = np.random.choice(nwalkers*(nsteps-nburn), 1000, replace=False)
	
	
	
bat_irs_obs = pd.read_csv('/Users/ttshimiz/Research/Thesis/bat-agn-spitzer-irs-analysis/bat_agn_spitzer_irs_observations.csv',
	                          index_col=0)
bat_info = pd.read_csv('/Users/ttshimiz/Github/bat-data/bat_info.csv', index_col=0)
	
ind_fit = ((bat_irs_obs['Spitzer Spectra?'] == 'Yes') & (bat_irs_obs['Spectra Orders'] == 'SL/LL') &
	       (bat_irs_obs['Spectra Type'] != 'B'))

pool = mp.Pool(processes=7)	
results = [pool.apply_async(fit_all_features, args=(n, bat_info.loc[n, 'Dist_[Mpc]'])) for n in bat_irs_obs.index[ind_fit]]
	
	