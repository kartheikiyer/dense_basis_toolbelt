from tqdm import tqdm
import emcee
import corner

#-------------------------------------------------------------------------------
#--------------------- getting SEDs in different ways --------------------------
#-------------------------------------------------------------------------------

def get_spec_from_pca_coeffs(pca_coeffs, pca):
    return 10**pca.inverse_transform(pca_coeffs)

def predict_NN_spec(params, net, pca):
    param_var = Variable(torch.from_numpy(params)).float()
    pred_coeffs = net(param_var).detach().numpy()
    pred_spec = get_spec_from_pca_coeffs(pred_coeffs, pca)
    return pred_spec

def make_fcs(lam, z_min, z_max, z_step = 0.01):

    fc_zgrid = np.arange(z_min-z_step, z_max+2*z_step, z_step)

    temp_fc, temp_lz, temp_lz_lores = db.make_filvalkit_simple(lam,priors.z_min,fkit_name = filter_list, filt_dir = filt_dir)

    fcs= np.zeros((temp_fc.shape[0], temp_fc.shape[1], len(fc_zgrid)))
    lzs = np.zeros((temp_lz.shape[0], len(fc_zgrid)))
    lzs_lores = np.zeros((temp_lz_lores.shape[0], len(fc_zgrid)))

    for i in tqdm(range(len(fc_zgrid))):
        fcs[0:,0:,i], lzs[0:,i], lzs_lores[0:,i] = db.make_filvalkit_simple(lam,fc_zgrid[i],fkit_name = filter_list, filt_dir = filt_dir)

    return fcs, fc_zgrid

def sed_from_PCA_NN(theta, fcs, zgrid, net, pca):
    spec = predict_NN_spec(theta, net, pca)
    zarg = np.argmin(np.abs(zgrid - theta[-1]))
    filcurves = fcs[0:,0:,zarg]
    sed = db.calc_fnu_sed_fast(spec, filcurves)
    #sed = db.calc_fnu_sed(spec, theta[-1], lam, fkit_name = filter_list, filt_dir = filt_dir)
    return sed

#----------------------------------older stuff----------------------------------

def sed_from_PCA_NN_slow(theta, model, lam):
    spec = predict_NN_spec(theta)
    sed = db.calc_fnu_sed(spec, theta[-1], lam, fkit_name = filter_list, filt_dir = filt_dir)
    return sed

def sed_from_pregrid(theta):
    mstar, sfr, t50, Z, Av, z = theta
    sfh_tuple = np.array([mstar, sfr, 1.0, t50])
    spec, lam = db.make_spec(sfh_tuple, Z, Av, z, return_lam = True)
    _, sfr_true, mstar_true = db.make_spec(sfh_tuple, Z, Av, z, return_ms = True)
    sed = db.calc_fnu_sed(spec, z, lam, fkit_name = filter_list, filt_dir = filt_dir)
    #print(mstar, mstar_true, mstar-mstar_true, sfr, sfr_true)
    sed = sed/10**(mstar_true-mstar)
    return sed

def sed_from_NDinterp(theta, pg_params, pg_seds):
    interpolator = NearestNDInterpolator(pg_params.T, pg_seds.T)
    sed = interpolator(theta).ravel()
    return sed

def make_specs(theta):
    spec_nn = predict_NN_spec(theta)
    mstar, sfr, t50, Z, Av, z = theta
    sfh_tuple = np.array([mstar, sfr, 1.0, t50])
    spec_fsps, lam = db.make_spec(sfh_tuple, Z, Av, z, return_lam = True)
    _, sfr_true, mstar_true = db.make_spec(sfh_tuple, Z, Av, z, return_ms = True)
    spec_fsps = spec_fsps/10**(mstar_true-mstar)
    return spec_fsps, spec_nn

#-------------------------------------------------------------------------------
#---------------------- emcee priors and likelihoods ---------------------------
#-------------------------------------------------------------------------------

# currently set up only for N_param == 1, generalize to all N_param.

def lnprior(theta):
    mstar, sfr, t50, Z, Av, z = theta
    if 8.0 < mstar < 12.0 and -3.0 < sfr < 3.0 and 0.1 < t50 < 0.9 and -1.0 < Z < 0.5 and 0.0 < Av < 0.5 and 0.9<z<1.1:
        return 0.0
    return -np.inf

# likelihood chi^2
def lnlike(theta, sed, sed_err, fcs, zgrid, net, pca):
    mstar, sfr, t50, Z, Av, z = theta
    model_sed = sed_from_PCA_NN(theta, fcs, zgrid, net, pca)
    chi2 = (sed - model_sed)**2 / (sed_err)**2
    return np.sum(-chi2/2)

def lnprob(theta, sed, sed_err, fcs, zgrid, net, pca):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, sed, sed_err, fcs, zgrid, net, pca)

#-------------------------------------------------------------------------------
#--------------------------------- run emcee -----------------------------------
#-------------------------------------------------------------------------------

def run_mcmc_routine(obs_sed, obs_err, fcs, zgrid, net, pca, pg_params, nwalkers = 100, nsteps = 10000, n_burnin = 100, threads = 6):

    ndim = pg_params.shape[0]

    pos = pg_params[0:,np.random.choice(pg_params.shape[1], size = nwalkers)].T

    #fig = corner.corner(pos, labels = ['log M*', 'log SFR', 't50', 'met', 'dust','redshift'],
    #                    truths = [rand_sfh_tuple[0], rand_sfh_tuple[1], rand_sfh_tuple[3], rand_Z, rand_Av,rand_z])
    #fig.set_size_inches(12,12)
    #plt.show()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (obs_sed, obs_err, fcs, zgrid, net, pca), threads = threads)

    time0 = time.time()
    pos, prob, state = sampler.run_mcmc(pos, n_burnin)
    sampler.reset()

    time1 = time.time()
    print('burn-in time: %.1f sec' %(time1-time0))

    time0 = time.time()

    width = 100
    for i, result in enumerate(sampler.sample(pos, iterations = nsteps)):
        n = int((width+1)*float(i)/nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n , ' '*(width-n)))
    sys.stdout.write("\n")

    time1 = time.time()
    print('time taken to run: %.1f min.' %((time1-time0)/60))

    samples = sampler.flatchain

    return samples

def plot_emcee_posterior(samples, sed_truths = []):

    if len(sed_truths) > 1:
        fig = corner.corner(samples, labels = ['log M*', 'log SFR', 't$_{50}$', 'log Z/Z$_\odot$', 'A$_V$', 'redshift'],
                            truths = sed_truths,
                            plot_datapoints=False, fill_contours=True,
                            bins=20, smooth=1.0,
                            quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                            label_kwargs={"fontsize": 30}, show_titles=True)

    else:
        fig = corner.corner(samples, labels = ['log M*', 'log SFR', 't$_{50}$', 'log Z/Z$_\odot$', 'A$_V$', 'redshift'],
                            plot_datapoints=False, fill_contours=True,
                            bins=20, smooth=1.0,
                            quantiles=(0.16, 0.84), levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                            label_kwargs={"fontsize": 30}, show_titles=True)

    fig.subplots_adjust(right=1.5,top=1.5)
    fig.set_size_inches(12,12)
    plt.show()
