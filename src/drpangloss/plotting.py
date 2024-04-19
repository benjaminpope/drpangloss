import matplotlib.pyplot as plt
import numpy as np
import matplotlib


matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['font.family'] = ['serif']
plt.rcParams.update({'font.size': 14})

'''
Plotting functions to automatically render outputs from the model fitting functions.
'''

def plot_likelihood_grid(loglike_im, samples_dict,truths=None):
    '''
    Plot the results of a likelihood_grid calculation.

    Parameters
    ----------
    loglike_im : array
        The likelihood grid, output of likelihood_grid
    samples_dict : dict
        Dictionary of samples used in the grid calculation
    truths : list, optional 
        List of true values for the parameters, default None
    '''

    plt.figure(figsize=(12,6))

    plt.imshow(loglike_im.T, cmap="inferno",aspect="equal",
            extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                            samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
    plt.colorbar(shrink=1,label='Log likelihood', pad=0.01)
    plt.scatter(0,0,s=140,c='y',marker='*')

    plt.xlabel('$\\Delta$RA [mas]')
    plt.ylabel('$\\Delta$DEC [mas]')

    if truths is not None:
        dra_inp, ddec_inp = truths[0], truths[1]
        plt.scatter(dra_inp,ddec_inp, 
                s=500,linestyle='--', facecolors='none', edgecolors='white',linewidths=2,alpha=1)
    plt.gca().invert_yaxis() # up is more Dec

def plot_optimized_and_grid(loglike_im, optimized, samples_dict):

    best_contrast_indices = np.argmax(loglike_im,axis=2)
    best_contrasts = samples_dict['flux'][best_contrast_indices]

    plt.figure(figsize=(14,5))
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['font.family'] = ['serif']
    plt.rcParams.update({'font.size': 14})
    plt.subplot(1,2,1)
    plt.imshow(optimized.T,cmap='inferno', norm = matplotlib.colors.LogNorm(),
                        extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                        samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
    plt.colorbar(shrink=1,label='Contrast', pad=0.01)
    plt.scatter(0,0,s=140,c='black',marker='*')
    plt.xlabel('$\\Delta$RA [mas]')
    plt.ylabel('$\\Delta$DEC [mas]')
    plt.title('Optimization')
    plt.gca().invert_yaxis()

    plt.subplot(1,2,2)
    plt.imshow(best_contrasts.T,cmap='inferno', norm = matplotlib.colors.LogNorm(),
                            extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                            samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
    plt.colorbar(shrink=1,label='Contrast', pad=0.01)
    plt.scatter(0,0,s=140,c='black',marker='*')
    plt.xlabel('$\\Delta$RA [mas]')
    plt.ylabel('$\\Delta$DEC [mas]')
    plt.title('Grid Search')
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0.)
    plt.show()

def plot_optimized_and_sigma(contrast, sigma_grid, samples_dict,snr=False):

    '''
    Plot the results of an optimized contrast grid calculation and the corresponding uncertainty grid.

    Parameters
    ----------
    contrast : array
        The optimized contrast grid, output of optimized_contrast_grid  
    sigma_grid : array
        The uncertainty grid, output of laplace_contrast_uncertainty_grid
    samples_dict : dict
        Dictionary of samples used in the grid calculation
    snr : bool, optional
        If True, plot the SNR instead of the uncertainty, default False
    
    '''

    plt.figure(figsize=(14,5))
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['font.family'] = ['serif']
    plt.rcParams.update({'font.size': 14})
    plt.subplot(1,2,1)
    plt.imshow(contrast.T,cmap='inferno', norm = matplotlib.colors.LogNorm(),
                        extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                        samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
    plt.colorbar(shrink=1,label='Contrast', pad=0.01)
    plt.scatter(0,0,s=140,c='y',marker='*')
    plt.xlabel('$\\Delta$RA [mas]')
    plt.ylabel('$\\Delta$DEC [mas]')
    plt.title('Contrast')
    plt.gca().invert_yaxis()

    plt.subplot(1,2,2)
    if snr:
        plt.imshow(contrast.T/sigma_grid.T,cmap='inferno',norm=matplotlib.colors.PowerNorm(1),
                                extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                                samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
        plt.colorbar(shrink=1,label='SNR', pad=0.01)
        plt.scatter(0,0,s=140,c='y',marker='*') # mark star at origin
        plt.title('SNR')

    else:
        plt.imshow(sigma_grid.T,cmap='inferno', norm = matplotlib.colors.LogNorm(),
                                extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                                samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
        plt.colorbar(shrink=1,label='σ(Contrast)', pad=0.01)
        plt.scatter(0,0,s=140,c='y',marker='*') # mark star at origin
        plt.title('σ(Contrast)')

    plt.xlabel('$\\Delta$RA [mas]')
    plt.ylabel('$\\Delta$DEC [mas]')
    plt.gca().invert_yaxis()
    plt.tight_layout(pad=0.)
    plt.show()

def plot_contrast_limits(contrast_limits, samples_dict, rad_width, avg_width, std_width, true_values=None):  
    ''' 
    Plot the contrast limits calculated with the Ruffio or Absil methods.

    TODO: pass in the percentile or number of sigmas as a parameter to be used in legends.

    Parameters
    ----------
    contrast_limits : array
        The contrast limits calculated with the Ruffio or Absil methods.
    samples_dict : dict
        Dictionary of samples used in the grid calculation
    rad_width : array
        Radial width of the contrast limits.
    avg_width : array
        Average width of the contrast limits.
    std_width : array
        Standard deviation of the contrast limits.
    true_values : list, optional
        List of true values for the parameters, default None         
    
    '''

    plt.figure(figsize=(20,5))
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['font.family'] = ['serif']
    plt.rcParams.update({'font.size': 16})


    # first show x% upper limit map

    plt.subplot(1,2,1)
    plt.imshow(-2.5*np.log10(contrast_limits[:,:].T),cmap=matplotlib.colormaps['magma_r'],
                                extent = [samples_dict["dra"].max(), samples_dict["dra"].min(), # this may seem weird, but left is more RA and up is more Dec
                                samples_dict["ddec"].max(), samples_dict["ddec"].min()]) # this took me far too long to get the sign right for
    plt.colorbar(shrink=1, pad=0.01)
    plt.scatter(0,0,marker='*',s=100,c='black',alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title('98% Upper Limit Map ($\\Delta$mag)')
    plt.xlabel('$\\Delta$RA [mas]')
    plt.ylabel('$\\Delta$DEC [mas]')


    # then show contrast curve including detected target
    plt.subplot(1,2,2)
    dx = np.abs(np.median(np.diff(samples_dict["dra"])))
    plt.plot(rad_width*dx,avg_width,'-k',label="98% Upper Limit")
    plt.fill_between(rad_width*dx,avg_width - std_width,avg_width + std_width,color=(0.6,0.4,0.9),alpha=0.3)
    plt.ylabel('Contrast ($\\Delta$mag)')
    plt.xlabel('Separation [mas]')
    plt.gca().invert_yaxis()
    plt.xlim(np.nanmin(rad_width*dx+avg_width*0.),np.nanmax(rad_width*dx+avg_width*0.))
    plt.grid(color='black',alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout(pad=0.)

    if true_values is not None:
        true_dra, true_ddec, true_contrast = true_values
        plt.plot(np.sqrt(true_dra**2+true_ddec**2),-2.5*np.log10(true_contrast),marker='*',c='k',markersize=15) # detected value
