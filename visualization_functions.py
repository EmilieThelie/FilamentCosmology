from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt

################## Initial inspection of galaxies and DisPerSE output ##################

# assume axis aligning is done in data loading step
def plot_range_segs_crits_gals(segs, crits, gals, 
                               ax1, ax2,
                               xmin = 0, xmax = 75 / 0.6774,
                               ymin = 0, ymax = 75 / 0.6774,
                               zmin = 0, zmax = 75 / 0.6774):
    mask_segs = ((((segs['U0'] >= xmin) & (segs['U0'] < xmax)) |
                  ((segs['V0'] >= xmin) & (segs['V0'] < xmax))) &
                 (((segs['U1'] >= ymin) & (segs['U1'] < ymax)) |
                  ((segs['V1'] >= ymin) & (segs['V1'] < ymax))) &
                 (((segs['U2'] >= zmin) & (segs['U2'] < zmax)) |
                  ((segs['V2'] >= zmin) & (segs['V2'] < zmax))))
    mask_crits_2 = ((crits['X0'] >= xmin) & (crits['X0'] < xmax) & 
                    (crits['X1'] >= ymin) & (crits['X1'] < ymax) & 
                    (crits['X2'] >= zmin) & (crits['X2'] < zmax)) & (crits['type'] == 2)
    mask_crits_3 = ((crits['X0'] >= xmin) & (crits['X0'] < xmax) & 
                    (crits['X1'] >= ymin) & (crits['X1'] < ymax) & 
                    (crits['X2'] >= zmin) & (crits['X2'] < zmax)) & (crits['type'] == 3)
    mask_gals = ((gals['gal_x'] >= xmin) & (gals['gal_x'] < xmax) & 
                 (gals['gal_y'] >= ymin) & (gals['gal_y'] < ymax) & 
                 (gals['gal_z'] >= zmin) & (gals['gal_z'] < zmax))
    for seg in segs[mask_segs]:
        plt.plot([seg['U' + str(ax1)], seg['V' + str(ax1)]],
                 [seg['U' + str(ax2)], seg['V' + str(ax2)]],
                 c = 'silver')
    plt.scatter(crits['X' + str(ax1)][mask_crits_2], crits['X' + str(ax2)][mask_crits_2],
                marker = 'x', c = 'aquamarine', s = 8)
    plt.scatter(crits['X' + str(ax1)][mask_crits_3], crits['X' + str(ax2)][mask_crits_3],
                marker = 'x', c = 'darkorange', s = 8)
    plt.scatter(gals['gal_' + 'xyz'[ax1]][mask_gals], gals['gal_' + 'xyz'[ax2]][mask_gals],
                c = 'darkslateblue', s = 5, alpha = 0.6)
    plt.xlabel('XYZ'[ax1])
    plt.ylabel('XYZ'[ax2])
    plt.gca().set_aspect('equal')

################## Individual filament plotting with optional galaxy-filament association ##################

def plot_fil(fil_path, ax1, ax2, plot_association = False, fil_gal = None, plot_gal = False, label_axes = False):
    plt.plot([pt[1][ax1] for pt in fil_path], [pt[1][ax2] for pt in fil_path], c = 'silver')
    plt.scatter(fil_path[0][1][ax1], fil_path[0][1][ax2], marker = 'x', c = 'aquamarine', s = 8)
    plt.scatter(fil_path[-1][1][ax1], fil_path[-1][1][ax2], marker = 'x', c = 'darkorange', s = 8)
    if plot_association:
        for i in range(len(fil_gal['galaxy_coords'])):
            plt.plot([fil_gal['galaxy_coords'][i][ax1], fil_gal['connections'][i][ax1]],
                     [fil_gal['galaxy_coords'][i][ax2], fil_gal['connections'][i][ax2]],
                     c = 'pink', lw = 0.5)
    if plot_gal:
        for i in range(len(fil_gal['galaxy_coords'])):
            plt.scatter(fil_gal['galaxy_coords'][i][ax1], fil_gal['galaxy_coords'][i][ax2],
                        c = 'darkslateblue', s = 5, alpha = 0.6)
    plt.gca().set_aspect('equal')
    if label_axes:
        plt.xlabel('XYZ'[ax1])
        plt.ylabel('XYZ'[ax2])

################## Range zoom in with optional galaxy-filament association ##################

def plot_range(fil_paths, gals, fil_gals,
               ax1, ax2,
               xmin = 0, xmax = 75 / 0.6774,
               ymin = 0, ymax = 75 / 0.6774,
               zmin = 0, zmax = 75 / 0.6774,
               plot_association = False):
    mask_gal = ((gals['gal_x'] >= xmin) & (gals['gal_x'] < xmax) & 
                (gals['gal_y'] >= ymin) & (gals['gal_y'] < ymax) & 
                (gals['gal_z'] >= zmin) & (gals['gal_z'] < zmax))
    # saddle point in range
    mask_fil = ((np.array([fil[0][1][0] for fil in fil_paths]) >= xmin) &
                (np.array([fil[0][1][0] for fil in fil_paths]) < xmax) &
                (np.array([fil[0][1][1] for fil in fil_paths]) >= ymin) &
                (np.array([fil[0][1][1] for fil in fil_paths]) < ymax) &
                (np.array([fil[0][1][2] for fil in fil_paths]) >= zmin) &
                (np.array([fil[0][1][2] for fil in fil_paths]) < zmax))
    for i in range(len(fil_paths)):
        if mask_fil[i]:
            plot_fil(fil_paths[i], ax1, ax2, plot_association = plot_association, fil_gal = fil_gals[i])
    plt.scatter(gals['gal_' + 'xyz'[ax1]][mask_gal], gals['gal_' + 'xyz'[ax2]][mask_gal],
                c = 'darkslateblue', alpha = 0.6, s = 5, label = 'galaxies')
    plt.xlabel('XYZ'[ax1])
    plt.ylabel('XYZ'[ax2])
    plt.gca().set_aspect('equal')

################## Network statistics ##################

def plot_network_stats(stats):
    length_bins = np.linspace(0, np.max(stats['l_arr']), 20)
    curve_bins = np.linspace(0, np.max(stats['curve_ratio']), 20)
    connectivity_bins = np.arange(np.max(stats['nfil_per_node']) + 1) - 0.5
    fig, ax = plt.subplots(1, 3, sharey = True, figsize = [12, 3])
    plt.subplots_adjust(wspace = 0)
    fig.suptitle('Filament network statistics')
    ax[0].set_ylabel('Count')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Filament length [Mpc]')
    ax[1].set_xlabel('Curve ratio')
    ax[2].set_xlabel('Node connectivity')
    ax[0].hist(stats['l_arr'], bins = length_bins, ec = 'k', histtype = u'step')
    ax[0].text(length_bins[-1] / 3, 1e3, 'Total length: ' + str(stats['l_tot'])[:8] + 'Mpc')
    ax[1].hist(stats['curve_ratio'], bins = curve_bins, ec = 'k', histtype = u'step')
    ax[1].set_xlim(0.5, np.max(stats['curve_ratio']) + 0.5)
    ax[2].hist(stats['nfil_per_node'], bins = connectivity_bins, ec = 'k', histtype = u'step')

################## Association statistics ##################

def plot_association_stats(stats):
    dist_bins = np.linspace(0, 2, 21)
    richness_bins = np.linspace(0, np.max(stats['richness']), 20)
    mass_richness_bins = np.logspace(7.5, 13, 20)
    dist_crit_bins = np.linspace(0, 15, 20)
    richness_norm_bins = np.linspace(0, np.max(stats['richness'] / stats['l_arr']), 20)
    mass_richness_norm_bins = np.logspace(7, 13.5, 20)
    fig, ax = plt.subplots(2, 3, sharey = True, figsize = [12, 7])
    plt.subplots_adjust(wspace = 0)
    fig.suptitle('Galaxy-filament association statistics')
    ax[0, 0].set_ylabel('Count')
    ax[0, 0].set_yscale('log')
    ax[1, 0].set_ylabel('Count')
    ax[1, 0].set_yscale('log')
    ax[0, 0].set_xlabel('Galaxy-filament distance')
    ax[0, 1].set_xlabel('Number richness')
    ax[0, 2].set_xlabel('Stellar mass richness')
    ax[1, 0].set_xlabel('Connection point location')
    ax[1, 1].set_xlabel('Number richness per unit length')
    ax[1, 2].set_xlabel('Stellar mass richnessper unit length')
    ax[0, 0].hist(stats['distances_all'], bins = dist_bins, ec = 'k', histtype = u'step')
    ax[0, 1].hist(stats['richness'], bins = richness_bins, ec = 'k', histtype = u'step')
    ax[0, 2].hist(stats['mass_richness'], bins = mass_richness_bins, ec = 'k', histtype = u'step')
    ax[0, 2].set_xscale('log')
    ax[0, 1].text(50, 8e3, r'$r(N, L)=$' + str(stats['corr_len_richness'])[:5])
    ax[0, 2].text(1e10, 8e3, r'$r(M_*, L)=$' + str(stats['corr_len_mass_richness'])[:5])
    ax[1, 0].hist(stats['dist_2'], bins = dist_crit_bins, ec = 'k', linestyle = '-', histtype = u'step', label = 'distance to saddle point')
    ax[1, 0].hist(stats['dist_3'], bins = dist_crit_bins, ec = 'k', linestyle = '--', histtype = u'step', label = 'distance to node')
    ax[1, 0].legend(frameon = False)
    ax[1, 1].hist(stats['richness_per_length'], bins = richness_norm_bins, ec = 'k', histtype = u'step')
    ax[1, 2].hist(stats['mass_richness_per_length'], bins = mass_richness_norm_bins, ec = 'k', histtype = u'step')
    ax[1, 2].set_xscale('log')
