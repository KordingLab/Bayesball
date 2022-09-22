########################################################################
#                                                                      #
#                            FIGURE FUNCTIONS                          #
#                                                                      #
########################################################################

def get_color_list(cmap="plasma"):
    these_colors = []
    for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
        these_colors.append([*row,1.])
    return these_colors

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def calc_bin_means(x):
    return ((x + np.roll(x,1))/2.0)[1:]

def my_violinplot(local_data=[],true_shift=[],deviation=[],ax=[],which_pitches=[],bin_locs=[],widths=.1):
    
    ''' Based on this code: https://matplotlib.org/stable/gallery/statistics/customized_violin.html '''

    local_data = local_data.loc[local_data.pitch_name.isin(which_pitches),:].copy()

    # Create new axis if none is provided
    if ax is None:
        ax = plt.gca()
        
    # Convert data into numpy array with different column sizes (http://www.asifr.com/transform-grouped-dataframe-to-numpy.html)
    # xt  = local_data[local_data.pitch_name.isin(which_pitches)].loc[:,[deviation]].values
    # g   = local_data[local_data.pitch_name.isin(which_pitches)].reset_index(drop=True).groupby(true_shift + "_qbinned")
    xt  = local_data.loc[:,[deviation]].values
    g   = local_data.reset_index(drop=True).groupby(true_shift + "_qbinned")
    xtg = [xt[i.values,:] for k,i in g.groups.items()]
    xout = np.array(xtg,dtype=object)

    # Make violin plot
    parts = plt.violinplot(xout,positions=bin_locs, vert=False,showmeans=False, showmedians=False,
            showextrema=False,widths=widths)

    # Customize - mostly copied from https://matplotlib.org/stable/gallery/statistics/customized_violin.html
    for pc in parts['bodies']:
        pc.set_facecolor('lightgrey')
        pc.set_edgecolor('darkgrey')
        pc.set_alpha(1)
        pc.set_zorder(2)

        # pc.set_facecolor((0.7,0.7,0.7))
        # pc.set_edgecolor('black')
        # pc.set_alpha(1)
    
    
    # Customize violins
    quartile1, medians, quartile3 = [], [], []
    for arr in xout:
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        quartile1.append(q1)
        medians.append(np.mean(arr))
        quartile3.append(q3)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(xout, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    iqr_color=(0.5,0.5,0.5)#"grey"
    s1   = ax.scatter(medians, bin_locs, marker='o', color='white', s=15, zorder=4)
    ln1  = ax.hlines(bin_locs, quartile1, quartile3, color=iqr_color, linestyle='-', lw=5)
    s2   = ax.scatter(quartile3,bin_locs, marker='o', color=iqr_color, s=12, zorder=3)
    s3   = ax.scatter(quartile1,bin_locs, marker='o', color=iqr_color, s=12, zorder=3)
    ln2  = ax.hlines(bin_locs, whiskers_min, whiskers_max, color=iqr_color, linestyle='-', lw=1)
    
    return ax, parts

def run_regression(xx,yy):
    X = sm.add_constant(xx)#, prepend=False)
    ols = sm.OLS(yy,X)
    ols_result = ols.fit()
    return ols_result

# Calculate time for pitch velo and distance
def time_for_pitch_velocity_and_distance(velocities=[95],distances=[20, 40, 60]):
    
    """ This function uses a VERY basic calculation to compute an estimate of how long it takes for a pitch to reach different positions. 
    Does not assume slowing down of pitch over time.  A future version will be based on the trajectory calculator by Alan N. Nathan. """

    for velo in velocities:
        print('It takes a ball thrown {} mph: '.format(velo))
        for d in distances:
            t = d / (velo * 1.46667)
            print("{} ms to go {} feet.".format(round(t,5)*1000,d))
        print("-----------------------------")