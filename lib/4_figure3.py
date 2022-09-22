########################################################################
#                                                                      #
#                MAKE SUBPLOTS FOR FIGURE 3 - PITCH TIPPING            #
#                                                                      #
########################################################################

# Figure 3 is made in inkscape, but the data figures are made here

# Pitches and order_glasnow
pitches_glasnow = ["4-Seam Fastball","Curveball","Slider","Changeup"]    
order_glasnow = order.loc[order.pitch.isin(pitches_glasnow),:]
order_glasnow.reset_index(inplace=True)


# Get data for Tyler Glasnow
glasnow  = data.loc[data.game_year.isin([2018, 2019, 2020]) & data.player_name.isin(["Glasnow, Tyler"]) & data.pitch_name.isin(pitches_glasnow),:].copy()#[2018, 2019, 2020]
glasnow.loc[:,"release_speed_float"] = glasnow.release_speed.astype("float64")
glasnow.loc[:,"release_spin_float"] = glasnow.release_spin_rate.astype("float64")
glasnow.loc[:,"release_pos_z_float"] = glasnow.release_pos_z.astype("float64")
glasnow.loc[:,"pfx_z_float"] = glasnow.pfx_z.astype("float64")
glasnow.loc[:,"plate_z_float"] = glasnow.plate_z.astype("float64")
glasnow.loc[:,"plate_z_norm"] = np.array((glasnow.plate_z - glasnow.sz_bot) / (glasnow.sz_top - glasnow.sz_bot), dtype="float")

# Get data for ALDS game
glasnow_alds = glasnow.loc[glasnow.game_date.isin(["2019-10-10"]),:]

#########################################
#                                       #
#    Figure 1A - Bayesian estimation    #
#                                       #
#########################################

# Get pdf for glasnow data - aka prior
mean_fb,std_fb=stats.norm.fit(glasnow.loc[glasnow.pitch_name.isin(["4-Seam Fastball"]),"plate_z_norm"])
X= np.linspace(-.5, 1.5, 100)
glasnow_prior_fastball = stats.norm.pdf(X, mean_fb, std_fb)
glasnow_prior_fastball /= np.sum(glasnow_prior_fastball)
# Get likelihood
likelihood_fastball = stats.norm.pdf(X, 0.75, 0.2)
likelihood_fastball /= np.sum(likelihood_fastball)
# Get posterior
posterior_fastball = np.multiply(glasnow_prior_fastball,likelihood_fastball)
posterior_fastball /= np.sum(posterior_fastball)

# Get pdf for glasnow data - aka prior
mean_cb,std_cb=stats.norm.fit(glasnow.loc[glasnow.pitch_name.isin(["Curveball"]),"plate_z_norm"])
X= np.linspace(-.5, 1.5, 100)
glasnow_prior_curveball = stats.norm.pdf(X, mean_cb, std_cb)
glasnow_prior_curveball /= np.sum(glasnow_prior_curveball)
# Get likelihood
likelihood_curveball = stats.norm.pdf(X, 0.25, 0.2) 
likelihood_curveball /= np.sum(likelihood_curveball )
# Get posterior
posterior_curveball = np.multiply(glasnow_prior_curveball,likelihood_curveball)
posterior_curveball /= np.sum(posterior_curveball)

# ------------------------------------ #
#                                      #
#            Set up figure             #
#                                      #
# ------------------------------------ #

# Define figure
fig = plt.figure(figsize=(4.5,1.85))

# Add gridspec for legend
ncols=4
gs = fig.add_gridspec(nrows=1, ncols=ncols, left=0.001, right=1, bottom=0.001,top=1)#,width_ratios=[1.25,1,1,1,1,1.5])
ax = []
for ii in range(ncols):
    ax.append(fig.add_subplot(gs[ii]))

# Make figure adjustments
def figure_3A_cleanup(ax):
    ax.set_ylim(-.5,1.5)
    bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
    ax.set_yticks([0,1])#ticks=bin_ticks)
    ax.set_yticklabels([])
    ax.tick_params(axis='y')#,direction='out')
    ax.set_ylabel(None)
    ax.xaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

# ------------------------------------ #
#                                      #
#     Expect fastball, get fastball    #
#                                      #
# ------------------------------------ #

ax[0].plot(glasnow_prior_fastball,X,color="dodgerblue",zorder=2,label="Prior")
ax[0].plot(likelihood_fastball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
ax[0].plot(posterior_fastball,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
ax[0].scatter(posterior_fastball.max(),X[posterior_fastball.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")
ax[0] = figure_3A_cleanup(ax[0])

# ------------------------------------ #
#                                      #
#    Expect curveball, get curveball   #
#                                      #
# ------------------------------------ #
ax[1].plot(glasnow_prior_curveball,X,color="dodgerblue",zorder=2,label="Prior")
ax[1].plot(likelihood_curveball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
ax[1].plot(posterior_curveball,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
ax[1].scatter(posterior_curveball.max(),X[posterior_curveball.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")
ax[1] = figure_3A_cleanup(ax[1])

# ------------------------------------ #
#                                      #
#    Expect fastball, get curveball    #
#                                      #
# ------------------------------------ #
ax[2].plot(glasnow_prior_fastball,X,color="dodgerblue",zorder=2,label="Prior")
ax[2].plot(likelihood_curveball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
_P_mismatch = np.multiply(glasnow_prior_fastball,likelihood_curveball)
_P_mismatch /= np.sum(_P_mismatch)
ax[2].plot(_P_mismatch,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
ax[2].scatter(_P_mismatch.max(),X[_P_mismatch.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")
ax[2] = figure_3A_cleanup(ax[2])

# ------------------------------------ #
#                                      #
#    Expect curveball, get fastball    #
#                                      #
# ------------------------------------ #
ax[3].plot(glasnow_prior_curveball,X,color="dodgerblue",zorder=2,label="Prior")
ax[3].plot(likelihood_fastball,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")
_P_mismatch = np.multiply(glasnow_prior_curveball,likelihood_fastball)
_P_mismatch /= np.sum(_P_mismatch)
ax[3].plot(_P_mismatch,X,color=order_glasnow[order_glasnow.pitch.isin(["Curveball"])].color.tolist()[0],zorder=4,label="Posterior")
ax[3].scatter(_P_mismatch.max(),X[_P_mismatch.argmax()],s=50,color=order_glasnow[order_glasnow.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],zorder=5,marker="*",label="Estimate")
ax[3] = figure_3A_cleanup(ax[3])

# Save the panel for figure 3 A
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure3","Figure3A-1.svg"))

#########################################
#                                       #
#    Figure 3A - Launch angle vs Z      #
#                                       #
#########################################

fig = plt.figure(figsize=(1.5,2))
ax = fig.add_subplot()

# For each bin, plot as jittered scatter plot
for (bin, bin_loc) in zip(sorted(glasnow.plate_z_norm_qbinned.unique()),which_bins_mean):
    _yvals = glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow) & (glasnow.plate_z_norm_qbinned==bin),:].launch_angle
    xwithnoise = len(_yvals)*[bin_loc] + np.random.normal(0,.001,len(_yvals))
    ax.scatter(_yvals,xwithnoise,s=5,color="lightgray")



# Regression for ALDS
thisx=np.arange(-1,2,0.01)
ols_result = run_regression(glasnow_alds.loc[glasnow.pitch_name.isin(pitches_glasnow)].plate_z_norm,glasnow_alds.loc[glasnow.pitch_name.isin(pitches_glasnow)].launch_angle)
ax.plot(ols_result.params[0] + (ols_result.params[1]*thisx),thisx,linewidth=2, color="firebrick",label="Games with tipping")

# Regression line for all pitches_greinke
ols_result = run_regression(glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow)].plate_z_norm,glasnow.loc[glasnow.pitch_name.isin(pitches_glasnow)].launch_angle)
ax.plot(ols_result.params[0] + (ols_result.params[1]*thisx),thisx,linewidth=2, color='k',label="Games with no tipping")

# Clean up
plt.rcParams.update({'font.sans-serif':'Arial','font.size':6})
ax.set_xticks([-100,-50,0,50,100])
ax.set_yticks([0,.5,1])
ax.set_yticklabels(["Knees", "Belt","Chest"],color="firebrick",fontweight="bold",rotation=0,va="center")
ax.set_ylim(-.75, 1.75)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.rcParams.update({'font.sans-serif':'Arial','font.size':6})
ax.set_ylabel("Vertical Plate Position (% strike zone)",fontweight="bold",fontsize=6)
ax.set_xlabel("Launch Angle (deg)",fontweight="bold",fontsize=6)
ax.legend(frameon=False,loc="center",bbox_to_anchor=(.5,1.1))

# Save the p(z,launch angle)
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure3","Figure3A-2.svg"))

#########################################
#                                       #
#    Figure 3A - Simulation of error    #
#                                       #
#########################################
fig, ax = plt.subplots(2,1)
fig.set_size_inches(4,2.5)


nsamples = 1000
likelihood_std = 0.2

# ------------------------------------ #
#                                      #
#       Fastball is true pitch         #
#                                      #
# ------------------------------------ #

error_true = []
error_false = []
X= np.linspace(-.5, 1.5, 100)
for ii in range(nsamples):
    sample_prior = np.random.normal(loc=mean_fb,scale=std_fb,size=1)
    sample_likelihood = np.random.normal(loc=sample_prior,scale=likelihood_std,size=1)
    likelihood = stats.norm.pdf(X, sample_prior,likelihood_std)
    likelihood /= np.sum(likelihood)
    posterior_true = np.multiply(glasnow_prior_fastball,likelihood)
    posterior_true /= np.sum(posterior_true)
    error_true.append(X[posterior_true.argmax()] - sample_likelihood.item())

    sample_prior = np.random.normal(loc=mean_cb,scale=std_cb,size=1)
    sample_likelihood = np.random.normal(loc=sample_prior,scale=likelihood_std,size=1)
    likelihood = stats.norm.pdf(X, sample_prior,likelihood_std)
    likelihood /= np.sum(likelihood)
    posterior_false = np.multiply(glasnow_prior_fastball,likelihood)
    posterior_false /= np.sum(posterior_false)
    error_false.append(X[posterior_false.argmax()] - sample_likelihood.item())

XX= np.linspace(-1, 1, 100)
mean_true,std_true=stats.norm.fit(error_true)
true_dist = stats.norm.pdf(XX,mean_true,std_true)

ax[0].plot(XX,true_dist,color=order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0])
ax[0].axvline(x=mean_true,color=order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0],linestyle=":")
mean_false,std_false=stats.norm.fit(error_false)
false_dist = stats.norm.pdf(XX,mean_false,std_false)
ax[0].plot(XX,false_dist,color='k')
ax[0].axvline(x=mean_false,color='k',linestyle="--")
ax[0].scatter(XX[true_dist.argmax()],true_dist.max(),marker="*",color=order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0])
ax[0].scatter(XX[false_dist.argmax()],false_dist.max(),marker="o",color='k')
ax[0].yaxis.set_visible(False)
ax[0].set_xticks([0])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['left'].set_visible(False)

# ------------------------------------ #
#                                      #
#       Curveball is true pitch         #
#                                      #
# ------------------------------------ #

error_true = []
error_false = []
X= np.linspace(-.5, 1.5, 100)
for ii in range(nsamples):
    sample_prior = np.random.normal(loc=mean_cb,scale=mean_cb,size=1)
    sample_likelihood = np.random.normal(loc=sample_prior,scale=likelihood_std,size=1)
    likelihood = stats.norm.pdf(X, sample_prior,likelihood_std)
    likelihood /= np.sum(likelihood)
    posterior_true = np.multiply(glasnow_prior_curveball,likelihood)
    posterior_true /= np.sum(posterior_true)
    error_true.append(X[posterior_true.argmax()] - sample_likelihood.item())

    sample_prior = np.random.normal(loc=mean_fb,scale=std_fb,size=1)
    sample_likelihood = np.random.normal(loc=sample_prior,scale=likelihood_std,size=1)
    likelihood = stats.norm.pdf(X, sample_prior,likelihood_std)
    likelihood /= np.sum(likelihood)
    posterior_false = np.multiply(glasnow_prior_curveball,likelihood)
    posterior_false /= np.sum(posterior_false)
    error_false.append(X[posterior_false.argmax()] - sample_likelihood.item())


XX= np.linspace(-1, 1, 100)
mean_true,std_true=stats.norm.fit(error_true)
true_dist = stats.norm.pdf(XX,mean_true,std_true)
ax[1].axvline(x=mean_true,color=(.85,.6,0),linestyle=":")
ax[1].plot(XX,true_dist,color=(.85,.6,0))
mean_false,std_false=stats.norm.fit(error_false)
false_dist = stats.norm.pdf(XX,mean_false,std_false)
ax[1].plot(XX,false_dist,color='k')
ax[1].axvline(x=mean_false,color='k',linestyle="--")
ax[1].scatter(XX[true_dist.argmax()],true_dist.max(),marker="*",color=(.85,.6,0))
ax[1].scatter(XX[false_dist.argmax()],false_dist.max(),marker="o",color='k')
ax[1].yaxis.set_visible(False)
ax[1].set_xticks([0])
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_visible(False)

# Save the simulation for figure 3 A
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure3","Figure3A-3.svg"))






########################################################################
#                                                                      #
#                MAKE FIGURE - TYLER GLASNOW PITCH TIPPING             #   
#                                                                      #
########################################################################

# Get data for Tyler Glasnow
tg_all  = data[data.game_year.isin([2018, 2019, 2020]) & data.player_name.isin(["Glasnow, Tyler"])]
tg_alds = data[data.game_date.isin(["2019-10-10"]) & data.player_name.isin(["Glasnow, Tyler"])]

# Define figure layout using gridspec   
fig = plt.figure(figsize=(7.5,3.5))
plt.rcParams.update({'font.sans-serif':'Arial','font.size':9})

# Add first gridspec for left and middle axes
gs0 = fig.add_gridspec(nrows=1, ncols=2, left=0.001, right=0.35,bottom=.01,top=0.95, wspace=0.01)
ax0 = fig.add_subplot(gs0[0])
ax00 = fig.add_subplot(gs0[1])

gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.45, bottom=0.025,right=0.815,top=.9, wspace=.35)
ax1 = fig.add_subplot(gs1[0])

# Add second gridspec for right axis
gs2 = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.5,1], left=0.85, bottom=.025, top=.9, right=0.99,wspace=0.05)
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs2[1])


###########################################
#                                         #
#      Add images for each pitch type     #
#                                         #
###########################################
# Add images
plt.sca(ax0)
img = mpimg.imread('img/tyler-glasnow-pitchtipping-Fastball.jpg')
ax0.imshow(img)
ax0.axis('off') 
sns.despine()
ax0.set_title("Fastball", fontsize=11, fontweight='bold',y=-0.085)

# Add images
plt.sca(ax00)
img = mpimg.imread('img/tyler-glasnow-pitchtipping-Curveball.jpg')
ax00.imshow(img)
ax00.axis('off') 
sns.despine()
ax00.set_title("Curveball", fontsize=11, fontweight='bold',y=-0.085)

# Make violin plot
plt.sca(ax1)

########################################################
#                                                      #
#    Make violinplot and add prior/likelihood lines    #
#                                                      #
########################################################

# Draw lines for prior and likelihood
likelihood_line = ax1.axvline(x=results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0],color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
prior_bias = results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0] - (-0.09*0.5)
prior_line = ax1.plot(extended_xvals*-0.09 + prior_bias,extended_xvals,linestyle=(0,(1,1)),color="black",linewidth=3,zorder=1,label="Mostly Prior")

# Make violinplot
my_violinplot(local_data=tg_all,
              true_shift=which_data_true_shift,
              deviation=which_data_deviation,
              ax=ax1,
              which_pitches=which_pitches,
              bin_locs=which_bins_mean)

# Compute regression  for each game
slope_vals = []
error_vals = []
game_ids   = []
extended_xvals = np.linspace(-0.5,1.5,100)
for game_id in tg_all.game_pk.unique():
    try:

        tempdat = tg_all[(tg_all.game_pk.isin([game_id])) & (tg_all.pitch_name.isin(which_pitches))]
        xx = np.array(tempdat[which_data_true_shift])
        yy = np.array(tempdat[which_data_deviation])
        if (len(xx) >= 10):

            ols_result = run_regression(xx,yy)
            slope_vals.append(ols_result.params[1])
            error_vals.append(ols_result.params[0] + (ols_result.params[1] * (0)))
            
            # Get data from tipping game
            if (game_id == 599341):
                game_ids.append( "Tipping")
            else:
                game_ids.append("Rest of Season")
    except:
        print(len(xx),len(yy))

# Add data to dataframe
scatter_data = pd.DataFrame({"slope": slope_vals,"abs_slope": [abs(ii) for ii in slope_vals], "error": error_vals,"game_ids": game_ids})

# Regression line params
local_color = (0.1,0.1,0.1)
regression_color = "firebrick"
extended_xvals = np.linspace(-0.65,1.65,100)
ols_result = run_regression(tg_all[which_data_true_shift],tg_all[which_data_deviation])
CI = ols_result.get_prediction(sm.add_constant(extended_xvals)).summary_frame(alpha=0.05)
ax1.fill_betweenx(extended_xvals,CI.mean_ci_lower.to_numpy(),CI.mean_ci_upper.to_numpy(),color=local_color,alpha=0.35)#,label="Confidence Intervals")
ax1.plot(ols_result.params[0] + ols_result.params[1]*extended_xvals, extended_xvals,color=local_color,label="No Tipping")

ols_result = run_regression(tg_alds[which_data_true_shift],tg_alds[which_data_deviation])
CI = ols_result.get_prediction(sm.add_constant(extended_xvals)).summary_frame(alpha=0.05)

ax1.fill_betweenx(extended_xvals,CI.mean_ci_lower.to_numpy(),CI.mean_ci_upper.to_numpy(),color=regression_color,alpha=0.05)#,label="Confidence Intervals")
ax1.plot(ols_result.params[0] + ols_result.params[1]*extended_xvals, extended_xvals,color=regression_color,label="No Tipping")
xticksvals = [-.05, 0, 0.05, 0.1]
ax1.set_xticks(xticksvals)
ax1.set_xticklabels(list(map(str,xticksvals)))
sns.despine(offset=10, trim=True,ax=ax1)
ax1.set_yticklabels([" ","Knees (0 %)", "50 %", "Chest (100 %)"," "],rotation=90,va="center")
ax1.get_yticklabels()[1].set_color("firebrick")
ax1.get_yticklabels()[3].set_color("firebrick")
ax1.get_yticklabels()[1].set_weight("bold")
ax1.get_yticklabels()[3].set_weight("bold")
ax1.set_ylabel("Vertical plate position (% strike zone)",fontweight='bold')
ax1.set_xlabel("Vertical contact error (% strike zone)",fontweight='bold')
ax1.set_title("Contact error vs Position in Strike Zone", fontsize=9, fontweight='bold',y=1)
ax1.legend(loc="upper left",frameon=False, bbox_to_anchor=(0,1.275),ncol=2)

###########################################
#                                         #
#  Make boxplot/swarmplot and histogram   #
#                                         #
###########################################

plt.sca(ax2)
sns.histplot(y="abs_slope", data=scatter_data[scatter_data.game_ids=="Rest of Season"],bins=9,color=regression_color,kde=True)
ax2.set_ylabel(None)
ax2.set_xlabel(None)
ax2.xaxis.set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_title("Slope (per game)", fontsize=9, fontweight='bold')
plt.yticks(fontsize=9)

plt.sca(ax3)
sns.boxplot(ax=ax3,y="abs_slope",  data=scatter_data[scatter_data.game_ids=="Rest of Season"],color=local_color,boxprops=dict(alpha=.5))
sns.swarmplot(ax=ax3,y="abs_slope", data=scatter_data[scatter_data.game_ids=="Rest of Season"],color=local_color)
sns.swarmplot(ax=ax3,y="abs_slope", data=scatter_data[scatter_data.game_ids=="Tipping"],color=regression_color,s=20,marker="*",label="$\\bf{p < 0.001}$")
ax3.axis('off')
ax3.set_zorder(3)

# Save the simulation for figure 3 A
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure3","Figure3B.svg"))
