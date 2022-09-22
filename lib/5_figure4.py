########################################################################
#                                                                      #
#           MAKE SUBPLOTS FOR FIGURE 4 - KNUCKLEBALL & EEPHUS          #
#                                                                      #
########################################################################
 
# Pitches and order_dickey
pitches_dickey = ["4-Seam Fastball","Knuckleball"]   
order_addKN = pd.concat([order,pd.DataFrame({"pitch": ["Knuckleball"], "color": ["black"],"marker": ["v"], "code": [pitch_type_codes["Knuckleball"]]})],ignore_index=True, axis=0) 
order_dickey = order_addKN.loc[order_addKN.pitch.isin(pitches_dickey),:]
order_dickey.reset_index(inplace=True)

# Get data for R.A. Dickey
dickey  = data.loc[data.player_name.isin(["Dickey, R.A."]) & data.pitch_name.isin(pitches_dickey),:].copy() 
dickey.loc[:,"release_speed_float"] = dickey.release_speed.astype("float64")
dickey.loc[:,"release_spin_float"] = dickey.release_spin_rate.astype("float64")
dickey.loc[:,"release_pos_z_float"] = dickey.release_pos_z.astype("float64")
dickey.loc[:,"pfx_z_float"] = dickey.pfx_z.astype("float64")
dickey.loc[:,"plate_z_float"] = dickey.plate_z.astype("float64")
dickey.loc[:,"plate_z_norm"] = np.array((dickey.plate_z - dickey.sz_bot) / (dickey.sz_top - dickey.sz_bot), dtype="float")

prior_color = "dodgerblue"
likelihood_color = (0.5,0.5,0.5)
posterior_color = order[order.pitch.isin(["Curveball"])].color.tolist()[0]
MAPcolor = order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]

##############################
#                            #
#      Calculate priors      #
#                            #
##############################

# Get pdf for dickey data - aka prior
mean,std=stats.norm.fit(dickey.loc[dickey.pitch_name.isin(["Knuckleball"]),"plate_z_norm"])#dickey.plate_z_norm)
X= np.linspace(-.5, 1.5, 100)
dickey_prior_knuckleball_true = stats.norm.pdf(X, mean,std) #stats.norm.pdf(X, mean, std) #stats.norm.pdf(X, 0.5, 0.5) #
dickey_prior_knuckleball_true /= np.sum(dickey_prior_knuckleball_true)
dickey_prior_knuckleball = stats.norm.pdf(X, 0.5, 0.6) #stats.norm.pdf(X, mean, std) #stats.norm.pdf(X, 0.5, 0.5) #
dickey_prior_knuckleball /= np.sum(dickey_prior_knuckleball)


#########################################
#                                       #
#      Full bayes for Knuckleball       #
#                                       #
#########################################

fig = plt.figure(figsize=(10,1.85))

# Add grid spec for data
nn = 3
gs1 = fig.add_gridspec(nrows=1, ncols=nn, wspace=0.05, left=0.2, right=.75,bottom=0.01,top=0.85)# bottom=0.55,top=1)#,hspace=.25)
ax1 = []
for ii in range(nn):
    thisax = fig.add_subplot(gs1[ii])
    # thisax.set_ylim(-.5,1.5)
    thisax.spines['top'].set_visible(False)
    thisax.spines['right'].set_visible(False)
    thisax.spines['bottom'].set_visible(False)

    ax1.append(thisax)

# Get pdf for greinke data
mean,std=stats.norm.fit(greinke.plate_z_norm)
xx = np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(xx, mean, std) 
xpos, ypos_chest, ypos_knees = 1, 1.15, -.2 #.9875, -.02
greinke_prior /= np.sum(greinke_prior)

ll = []
likelihood = []
posterior = []
MAP=[]
MAPval=[]
MAPidx=[]
MAPx=[]
MAPy=[]
Lcolor="grey"
Pcolor=order[order.pitch.isin(["Curveball"])].color.tolist()[0]
MAPcolor = order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]

xvals=np.linspace(0,1,nn)

# Bayes
for cnt,xval in enumerate(xvals):
    
    # Make likelihood and multiply with prior to get posterior
    _L = stats.norm.pdf(X, xval, 0.25)
    _L /= np.sum(_L)
    likelihood.append(_L)
    _P = np.multiply(_L,dickey_prior_knuckleball)
    _P /= np.sum(_P)

    # Plot everything
    posterior.append(_P)
    ax1[cnt].plot(dickey_prior_knuckleball,xx,color="dodgerblue",zorder=3,lw=1.5)
    ax1[cnt].plot(_L,xx,color=Lcolor,zorder=3,lw=1.5)
    ax1[cnt].plot(_P,xx,color=Pcolor,zorder=4,lw=1.5)#order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]
    ax1[cnt].axvline(x=0,color="darkgrey",linestyle="-")

    # Compute MAP and store
    ax1[cnt].scatter(_P.max(),xx[_P.argmax()],marker="*",color=MAPcolor,s=30,zorder=5,clip_on=False)
    xlims = ax1[cnt].get_xlim()
    ax1[cnt].set_xlim(0,np.maximum(_P.max(),_L.max())+.01)
    MAPval.append(_P.max())
    MAPidx.append(xx[_P.argmax()])
    
    # Clean up
    ax1[cnt].set_yticks([0,1])
    ax1[cnt].set_yticklabels([" "," "])
    ax1[cnt].set_ylim(-.65,1.65)
    ax1[cnt].get_xaxis().set_visible(False)

# Save the panel for figure 4B
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure4","Figure4B.svg"))


#########################################
#                                       #
#         Full bayes for Eephus         #
#                                       #
#########################################

fig = plt.figure(figsize=(10,1.85))

# Add grid spec for data
nn = 3
gs1 = fig.add_gridspec(nrows=1, ncols=nn, wspace=0.05, left=0.2, right=.75,bottom=0.01,top=0.85)# bottom=0.55,top=1)#,hspace=.25)
ax1 = []
for ii in range(nn):
    thisax = fig.add_subplot(gs1[ii])
    # thisax.set_ylim(-.5,1.5)
    thisax.spines['top'].set_visible(False)
    thisax.spines['right'].set_visible(False)
    thisax.spines['bottom'].set_visible(False)

    ax1.append(thisax)

# Get pdf for greinke data
mean,std=stats.norm.fit(greinke.plate_z_norm)
xx = np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(xx, mean, std) 
xpos, ypos_chest, ypos_knees = 1, 1.15, -.2 #.9875, -.02
greinke_prior /= np.sum(greinke_prior)

ll = []
likelihood = []
posterior = []
MAP=[]
MAPval=[]
MAPidx=[]
MAPx=[]
MAPy=[]
Lcolor="grey"
Pcolor=order[order.pitch.isin(["Curveball"])].color.tolist()[0]
MAPcolor = order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]

xvals=np.linspace(0,1,nn)

# Bayes
for cnt,xval in enumerate(xvals):

    # Make likelihood and multiply with prior to get posterior
    _L = stats.norm.pdf(X, xval, 0.1)
    _L /= np.sum(_L)
    likelihood.append(_L)
    _P = np.multiply(_L,dickey_prior_knuckleball)
    _P /= np.sum(_P)


    # Plot everything
    posterior.append(_P)
    ax1[cnt].plot(dickey_prior_knuckleball,xx,color="dodgerblue",zorder=3,lw=1.5)
    ax1[cnt].plot(_L,xx,color=Lcolor,zorder=3,lw=1.5)
    ax1[cnt].plot(_P,xx,color=Pcolor,zorder=4,lw=1.5)
    ax1[cnt].axvline(x=0,color="darkgrey",linestyle="-")

    # Compute MAP and store
    ax1[cnt].scatter(_P.max(),xx[_P.argmax()],marker="*",color=MAPcolor,s=30,zorder=5,clip_on=False)
    xlims = ax1[cnt].get_xlim()
    ax1[cnt].set_xlim(0,np.maximum(_P.max(),_L.max())+.01)
    MAPval.append(_P.max())
    MAPidx.append(xx[_P.argmax()])
    
    # Clean up
    ax1[cnt].set_yticks([0,1])
    ax1[cnt].set_yticklabels([" "," "])
    ax1[cnt].set_ylim(-.65,1.65)
    ax1[cnt].get_xaxis().set_visible(False)

# Save the panel for figure 4C
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure4","Figure4C.svg"))





########################################################################
#                                                                      #
#      MAKE FIGURE - SIDE BY SIDE KNUCKLEBALL AND EEPHUS               #   
#                                                                      #
########################################################################

# Define figure
fig = plt.figure(figsize=(7,2.5))
plt.rcParams.update({'font.sans-serif':'Arial','font.size':9})
# Add first gridspec for violinplot
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.45, bottom=0.05,top=0.9)
ax1 = fig.add_subplot(gs1[0])

# Add gridspec for slope bar plot
gs2 = fig.add_gridspec(nrows=2, ncols=1, left=0.625, right=0.99,bottom=.05,top=.9,hspace=.3)
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs2[1])

#### Define colors ####
from matplotlib import cm
cmap = cm.get_cmap('plasma')
define_colors = [cmap(0.8), cmap(0.5)]
these_colors = len(which_pitches)*[(0.5,0.5,0.5,1)] + define_colors#[]
markers = ["X","^","*","s","o","P","D"]

##################################################
#                                                #   
#        Make violin plot for Knuckleball        #
#                                                #
##################################################

plt.sca(ax1)
likelihood_line = ax1.plot(len(extended_xvals)*[results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0]],extended_xvals,color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
prior_bias = results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0] - (-0.075*0.5)
prior_line = ax1.plot((extended_xvals*-0.075 + prior_bias),extended_xvals,linestyle=(0,(1,1)),color="black",linewidth=3,zorder=1,label="Mostly Prior")

my_violinplot(local_data=data,
              true_shift=which_data_true_shift,
              deviation=which_data_deviation,
              ax=ax1,
              which_pitches=which_pitches,
              bin_locs=which_bins_mean)

# Get data for plotting
xvals = np.unique(xx_all)
yvals = results_by_pitch[results_by_pitch.pitch.isin(["All"])].const.values[0] + (results_by_pitch[results_by_pitch.pitch.isin(["All"])].means.values[0] * xvals)
extended_xvals = np.linspace(-0.5,1.5,100)

# Plot regression line
regression_color = 'black'#'dodgerblue' #np.array((0, 158, 115))/256) # blue: 0,114,178 ; green = 0, 158, 115 ; pink = 204,121,167
ols_result = run_regression(data.loc[data.pitch_name.isin(which_pitches),which_data_true_shift],data.loc[data.pitch_name.isin(which_pitches),which_data_deviation])
CI = ols_result.get_prediction(sm.add_constant(extended_xvals)).summary_frame(alpha=0.05)
ax1.fill_betweenx(extended_xvals,CI.mean_ci_lower.to_numpy(),CI.mean_ci_upper.to_numpy(),color=regression_color,alpha=0.15)#
reg = ax1.plot(yvals,xvals,lw=2,color=regression_color,label="All")

# Knuckleball data
knuckleballs = data.loc[data.pitch_name.isin(['Knuckleball']),:].copy()
ols_result = run_regression(knuckleballs[which_data_true_shift],knuckleballs[which_data_deviation])
CI = ols_result.get_prediction(sm.add_constant(extended_xvals)).summary_frame(alpha=0.1)
ax1.fill_betweenx(extended_xvals,CI.mean_ci_lower.to_numpy(),CI.mean_ci_upper.to_numpy(),color=define_colors[0],alpha=0.15)#,label="Confidence Intervals")
ax1.plot(ols_result.params[0] + ols_result.params[1]*extended_xvals, extended_xvals,color=define_colors[0],label="Knuckleball")
        
# Eephus data
eephus = data.loc[data.pitch_name.isin(['Eephus']),:].copy()
ols_result = run_regression(eephus[which_data_true_shift],eephus[which_data_deviation])
CI = ols_result.get_prediction(sm.add_constant(extended_xvals)).summary_frame(alpha=0.1)
ax1.fill_betweenx(extended_xvals,CI.mean_ci_lower.to_numpy(),CI.mean_ci_upper.to_numpy(),color=define_colors[1],alpha=0.15)#
ax1.plot(ols_result.params[0] + ols_result.params[1]*extended_xvals, extended_xvals,color=define_colors[1],label="Eephus")

# Get knuckleball results    
xx = np.array(knuckleballs[which_data_true_shift])#_qbinned_percent_mean
yy = np.array(knuckleballs[which_data_deviation])
X = sm.add_constant(xx)#, prepend=False)
ols = sm.OLS(yy,X)
ols_result = ols.fit()
slope_vals.append(ols_result.params[1])
error_vals.append(ols_result.params[0] + (ols_result.params[1] * (0.4)))
prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(ols_result) # for getting confidence intervals
xvals = np.unique(xx)
yvals = ols_result.params[0] + ols_result.params[1] * xvals
   
knuckleball_df = pd.DataFrame({'pitch': ['Knuckleball'],'pitch_code': ['KN'],
                               'const': [ols_result.params[0]],
                               'means': [ols_result.params[1]], 
                               'absmeans': [abs(ols_result.params[1])], 
                               'se_mean': [[ols_result.bse[0], ols_result.bse[1]]],
                               'se_mean_lo': [ols_result.params[1] - ols_result.bse[0]],
                               'se_mean_hi': [ols_result.params[1] + ols_result.bse[1]],
                                'error_at_mid': [ols_result.params[0] + ols_result.params[1] * (0)]})
    
# Get eephus results
xx = np.array(eephus[which_data_true_shift])#_qbinned_percent_mean
yy = np.array(eephus[which_data_deviation])
X = sm.add_constant(xx)#, prepend=False)
ols = sm.OLS(yy,X)
ols_result = ols.fit()
slope_vals.append(ols_result.params[1])
error_vals.append(ols_result.params[0] + (ols_result.params[1] * (0.4)))
prstd_ols, iv_l_ols, iv_u_ols = wls_prediction_std(ols_result) # for getting confidence intervals
xvals = np.unique(xx)
yvals = ols_result.params[0] + ols_result.params[1] * xvals

eephus_df = pd.DataFrame({'pitch': ['Eephus'],'pitch_code': ['EP'],
                               'const': [ols_result.params[0]],
                               'means': [ols_result.params[1]], 
                               'absmeans': [abs(ols_result.params[1])], 
                               'se_mean': [[ols_result.bse[0], ols_result.bse[1]]],
                               'se_mean_lo': [ols_result.params[1] - ols_result.bse[0]],
                               'se_mean_hi': [ols_result.params[1] + ols_result.bse[1]],
                                'error_at_mid': [ols_result.params[0] + ols_result.params[1] * (0)]})

# Join DFs
results = pd.concat([results_by_pitch.sort_values("absmeans",ascending=False),knuckleball_df,eephus_df],axis=0)
results = results[results.pitch != "All"]

# Add labels
sns.despine(offset=10, trim=True,ax=ax1)
ax1.set_ylabel("Vertical plate position (% strike zone)",fontweight='bold')
ax1.set_xlabel("Vertical contact error (% strike zone)",fontweight='bold')
ax1.set_title("Contact Error vs Position in Strike Zone", fontsize=11, fontweight='bold')
ax1.set_xlim(-.06, .175)
ax1.set_ylim(extended_xvals[0],extended_xvals[-1:])

# Add legend
ax1.legend(loc="upper right",frameon=False,fontsize=9,ncol=3, bbox_to_anchor=(2.5,1.25))

# Plot slope results
plt.sca(ax2)
results.plot(y="absmeans", x="pitch", kind="bar", yerr=np.array(results.se_mean.tolist()).T,legend=None,ax=ax2,color=list(these_colors + define_colors))
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.xaxis.set_visible(False)
ax2.set_xlabel(None)
ax2.set_ylabel("Slope",fontweight="bold")
ax2.set_yticks([0.0, 0.01, 0.02])
ax2.set_ylim(0,0.025)

# Plot bias results
plt.sca(ax3)
results.plot(y="const", x="pitch", kind="bar", yerr=np.array(results.se_mean.tolist()).T,legend=None,ax=ax3,color=list(these_colors + define_colors))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.set_yticks([0.0, 0.01, 0.02])
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax3.set_xlabel(None)
ax3.set_ylabel("Bias",fontweight="bold")
ax3.tick_params(axis="x",length=0)
ax3.set_ylim(0,0.025)

# Save the panel for figure 4D
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure4","Figure4D.svg"))
