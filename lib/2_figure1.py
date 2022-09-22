########################################################################
#                                                                      #
#                    MAKE SUBPLOTS FOR FIGURE 1                        #
#                                                                      #
########################################################################

# The figure is made in inkscape, but the data figures are made here.

# Pitches and order
pitches = ["Sinker","Cutter","4-Seam Fastball","Slider","Curveball","Changeup"]    
markers = ["X","o","*","s","^","P"]
cmap =  "colorblind"
greinke_colors = []
for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(pitches)): #frequent_pitches.index)):
    greinke_colors.append([*row,1.])    
order = pd.DataFrame({"pitch": [], "colormap": [], "color": [], "marker": []})
for (pitch,mrkr,this_clr) in zip(pitches,markers,greinke_colors):   
    order = pd.concat([order,pd.DataFrame({"pitch": pitch, "color": [this_clr],"marker": mrkr, "code": pitch_type_codes[pitch]})],ignore_index=True, axis=0)

order.reset_index(inplace=True)

# Get data for Zack Greinke
greinke = data[data.player_name.isin(["Greinke, Zack"])].copy()

# Convert to float
greinke.loc[:,"release_speed_float"] = greinke.release_speed.astype("float64")
greinke.loc[:,"release_spin_float"] = greinke.release_spin_rate.astype("float64")
greinke.loc[:,"release_pos_x_float"] = greinke.release_pos_x.astype("float64")
greinke.loc[:,"release_pos_z_float"] = greinke.release_pos_z.astype("float64")
greinke.loc[:,"pfx_z_float"] = greinke.pfx_z.astype("float64")
greinke.loc[:,"plate_z_float"] = greinke.plate_z.astype("float64")
greinke.loc[:,"plate_z_norm"] = np.array((greinke.plate_z - greinke.sz_bot) / (greinke.sz_top - greinke.sz_bot), dtype="float")

# Save a copy
original = greinke.copy()

#########################################
#                                       #
#    Figure 1A - Prior and Likelihood   #
#                                       #
#########################################

# Get pdf for greinke data - aka prior
mean,std = stats.norm.fit(greinke.loc[greinke.pitch_name.isin(["4-Seam Fastball"]),"plate_z_norm"])
X= np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(X, mean, std)
greinke_prior /= np.sum(greinke_prior)

# Get likelihood
likelihood = stats.norm.pdf(X, 0.05, 0.2)
likelihood /= np.sum(likelihood)

# Make figure
fig = plt.figure(figsize=(1.5,4))
ax = fig.add_subplot()

# Plot prior and likelihood
ax.plot(greinke_prior,X,color="dodgerblue",zorder=2,label="Prior") #order_greinke.loc[order_greinke.pitch.isin(["4-Seam Fastball"]),"color"].tolist()[0]
ax.plot(likelihood,X,color=(0.5,0.5,0.5),zorder=3,label="Likelihood")

# Make figure adjustments
ax.set_ylim(-.5,1.5)
bin_ticks=[-.5,0,0.25,0.5,0.75,1,1.5]
ax.set_yticks([0,1])
ax.set_yticklabels([])
ax.tick_params(axis='y')
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure1","Figure1A.svg"))


#########################################
#                                       #
#               Figure 1B               #
#                                       #
#########################################

# Define figure
fig = plt.figure(figsize=(7.5,1.85))

# Add gridspec for legend
ncols=6
gs = fig.add_gridspec(nrows=1, ncols=ncols, left=0.001, right=1, bottom=0.001,top=1,width_ratios=[1.25,1,1,1,1,1.5])
ax = []
for ii in range(ncols):
    ax.append(fig.add_subplot(gs[ii]))

# ------------------------------------ #
#                                      #
#      Figure 1B-1 : P(pitch type)     #
#                                      #
# ------------------------------------ #

sns.histplot(ax=ax[0],data=greinke[greinke.pitch_name.isin(pitches)],y="player_name",hue="pitch_name",hue_order=order.pitch.tolist(),palette=order.color.tolist(), multiple="dodge", 
              stat='density', shrink=0.8, common_norm=True)

ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].xaxis.set_visible(False)
ax[0].set_xlabel(None)
ax[0].set(xlabel=None,ylabel=None)

# Clean up func for middle parts of figure 2 B
def figure_1B_cleanup(ax):
    ax.set_ylim(-.5,1.5)
    ax.set_yticks([0,1])
    ax.set_yticklabels([])
    ax.tick_params(axis='y')
    ax.set_ylabel(None)
    ax.xaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

# ------------------------------------ #
#                                      #
#    Figure 1B-2 : P(z | pitch type)   #
#                                      #
# ------------------------------------ #

for pitch in pitches:
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pitch_name.isin([pitch]),"plate_z_norm"])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.pitch_name.isin([pitch]),:])
    ax[1].plot(prior,X,color=order.loc[order.pitch.isin([pitch]),"color"].tolist()[0],label="Prior")

# Make figure adjustments
ax[1] = figure_1B_cleanup(ax[1])

# ------------------------------------ #
#                                      #
#    Figure 1B-3 : P(z | spin axis)    #
#                                      #
# ------------------------------------ #

greinke.dropna(subset=['spin_axis'],inplace=True)

spin_cutbins = [0,150,greinke.spin_axis.max()]
greinke.loc[:,'spin_axis_binned'] = pd.cut(greinke['spin_axis'].astype("float"),bins=spin_cutbins)

clrs = ['black','darkgrey','grey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'spin_axis_binned'].unique(),['black','grey'],['-',':'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.spin_axis_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.spin_axis_binned == bin,'plate_z_norm'])
    ax[2].plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} degrees".format(spin_cutbins[cnt],spin_cutbins[cnt+1]))

# Make figure adjustments
ax[2] = figure_1B_cleanup(ax[2])
ax[2].legend(frameon=False)

# ------------------------------------ #
#                                      #
#     Figure 1B-4 : P(z | velocity)    #
#                                      #
# ------------------------------------ #

greinke = original.copy()
greinke.dropna(subset=['release_speed'],inplace=True)
cut_bins = [greinke.release_speed.min(),80,greinke.release_speed.max()]
greinke.loc[:,'release_speed_binned'] = pd.cut(greinke['release_speed'].astype("float"),bins=cut_bins)

clrs = ['black','darkgrey','grey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'release_speed_binned'].unique(),['black','grey'],['-','--'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.release_speed_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std)
    prior /= len(greinke.loc[greinke.release_speed_binned == bin,'plate_z_norm'])
    ax[3].plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} mph".format(cut_bins[cnt],cut_bins[cnt+1]))

# Make figure adjustments
ax[3] = figure_1B_cleanup(ax[3])
ax[3].legend()

# ------------------------------------ #
#                                      #
#     Figure 1B-5 : P(z | movement)    #
#                                      #
# ------------------------------------ #

greinke = original.copy()
greinke.dropna(subset=['pfx_z_norm'],inplace=True)

pfxz_cutbins = [-1,-0.25,0.5,1.5]#[greinke['pfx_z_float'].min(), 0 , greinke['pfx_z_float'].max()]
greinke.loc[:,'pfx_z_binned'] = pd.cut(greinke['pfx_z_norm'].astype("float"),bins=pfxz_cutbins)

clrs = ['black','darkgrey','lightgrey']
for cnt,(bin,clr,lnstyle) in enumerate(zip(greinke.loc[:,'pfx_z_binned'].unique(),clrs,['-',':','--'])):
    # Get pdf for greinke data - aka prior
    mean,std=stats.norm.fit(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
    X= np.linspace(-.5, 1.5, 100)
    prior = stats.norm.pdf(X, mean, std) / len(greinke.loc[greinke.pfx_z_binned == bin,'plate_z_norm'])
    ax[4].plot(prior,X,color=clr,linestyle=lnstyle,label="{}-{} inches".format(pfxz_cutbins[cnt],pfxz_cutbins[cnt+1]))
    
# Make figure adjustments
ax[4] = figure_1B_cleanup(ax[4])
ax[4].legend()

# ------------------------------------ #
#                                      #
#   Figure 1B-6 : P(launch angle, z)   #
#                                      #
# ------------------------------------ #
greinke = original.copy()
# For each bin, plot as jittered scatter plot
for (bin, bin_loc) in zip(sorted(greinke.plate_z_norm_qbinned.unique()),which_bins_mean):
    _yvals = greinke.loc[greinke.pitch_name.isin(pitches) & (greinke.plate_z_norm_qbinned==bin),:].launch_angle
    xwithnoise = len(_yvals)*[bin_loc] + np.random.normal(0,.001,len(_yvals))
    ax[5].scatter(_yvals,xwithnoise,s=5,color="lightgray")

# Regression line for all pitches_greinke
ols_result = run_regression(greinke.loc[greinke.pitch_name.isin(pitches)].plate_z_norm,greinke.loc[greinke.pitch_name.isin(pitches)].launch_angle)
thisx=np.arange(-1,2,0.01)
MAPcolor = order[order.pitch.isin(["4-Seam Fastball"])].color.tolist()[0]
ax[5].plot(ols_result.params[0] + (ols_result.params[1]*thisx),thisx,linewidth=2, color=MAPcolor,label="All pitches_greinke")

# Clean up
ax[5].set_xticks([-100,-50,0,50,100])
ax[5].set_yticks([0,.5,1])
ax[5].set_yticklabels(["Knees", "Belt","Chest"],color="firebrick",fontweight="bold",rotation=90,va="center")
ax[5].set_ylim(-.75, 1.75)
ax[5].spines['top'].set_visible(False)
ax[5].spines['right'].set_visible(False)
plt.rcParams.update({'font.sans-serif':'Arial','font.size':8})
ax[5].set_ylabel("Vertical Plate Position (% strike zone)",fontweight="bold",fontsize=8)
ax[5].set_xlabel("Launch Angle (deg)",fontweight="bold",fontsize=8)


# Save the full figure 1B
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure1","Figure1B.svg"))


#########################################
#                                       #
#               Figure 1C               #
#                                       #
#########################################

fig = plt.figure(figsize=(5,1.85))

# Add grid spec for data
nn = 3
gs1 = fig.add_gridspec(nrows=1, ncols=nn, wspace=0.15, left=0.01, right=1,bottom=0.01,top=1)# bottom=0.55,top=1)#,hspace=.25)
ax1 = []
for ii in range(nn):
    thisax = fig.add_subplot(gs1[ii])
    thisax.spines['top'].set_visible(False)
    thisax.spines['right'].set_visible(False)
    thisax.spines['bottom'].set_visible(False)
    ax1.append(thisax)

# Get pdf for greinke data
greinke = original.copy()
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
    _L = stats.norm.pdf(xx, xval, 0.2)
    _L /= np.sum(_L)
    likelihood.append(_L)
    _P = np.multiply(_L,greinke_prior)
    _P /= np.sum(_P)

    # Plot everything
    posterior.append(_P)
    ax1[cnt].plot(greinke_prior,xx,color="dodgerblue",zorder=3,lw=1.5)
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

# Save the full figure 1C
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure1","Figure1C.svg"))


#########################################
#                                       #
#               Figure 1D               #
#                                       #
#########################################
fig = plt.figure(figsize=(1.5,2))

# Add grid spec for data
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.01, right=.95,bottom=0.01,top=.95)# bottom=0.55,top=1)#,hspace=.25)
ax1 = fig.add_subplot(gs1[0])
ax1.set_ylim(-.75,2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel("Plate Position (% strike zone)",fontweight="bold",fontsize=9)
ax1.set_xlabel("Contact error (% strike zone)",fontweight="bold",fontsize=9)


# Get pdf for greinke data
mean,std=stats.norm.fit(greinke.plate_z_norm)
xx = np.linspace(-.5, 1.5, 100)
greinke_prior = stats.norm.pdf(xx, mean, std)

# Loop through positions in strike zone
xvals=np.linspace(0,1,9)
meanP=[]
allx =[]
allL =[]
for cnt,xval in enumerate(xvals):
    _L = stats.norm.pdf(xx, xval, 0.1)
    _P = np.multiply(_L,greinke_prior)
    
    _Lvals = np.random.normal(xval,0.35,100)
    _Pfit  = stats.norm.fit(_P)
    _Pvals = np.random.normal(_Pfit[0],_Pfit[1],100)
    xwithnoise = len(_Lvals)*[xval] + np.random.normal(.001,.01,len(_Lvals))
    ax1.scatter(_Lvals,xwithnoise,s=10,color="lightgray")
    meanP.append(_Lvals.mean())

# Fit data
X = sm.add_constant(xvals)# prepend=False)
ols = sm.OLS(meanP,X)
ols_result = ols.fit()

# Plot data
thisx = np.linspace(-.15,1.15, 100)
ax1.plot(len(thisx)*[0.5],thisx,color="grey",label="Mostly Likelihood")
slope = 3 # slope
intercept = 0.5 - (slope*0.5)
ax1.plot(intercept + slope *(thisx),thisx,color="black",linestyle="--",label="Mostly Prior")
ax1.plot(ols_result.params[0] + (ols_result.params[1]*thisx),thisx,linewidth=3, color=MAPcolor,label="Bayesian solution")

# Clean up
ax1.set_yticks([0,0.5,1])
ax1.set_xticks([])
ax1.set_yticklabels(["Chest (100 %)", "50 %", "Knees (0 %)"],fontweight="bold",fontsize=9)
ax1.get_yticklabels()[0].set_color("firebrick")
ax1.get_yticklabels()[2].set_color("firebrick")
ax1.legend(loc="upper center",frameon=False,ncol=2,fontsize=9,bbox_to_anchor=(.25,1.25))
ax1.set_ylim(1.15,-.15)

# Save the full figure 1D
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure1","Figure1D-1.svg"))

# ------------------------------------ #
#                                      #
#     Distribution of ball contact     #
#                                      #
# ------------------------------------ #

fig = plt.figure(figsize=(3,1))
ax = fig.add_subplot()
sns.histplot(data=data,x="vertical_contact_error",bins=10,ax=ax,color='firebrick',stat='density',kde=False,shrink=0.9,edgecolor="w", linewidth=1.5,zorder=1)
mean,std=stats.norm.fit(data.vertical_contact_error)
xx = np.linspace(-2,2, 100)
pdf = stats.norm.pdf(xx, mean, std)
ax.plot(xx,pdf,color="firebrick")
ax.fill_between(xx,pdf,where=(xx<data.vertical_contact_error.min()),color="silver")
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='x', labelrotation=90)
ax.invert_xaxis()
ax.set_xticks([])
ax.xaxis.set_visible(False)
plt.show()

# Save the full figure 1D
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure1","Figure1D-2.svg"))
