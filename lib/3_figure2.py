########################################################################
#                                                                      #
#                             MAKE FIGURE 2                            #
#                                                                      #
########################################################################


from tkinter import Y

from numpy import clip


regression_color = "dodgerblue"

cmap =  "viridis"#"plasma"#"crest"#"colorblind"
these_colors = []
for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
    these_colors.append([*row,1.])
    
markers = ["X","^","*","s","o","P","D"]#,"v"]

# Get bin tick values based on .qbin()
bin_ticks = []
for ii in range(len(which_bins)):
    bin_ticks.append(np.mean(which_bins[ii:(ii+1)]))
bin_ticks_labels = list(map(str, [round(val,1) for val in bin_ticks]))
bin_ticks.extend([0,1])
bin_ticks_labels.extend(['0','1'])

# Define figures using gridspec
fig = plt.figure(constrained_layout=False)
fig.set_size_inches(7.5,8)
plt.rcParams.update({'font.sans-serif':'Arial','font.size':10})

###########################################
#                                         #
#   Define figure layout using gridspec   #
#                                         #
###########################################

# Add first gridspec for left and middle axes
gs1 = fig.add_gridspec(nrows=1, ncols=2, left=-0.001, right=0.325, wspace=.55,width_ratios=[1,.35],top=.99,bottom=0.55)
ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs1[0, 1])

# Add second gridspec for right axis
gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.55, right=0.999,top=0.985,bottom=0.55)#
ax3 = fig.add_subplot(gs2[0,0]) 

# gs3 = fig.add_gridspec(nrows=1, ncols=1, left=0.58, right=0.985,bottom=0.8)
# ax4 = fig.add_subplot(gs3[0,0])

gs4 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.35,bottom=0.025,top=0.425)
ax5 = fig.add_subplot(gs4[0,0])

gs5 = fig.add_gridspec(nrows=1, ncols=1, left=0.5, right=0.99,bottom=0.3,top=0.45)
ax6 = fig.add_subplot(gs5[0,0])

gs6 = fig.add_gridspec(nrows=1, ncols=1, left=0.5, right=0.99,bottom=0.05,top=0.225)
ax7 = fig.add_subplot(gs6[0,0])

##############################
#                            #       
#        Axis 1 [0]          #
#                            #
##############################

# Show batter image
img = mpimg.imread('img/batter.png')
ax1.imshow(img)
ax1.axis('off') 
sns.despine()
ax1.set_xlim(120,1100)
ax1.set_ylim(2500,90)

##############################
#                            #       
#        Axis 2 [1]          #
#                            #
##############################
plot_kde_only = False

if plot_kde_only:
    sns.kdeplot(data=data, y="plate_z_norm",ax=ax2,fill=False,color="k")
    xpos, ypos_chest, ypos_knees = .75, .9875, -.02
else:
    sns.histplot(data,y="plate_z_norm",bins=35,ax=ax2,color='silver',stat='density',kde=True,shrink=0.9,edgecolor="w", linewidth=1.5,zorder=2)
    sns.kdeplot(data=data[data.pitch_name.isin(which_pitches)], y="plate_z_norm",ax=ax2,fill=False,color="dodgerblue",zorder=2)#color="darkgrey")
    xpos, ypos_chest, ypos_knees = 1, 1.05, -.1

# Add lines to label strikezone
ax2.axhline(y = 1, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
ax2.axhline(y = 0, xmin=-.01, xmax=0.85,color='firebrick',linestyle=':',clip_on=False,zorder=2,lw=2)
# txt1 = ax2.text(xpos,ypos_chest,"Chest",color='firebrick',fontweight="bold",zorder=2)
# txt2 = ax2.text(xpos,ypos_knees,"Knees",color='firebrick',fontweight="bold",zorder=2)

# Make figure adjustments
ax2.set_xlim(0,1.25)
ax2.set_ylim(-.7,1.7)
ax2.set_yticks(ticks=[0,.5,1])
ax2.set_yticklabels(['0 %','50 %','100 %'])#bin_ticks_labels)#['1','2','3','4','5','6','7','8'])
# ax2.tick_params(axis='y')#,direction='out')
ax2.xaxis.set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_ylabel(None)
ax2.text(-1.35,0.45,"% Strikezone",fontweight='bold',va="center",rotation="vertical")
ax2.text(0,-.95,"Home plate",fontsize=10,fontweight="bold",ha="center")
ax2.set_title("Vertical Pitch Position \n for all 1 million pitches", fontsize=10, fontweight='bold',y=1.02,x=-1)

##############################
#                            #       
#        Axis 3 [2]          #
#                            #
##############################

plt.sca(ax3)
xvals = np.unique(xx_all)
yvals = results_by_pitch[results_by_pitch.pitch.isin(["All"])].const.values[0] + (results_by_pitch[results_by_pitch.pitch.isin(["All"])].means.values[0] * xvals)
extended_xvals = np.array([-.65,*xvals,1.65])
extended_yvals = results_by_pitch[results_by_pitch.pitch.isin(["All"])].const.values[0] + (results_by_pitch[results_by_pitch.pitch.isin(["All"])].means.values[0] * extended_xvals)

# Draw lines showing slope=0 and slope~=1
# likelihood_line = ax3.plot(len(extended_xvals)*[results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0]],extended_xvals,color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
likelihood_line = ax3.axvline(results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0],color="grey",linewidth=2.5,zorder=1,label="Mostly likelihood")
prior_bias = results_by_pitch.loc[results_by_pitch.pitch=="All","error_at_mid"].iloc[0] - (-0.075*0.5)
prior_line = ax3.plot((extended_xvals*-0.075 + prior_bias),extended_xvals,linestyle=(0,(1,1)),color="dodgerblue",linewidth=3,zorder=1,label="Mostly Prior")


# Make violin
my_violinplot(local_data=data,
              true_shift=which_data_true_shift,
              deviation=which_data_deviation,
              ax=ax3,
              which_pitches=which_pitches,
              bin_locs=which_bins_mean)




# Plot regression line #np.array((0, 158, 115))/256
# ) # blue: 0,114,178 ; green = 0, 158, 115 ; pink = 204,121,167

# ax3 = plot_regression_with_CI_flipped(ax3,results_by_pitch[results_by_pitch.pitch.isin(["All"])]["model"].item(),extended_xvals,regression_color)

reg = ax3.plot(extended_yvals,extended_xvals,lw=2.5,color="black",label="All Pitches")#,zorder=3)

# ax3.axhline(0,color='firebrick',linestyle='--')
# ax3.axhline(1,color='firebrick',linestyle='--')

# Add labels
ax3.set_ylim([-.5,1.5])
# plt.gca().invert_xaxis()
ax3.set_yticks([which_bins_mean[0]-(-0.1*which_bins_mean[0]),0,0.5,1,which_bins_mean[-1]+(0.1*which_bins_mean[-1])])
ax3.set_xlim([-.05,0.15])
ax3.set_ylabel("Vertical plate position (% of strike zone)",fontweight='bold')
ax3.set_xlabel("Vertical contact error (% of strike zone)",fontweight='bold')#,y=-.05)
ax3.set_title("Contact error vs Position in Strike Zone", fontsize=11, fontweight='bold')#,y=1.05)
# ax3.xaxis.set_label_coords(-.175,.5)
ax3.spines['top'].set_visible(False)#True)
ax3.spines['right'].set_visible(False)#True)
# ax3.spines['top'].set_visible(True)
# ax3.spines['right'].set_visible(True)
xticksvals = [-.05, 0, 0.05, 0.1,.15]
ax3.set_xticks(xticksvals)
ax3.set_xticklabels(list(map(str,xticksvals)))
ax3.legend(loc=(.5,.7),frameon=False)
sns.despine(offset=10, trim=True,ax=ax3)
ax3.set_yticklabels([" ","Knees (0 %)", "Belt (50 %)", "Chest (100 %)"," "])
ax3.get_yticklabels()[1].set_color("firebrick")
ax3.get_yticklabels()[2].set_color("firebrick")
ax3.get_yticklabels()[3].set_color("firebrick")
ax3.get_yticklabels()[1].set_weight("bold")
ax3.get_yticklabels()[2].set_weight("bold")
ax3.get_yticklabels()[3].set_weight("bold")

##############################
#                            #       
#          Axis 4            #
#                            #
##############################

# Define colors and markers
regression_color = "dodgerblue"

# cmap =  "viridis"#"plasma"#"crest"#"colorblind"
# these_colors = []
# for row in sns.color_palette(cmap,as_cmap=False,n_colors=len(which_pitches)): #frequent_pitches.index)):
#     these_colors.append([*row,1.])
    
from matplotlib.colors import LinearSegmentedColormap
# colors = [(30,144,255), (128,128,128)]
clrs = [(0.1171875, 0.5625, 0.99609375),(0.5,0.5,0.5)]
cm = LinearSegmentedColormap.from_list("mybluegrey", clrs, N=len(which_pitches))
bluegrey = [cm(i) for i in range(cm.N)]


markers = ["X","^","*","s","o","P","D"]#,"v"]
    
# Make bottom left panel 
plt.sca(ax5)    
sns.kdeplot(data=data_by_pitcher_trim,y="ols_slope_abs",x="ols_const",color="lightgrey",ax=ax5,fill=False,levels=5)
sns.regplot(data=data_by_pitcher_trim,y="ols_slope_abs",x="ols_const",color="black",ax=ax5,scatter=False)
scat = sns.scatterplot(data=results_by_pitch[results_by_pitch.pitch.isin(which_pitches)],y="absmeans",x="const",hue="pitch",style="pitch",s=100,ax=ax5,palette=bluegrey,hue_order=which_pitches,legend=False,markers=markers,zorder=3)#palette="crest")
ax5.spines['top'].set_visible(True)
ax5.spines['right'].set_visible(True)
ax5.set_xlabel("Bias",fontweight='bold')
ax5.set_ylabel("| Slope |",fontweight='bold')
ax5.set_title("| Slope | vs Bias \n Across All Pitch Types", fontsize=11, fontweight='bold')#,y=1.05)
ax5.set_ylim(0,.05)

sort_order = results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].sort_values("means",ascending=True).pitch.tolist()

plt.sca(ax6) 
results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].sort_values("means",ascending=True).plot(y="absmeans", x="pitch", kind="bar",color=bluegrey, yerr=np.array(results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].se_mean.tolist()).T,legend=None,ax=ax6)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.xaxis.set_visible(False)
ax6.set_xlabel(None)
ax6.set_ylabel("| Slope |",fontweight="bold")

# Plot slope and bias bar plots
plt.sca(ax7)
results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].sort_values("means",ascending=True).plot(y="error_at_mid", x="pitch", kind="bar", color=bluegrey, yerr=np.array(results_by_pitch[results_by_pitch.pitch.isin(which_pitches)].se_mean.tolist()).T,legend=None,ax=ax7,clip_on=False)
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['bottom'].set_visible(False)
plt.setp(ax7.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax7.set_xlabel(None)
ax7.set_ylabel("Bias",fontweight="bold")
for cnt,marker in enumerate(markers):
    ax7.scatter(x=cnt,y=-0.005,s=100,marker=marker,color=bluegrey[cnt])
ax7.tick_params(axis="x",length=0)
ax7.set_yticks([0,.01,.02,0.03])
ax7.set_ylim(-.01,0.02)

# Add patch to cover bottom 
from matplotlib.patches import Rectangle
ax7.add_patch(Rectangle((-0.55,-0.001),width=0.25,height=-0.04, zorder=3,facecolor="white"))

# Add labels
ax2.text(-5.5,1.75,"A.",fontweight="bold",fontsize=14)
ax2.text(3.,1.75,"B.",fontweight="bold",fontsize=14)
ax2.text(-5.5,-1.25,"C.",fontweight="bold",fontsize=14)
ax2.text(3.,-1.25,"D.",fontweight="bold",fontsize=14)
ax2.text(3.,-2.35,"E.",fontweight="bold",fontsize=14)
ax2.plot((-1,1),(-0.8,-0.8),'k-',linewidth=2.5,clip_on=False)
ax2.add_patch(Rectangle((-1,0),width=2,height=1, zorder=1,fill=False,facecolor=None,edgecolor="lightgrey",clip_on=False))

ptext = ax2.text(4.65,-2.315,"More Prior",fontweight="bold",fontsize=10,color=bluegrey[0])
ltext = ax2.text(11.25,-2.315,"More Likelihood",fontweight="bold",fontsize=10,color=(0.2,0.2,0.2))#,ha="right")
ax2.arrow(10.85, -2.29,-4.1,0, head_width=.075, head_length=.35, linewidth=.75, color=bluegrey[0], length_includes_head=True,clip_on=False)
ax2.arrow(6.75, -2.29,4.1, 0, head_width=.075, head_length=.35, linewidth=.75, color=bluegrey[-1], length_includes_head=True,clip_on=False)


# Save figure
plt.show()
fig.savefig(os.path.join(os.getcwd(),"figures","figure_parts","figure2","Figure2.svg"))
