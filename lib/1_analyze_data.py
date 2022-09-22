
# Define baseball radius and optimal launch angle
baseball_radius = 2.9/2
optimal_launch_angle = 25

# Dates for gathering data
start_date = '2015-04-01'
end_date   = '2021-11-01'

# Get data
if not load_data:
    # Get data using pybaseball
    try :
        data = pybaseball.statcast(start_dt=start_date,end_dt=end_date,parallel=True)
    except:
        data = pybaseball.statcast(start_dt=start_date,end_dt=end_date,parallel=False)
    first_total = len(data.index)
    print("Total number of pitches in raw data: ", first_total)
    
    # Sort by date and reset the index
    temp = data.sort_values(by="game_date")
    data = temp.reset_index(drop=True)

    # Remove rows from exhibition and spring training
    data = data[~data['game_type'].isin(['E','S'])]

    # Dump file for faster access later
    data.to_pickle("/data/statcast_data_2015-2021_raw.pkl")
else:    
    # Read raw data
    print("Loading data...",end="")
    data = pd.read_pickle(data_path)
total_pitches_raw = len(data.index)
print("done.")

print("Cleaning up dataframe...",end="")
# Add year as column
data['year']= pd.DatetimeIndex(data['game_date']).year

# Change knucklecurve to curve since its the same pitch for this analysis
data["pitch_name"].replace({"Knuckle Curve": "Curveball"},inplace=True)

# Sort data in order of date --> game --> inning --> at-bat # in game --> pitcher number in at-bat
data = data.sort_values(by=["game_date","game_pk","inning","at_bat_number","pitch_number"])

# Group data and add at-bat number in inning
grouped = data.groupby(["game_pk","inning","inning_topbot"])
data["at_bat_number_inning"] = grouped.at_bat_number.transform(lambda x: x - x.iloc[0])

#########################################################################
#                                                                       #
#                      Calculate and bin plate_z                        #
#                                                                       #
#########################################################################
# Clean up plate z - make vals < 0 --> nan and remove
data.loc[:,'plate_z'] = data.loc[:,'plate_z'].mask(data.loc[:,'plate_z']<0)
data.dropna(subset=['plate_z'],inplace=True)

# Convert plate_z crossing to % of strikezone
data.loc[:,'plate_z_norm'] = np.array((data.loc[:,'plate_z']-data.loc[:,'sz_bot']) / (data.loc[:,'sz_top'] - data.loc[:,'sz_bot']),dtype="float")

# Bin plate z percent data - clean up first to remove outliers
data = data[(data.loc[:,"plate_z_norm"] >= np.percentile(data.loc[:,"plate_z_norm"],0.1)) & (data.loc[:,"plate_z_norm"] <= np.percentile(data.loc[:,"plate_z_norm"],99.9))]
data = data.dropna(subset=['plate_z_norm'])

# Define bin labels and bin by # of percetiles
bin_labels = list(map(str, range(1,10))) 
plate_z_norm_qbinned , plate_z_norm_qbins = pd.qcut(data['plate_z_norm'].astype("float"),q=len(bin_labels), labels=bin_labels,retbins=True)
data = pd.concat([data, plate_z_norm_qbinned.rename("plate_z_norm_qbinned")],axis=1)

# Compute mean of bin from bin edges
plate_z_norm_qbins_mean = calc_bin_means(plate_z_norm_qbins)
data.loc[:,'plate_z_norm_qbinned_percentile'] = data['plate_z_norm_qbinned'].astype(int).transform(lambda x: plate_z_norm_qbins[int(x)-1])
data.loc[:,'plate_z_norm_qbinned_percentile_mean'] = data['plate_z_norm_qbinned'].astype(int).transform(lambda x: plate_z_norm_qbins_mean[int(x)-1])

# Spin induced movement
data.loc[:,"pfx_z"] = data.loc[:,"pfx_z"].astype("float")
data.loc[:,"pfx_x"] = data.loc[:,"pfx_x"].astype("float")
data.loc[:,"pfx_z_norm"] = np.array((data.loc[:,'pfx_z'])/(data.loc[:,'sz_top'] - data.loc[:,'sz_bot']),dtype="float")

#########################################################################
#                                                                       #
#                      Calculate contact error                          #
#                                                                       #
#########################################################################

data.loc[:,'launch_angle'] = data.loc[:,'launch_angle'].astype(float)
data.dropna(subset=["launch_angle"],inplace=True)

# Calculate contact error
data.loc[:,'vertical_contact_error'] = np.array(-baseball_radius*(np.sin (-optimal_launch_angle* np.pi / 180) - np.sin(-data.loc[:,'launch_angle']* np.pi / 180)),dtype="float")
data.dropna(subset=["vertical_contact_error"],inplace=True)

# Convert contact error to % of strikezone
data.loc[:,'vertical_contact_error_norm'] = np.array(data.loc[:,'vertical_contact_error']  / (12*(data.loc[:,'sz_top'] - data.loc[:,'sz_bot'])),dtype="float") #  (baseball_radius*2)
data.dropna(subset=["vertical_contact_error_norm"],inplace=True)

# Save cleaned data
if not load_data:
    data.to_pickle("/data/statcast_data_2015-2021_cleaned.pkl")

###########################################
#                                         #
#      Which data to use for analysis     #
#                                         #
###########################################
which_data_true_shift = "plate_z_norm"
exec("%s = %s" % ("which_bins",which_data_true_shift + "_qbins") )
exec("%s = %s" % ("which_bins_mean",which_data_true_shift + "_qbins_mean") )
which_data_deviation ="vertical_contact_error_norm"
print("done.")

#########################################################################
#                                                                       #
#                          Process Pitching                             #
#                                                                       #
#########################################################################

# Create dictionary for matching pitch names with codes
print("Processing pitch data...")
pitch_type_codes = {"Sinker"         : "SI",
                    "Changeup"       : "CH",
                    "Slider"         : "SL",
                    "4-Seam Fastball": "FF",
                    "Knuckle Curve"  : "KC",
                    "Curveball"      : "CU",
                    "2-Seam Fastball": "FT",
                    "Cutter"         : "FC",
                    "Split-Finger"   : "FS",
                    "Splitter"       : "FS",
                    "Pitch Out"      : "PO",
                    "Eephus"         : "EP",
                    "Forkball"       : "FO",
                    "Knuckleball"    : "KN",
                    "Fastball"       : "FA",
                    "Screwball"      : "SC",
                    "nan"            : "UN",
                    " "              : " "
                   }

# Get all pitches
all_pitches  = data.pitch_name.value_counts()
frequent_pitches = all_pitches[all_pitches > 10000]
which_pitches = frequent_pitches.index
    
# Pre-allocate variables    
const         = []
means         = []
se_mean       = []
se_lo         = []
se_hi         = []
err_at_middle = []
pitch_codes   = []
model         = []

# Run regression. This could be done neater with a groupby and then a function, but.. next time
for pitch in tqdm(which_pitches):
    # Get pitch and associated data
    temp = data.loc[data["pitch_name"].isin([pitch]),:].copy()
    # Regression 
    ols_result = run_regression(temp[which_data_true_shift + "_qbinned_percentile"],temp[which_data_deviation])
    # Append results to lists
    model.append(ols_result)
    pitch_codes.append( list(pitch_type_codes.values())[list(pitch_type_codes.keys()).index(pitch)] )    
    const.append(ols_result.params[0])
    means.append(ols_result.params[1])
    se_mean.append([ols_result.bse[0],ols_result.bse[1]])
    se_lo.append(ols_result.params[1] - ols_result.bse[0])
    se_hi.append(ols_result.params[1] + ols_result.bse[1])
    err_at_middle.append(ols_result.params[0] + ols_result.params[1] * (0.5))
    
    
# Compute regression using statmodels.OLS for ALL PITCHES in which_pitches
temp = data.loc[data["pitch_name"].isin([pitch]),:].copy()
xx_all = np.array(temp[which_data_true_shift + "_qbinned_percentile_mean"])
yy_all = np.array(temp[which_data_deviation])
ols_result_all = run_regression(xx_all,yy_all)
prstd_ols_all, iv_l_ols_all, iv_u_ols_all = wls_prediction_std(ols_result_all) # for getting confidence intervals

# Append results of all pitches to list
pitch_codes.append("All")    
const.append(ols_result_all.params[0])
means.append(ols_result_all.params[1])
se_mean.append([ols_result_all.bse[0],ols_result_all.bse[1]])
se_lo.append(ols_result_all.params[1] - ols_result_all.bse[0])
se_hi.append(ols_result_all.params[1] + ols_result_all.bse[1])
err_at_middle.append(ols_result_all.params[0] + ols_result_all.params[1] * (0.5))
model.append(ols_result_all)

# Add to dataframe for easy manipulation and plotting
pitch_list = which_pitches.values.tolist()
pitch_list.append('All')
results_by_pitch = pd.DataFrame({'pitch': pitch_list,
                                 'pitch_code': pitch_codes,
                                 'const': const,
                                 'means': means,
                                 'absmeans': [-1*ii for ii in means],
                                 'se_mean': se_mean,
                                 'se_mean_lo': se_lo,
                                 'se_mean_hi': se_hi,
                                 'error_at_mid': err_at_middle,
                                 'model':model})
print("Done.")

########################################################################
#                                                                      #
#                       GET DATA BY PITCHER                            #
#                                                                      #
########################################################################
# For each pitch type per pitcher estimate the variance in vertical axis. 
# Consider how slope/bias relates to width of distribution, i.e., uncertainty
min_pitches = 200

# Create arrays to store data for adding to dataframe
num_pitches   = []
pitcher_name  = []
pitcher_num   = []
pitch_name    = []
pitch_code    = []
year          = []
ols_const     = []
ols_slope     = []
ols_se_mean   = []
ols_se_lo     = []
ols_se_hi     = []
err_at_middle = []
err_at_bottom = []
err_at_top    = []
pfx_z_mean    = []
pfx_z_med     = []
pfx_z_std     = []
error_mean    = []
error_med     = []
error_std     = []
error_max     = []
pitch_mean    = []
pitch_med     = []
pitch_std     = []

# Group data by pitcher - each year they pitched will be mixed up

group_by_pitcher = data.groupby('pitcher')
print("Performing regression for each pitcher ... ")
# for each of the pitchers
for this_pitcher in tqdm(group_by_pitcher.pitcher.unique().index.tolist()):
        # Group by the pitcher
        group_by_pitch = group_by_pitcher.get_group(this_pitcher).groupby('pitch_name')

        # for each pitch type
        for this_pitch in group_by_pitch.pitch_name.unique().index.tolist():
            local_data = group_by_pitch.get_group(this_pitch)
            num_pitches.append(len(local_data))
            if len(local_data) > min_pitches:
                # Store pitcher info
                pitcher_num.append(this_pitcher)
                pitcher_name.append(local_data.player_name.unique()[0])
                pitch_name.append(this_pitch)
                pitch_code.append(local_data.pitch_type.unique()[0])
                # year.append(this_year)
                pfx_z_mean.append(local_data.pfx_z_norm.mean())
                pfx_z_med.append(local_data.pfx_z_norm.median())
                pfx_z_std.append(local_data.pfx_z_norm.std())

                # Fit data
                # ORIGINAL:
                # xx = np.array(local_data[which_data_true_shift])#_qbinned_percent_mean
                # yy = np.array(local_data[which_data_deviation])
                # For inverted axis:
                xx = np.array(local_data[which_data_true_shift])#_qbinned_percent_mean
                yy = np.array(local_data[which_data_deviation])
                ols_result = run_regression(xx,yy)
                
                # Append fit data to lists
                ols_const.append(ols_result.params[0])
                ols_slope.append(ols_result.params[1])
                ols_se_mean.append([ols_result.bse[0], ols_result.bse[1]])
                ols_se_lo.append(ols_result.params[1] - ols_result.bse[0])
                ols_se_hi.append(ols_result.params[1] + ols_result.bse[1])
                err_at_middle.append(ols_result.params[0] + ols_result.params[1] * (0.5))
                err_at_bottom.append(ols_result.params[0] + ols_result.params[1] * (0.0))
                err_at_top.append(ols_result.params[0] + ols_result.params[1] * (1))
                
                # Get stats for pitch location and spread
                pitch_mean.append(local_data[which_data_true_shift].mean())
                pitch_med.append(local_data[which_data_true_shift].median())
                pitch_std.append(local_data[which_data_true_shift].std())
                
                # Get stats for error
                error_mean.append(local_data[which_data_deviation].mean())
                error_med.append(local_data[which_data_deviation].median())
                error_std.append(local_data[which_data_deviation].std())
                error_max.append(local_data[which_data_deviation].max())
                
data_by_pitcher = pd.DataFrame({"pitcher": pitcher_num,
                                "player_name": pitcher_name,
                                "pitch_name": pitch_name,
                                "pitch_type": pitch_code,
                                #"year": year,
                                "pfx_z_mean": np.array(pfx_z_mean),
                                "pfx_z_med": np.array(pfx_z_med),
                                "pfx_z_std": np.array(pfx_z_std),
                                "ols_const": np.array(ols_const),
                                "ols_slope": np.array(ols_slope),
                                "ols_slope_abs": -np.array(ols_slope),
                                "ols_se_mean": ols_se_mean,
                                "ols_se_lo": np.array(ols_se_lo),
                                "ols_se_hi": np.array(ols_se_hi),
                                "err_at_middle": np.array(err_at_middle),
                                "err_at_bottom": np.array(err_at_bottom),
                                "err_at_top": np.array(err_at_top),
                                "pitch_mean": np.array(pitch_mean),
                                "pitch_med": np.array(pitch_med),
                                "pitch_std": np.array(pitch_std),
                                "error_mean": np.array(error_mean),
                                "error_med": np.array(error_med),
                                "error_std": np.array(error_std),
                                "error_max": np.array(error_max)})

data_by_pitcher_trim = data_by_pitcher[data_by_pitcher.pitch_name.isin(which_pitches)].copy()
