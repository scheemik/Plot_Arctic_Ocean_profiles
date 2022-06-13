"""
This is a script contains helper functions which are called to make plots by the
`make_plots.py` script

made by: Mikhail Schee (June 2022)
"""

## Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# For creating DataFrame objects
import pandas as pd
# For searching and listing directories
import os
# For matching regular expressions
import re
# For formatting date objects
import datetime
# For reading the ITP `cormat` files
import mat73
from scipy import io

"""
To install Cartopy and its dependencies, follow:
https://scitools.org.uk/cartopy/docs/latest/installing.html#installing

Relevent command:
$ conda install -c conda-forge cartopy
"""
import cartopy.crs as ccrs
import cartopy.feature

################################################################################
# This is the location of the data on your computer
science_data_file_path = '/Users/Grey/Documents/Research/Science_Data/'

################################################################################
################################################################################
# Declare plotting variables
################################################################################
################################################################################

dark_mode = True
# Enable dark mode plotting
if dark_mode:
    plt.style.use('dark_background')
    std_clr = 'w'
    clr_ocean = 'k'
    clr_land  = 'grey'
    clr_lines = 'w'
    clr_temp  = 'lightcoral'
    clr_salt  = 'silver'
else:
    std_clr = 'k'
    clr_ocean = 'w'
    clr_land  = 'grey'
    clr_lines = 'k'
    clr_temp  = 'lightcoral'
    clr_salt  = 'silver'

# Set some plotting styles
mrk_size      = 0.5
mrk_alpha     = 0.5
noise_alpha   = 0.25
lgnd_mrk_size = 60
map_mrk_size  = 7
pf_mrk_size   = 30
std_marker = '.'
map_marker = '.'
map_ln_wid = 0.5

#   Get list of standard colors
mpl_clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
#   Make list of marker styles
mpl_mrks = ['.', 'x', 'd', '*', '<', '>']
# Define array of linestyles to cycle through
l_styles = ['-', '--', '-.', ':']

#   Select particular color maps for certain types of plots
cmap_pf_no = 'plasma'
cmap_p     = 'cividis'
cmap_date  = 'viridis'
cmap_den_h = 'magma'

map_extent = 'Western_Arctic'

################################################################################
################################################################################
# Functions to load in data
################################################################################
################################################################################

def list_data_files(file_path):
    """
    Creates a list of all data files within a containing folder

    file_path       The path to the folder containing the data files
    """
    # Search the provided file path
    if os.path.isdir(file_path):
        data_files = os.listdir(file_path)
        # Remove .DS_Store directory from list
        if '.DS_Store' in data_files: data_files.remove('.DS_Store')
        # Restrict to just unique instruments (there shouldn't be any duplicates hopefully)
        data_files = np.unique(data_files)
    else:
        print(file_path, " is not a directory")
        data_files = None
    #
    return data_files

################################################################################

def load_data(plt_dict):
    """
    Find the data specified, filters, and loads them into a pandas dataframe

    plt_dict        A dictionary of parameters needed to load and filter the data
        such as:
    data_sources        A list of tuples of the data to find
                    Examples: ('AIDJEX', 'BigBear'), ('ITP', 3, 'cormat')
    filtering_types     A list of dictionaries of the filters to apply
                    Examples: [{'p_range': [260,280]}, {'p_range': [260,280], 'interpolate': 1.0}]
    """
    # Get list of sources
    data_sources = plt_dict['data_sources']
    # Get list of filters
    filters = plt_dict['filtering_types']
    # Find the relevant filters
    if len(filters) == 1:
        use_these_filters = filters[0]
    elif len(filters) > 1:
        use_these_filters = filters[i]
    else:
        use_these_filters = None
    # Check to see if there is a white_list
    #   Note: expects dictionary of dictionaries, like {source: {instrmt: [x,y,z]}}
    #     and you need to have a source/instrmt pair for all you entered in 'data_sources'
    #   Ex: {'AIDJEX': {'BigBear': ['1','3'], 'Caribou': ['5']}, 'ITP': {'2': ['1','3']}}
    if not isinstance(use_these_filters, type(None)) and 'white_list' in use_these_filters.keys():
        white_list = use_these_filters['white_list']
    else:
        white_list = None
    # Create a blank list to add each profile to
    output_list = []
    # Loop through the given sources
    for i in range(len(data_sources)):
        source = data_sources[i]
        source_type = source[0]
        instrmt = source[1]
        print('Loading data from',source_type,instrmt)
        # Set parameters based on the type of data to load
        if source_type == 'AIDJEX':
            format = ''
            file_path = science_data_file_path+'AIDJEX/AIDJEX/'+instrmt
            read_data_file = read_AIDJEX_data_file
        elif source_type == 'ITP':
            format = source[2]
            file_path = science_data_file_path+'ITPs/itp'+str(instrmt)+'/itp'+str(instrmt)+format
            read_data_file = read_ITP_data_file
        # Loop through the data files for each profile
        data_files = list_data_files(file_path)
        if isinstance(data_files, type(None)):
            print('Did not find any files')
            exit(0)
            continue
        #
        # Check to see if there is a white_list for this data source
        if not isinstance(white_list, type(None)):
            # Check source
            if source_type in white_list.keys():
                specific_white_list = white_list[source_type]
                if instrmt in specific_white_list.keys():
                    specific_white_list = specific_white_list[instrmt]
                else:
                    specific_white_list = None
            else:
                specific_white_list = None
        else:
            specific_white_list = None
        print('\t Loading',len(data_files),'files')
        for file in data_files:
            # Read in the data file for this profile
            pf_df = read_data_file(file_path, file, instrmt, format, specific_white_list)
            if not isinstance(pf_df, type(None)):
                # Apply filters (works even if filters=None)
                pf_df = filter_data(pf_df, use_these_filters)
            if not isinstance(pf_df, type(None)):
                # Remove rows of the data frame with missing data
                #   Note: only apply to temp, salt, and p because 'format' will often be
                #       set to a null value, for exmaple with AIDJEX data
                pf_df = pf_df[pf_df.temp.notnull() & pf_df.salt.notnull() & pf_df.p.notnull()]
                output_list.append(pf_df)
    # Concatenate all the profiles in the list into a dataframe
    # exit(0)
    if len(output_list) > 0:
        df = pd.concat(output_list)
        return df
    else:
        print('No profiles loaded, aborting script')
        exit(0)

################################################################################

def filter_data(data, filters):
    """
    Filters the data for one profile. Note: this assumes it is one and only one
    profile, the filtering will not work correctly if this is not the case

    data            A pandas dataframe with the following columns
        instrmt     A string of the instrument name that took the profile
        prof_no     The profile number
        format      The format of the datafile where the data came from
        temp        An array of temperature values
        salt        An array of salinity values
        p           An array of depth values (in m or dbar)
    filters             A dictionary of the filters to apply
                    Examples: {'p_range': [260,280]}, {'p_range': [260,280], 'direction': 'up'}
    """
    # Remove rows of the data frame with missing data
    #   Note: only apply to temp, salt, and p because 'format' will often be
    #       set to a null value, for exmaple with AIDJEX data
    df = data[data.temp.notnull() & data.salt.notnull() & data.p.notnull()]
    # Check for filters
    if isinstance(filters, type(None)):
        return df
    # Apply appropriate filters
    #
    # Depth range filter
    if 'p_range' in filters.keys():
        # Get endpoints of depth range
        p_max = max(filters['p_range'])
        p_min = min(filters['p_range'])
        # Filter the data frame to the specified pressure range
        df = df[(df['p'] < p_max) & (df['p'] > p_min)]
    #
    # Temperature range filter
    if 'T_range' in filters.keys():
        # Get endpoints of depth range
        T_max = max(filters['T_range'])
        T_min = min(filters['T_range'])
        # Filter the data frame to the specified pressure range
        df = df[(df['temp'] < T_max) & (df['temp'] > T_min)]
    #
    # Salinity range filter
    if 'S_range' in filters.keys():
        # Get endpoints of depth range
        S_max = max(filters['S_range'])
        S_min = min(filters['S_range'])
        # Filter the data frame to the specified pressure range
        df = df[(df['salt'] < S_max) & (df['salt'] > S_min)]
    #
    # Filter by direction of cast: up or down
    if 'cast_direction' in filters.keys():
        # Get direction of filter (either 'up' or 'down')
        #   Note: staircases are generally better resolved by up-casts
        #         because down-casts have wake issues
        direction = filters['cast_direction']
        # Add a new column for the resolution values
        df['res'] = None
        # Add first difference of depth values to data frame
        df['res'] = df['p'].diff()
        # Delete the first row because it will have nan in Resolution
        df = df.iloc[1:]
        # Filter by direction
        if direction == 'up':
            # If the first differences are negative, depth is decreasing
            #   therefore the cast is going up. Only keep such values
            df = df[df['res'] < 0]
        elif direction == 'down':
            # If the first differences are positive, depth is increasing
            #   therefore the cast is going down. Only keep such values
            df = df[df['res'] > 0]
        #
        # Make sure there are still enough points in the profile to be useful
        if len(df['p']) < 10:
            print('Not enough points in profile',np.unique(np.array(data['prof_no'])),'after filtering to',direction,'direction')
            return None
        #
        # Remove the 'res' column
        df.drop('res', axis=1, inplace=True)
        # Sort values to avoid issues with endpoints
        df = df.sort_values(by='p')
        # Add a note to remember which direction was kept
        df['notes'] = df['notes']+'-'+direction
    #
    return df

################################################################################
# Don't use the profiles specified below because they have errors
black_list = {'BigBear': [531, 535, 537, 539, 541, 543, 545, 547, 549],
              'BlueFox': [94, 308, 310],
              'Caribou': [],
              'Snowbird': [443]}

################################################################################

def read_AIDJEX_data_file(file_path, file_name, instrmt, format, white_list):
    """
    Reads certain data from an AIDJEX profile file
    Returns an array of strings

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             string of the instrument that took the data in the file
    format              Irrelevant for AIDJEX data
    white_list          Optional list of profile numbers to actually load
    """
    # Assuming file name format instrmt_YYY where the profile number,
    #   YYY, is always 3 digits
    prof_no = int(''.join(filter(str.isdigit, file_name)))
    # Check to make sure this one isn't on the black list
    if prof_no in black_list[instrmt]:
        print('Skipping',instrmt,'profile',prof_no)
        return None
    # If there's a white_list, check to see if this profile is on it
    if not isinstance(white_list, type(None)):
        if str(prof_no) not in white_list:
            return None
    # Declare variables
    lon = None
    lat = None
    # Read data file in as a pandas data object
    #   The arguments used are specific to how the AIDJEX files are formatted
    #   Reading in the file one line at a time because the number of items
    #       on each line is inconsistent between files
    dat0 = pd.read_table(file_path+'/'+file_name, header=None, nrows=1, engine='python', delim_whitespace=True).iloc[0]
    dat1 = pd.read_table(file_path+'/'+file_name, header=0, nrows=1, engine='python', delim_whitespace=True).iloc[0]
    # Extract certain data from the object, specific to how the files are formatted
    #   instrmt name
    #       Split the file name at a '/' and take the last chunk
    temp = file_name.split('/')[-1]
    #       Use a regular expression to remove all digits after splitting at a
    #       '/' and taking the second chunk
    # instrmt = re.sub(r'[0-9]', '', temp.split('_')[0])
    #   The date this profile was taken
    date_string = dat0[3]
    #   The time this profile was taken
    time_string = str(dat0[4]).zfill(4)
    #   Assuming date is in format d/MON/YYYY where d can have 1 or 2 digits,
    #       MON is the first three letters of the month, and YYYY is the year
    #   Assuming time is in format HMM where H can have 1 or 2 digits originally
    #       but is 2 digits because of zfill(4) and MM is the minutes of the hour
    try:
        date = datetime.datetime.strptime(date_string+' '+time_string, r'%d/%b/%Y %H%M')
    except:
        date = None
    #   Longitude and Latitude
    if ('Lat' in dat1[0]) or ('lat' in dat1[0]):
        lat = float(dat1[1])
    if ('Lon' in dat1[2]) or ('lon' in dat1[2]):
        lon = float(dat1[3])
    #       Check to make sure the lon and lat values are valid
    if lat == 99.9999 and lon == 99.9999:
        lat = None
        lon = None
    # Read in data from the file
    dat = pd.read_table(file_path+'/'+file_name,header=3,skipfooter=0,engine='python',delim_whitespace=True)
    # If it finds the correct column headers, put data into arrays
    if 'Depth(m)' and 'Temp(C)' and 'Sal(PPT)' in dat.columns:
        temp0 = dat['Temp(C)'][:].values
        salt0 = dat['Sal(PPT)'][:].values
        p0    = dat['Depth(m)'][:].values
        out_dict = {'source': ['AIDJEX']*len(temp0), # needs to be an array the same size as temp
                    'instrmt': [instrmt]*len(temp0), # needs to be an array the same size as temp
                    'prof_no': [str(prof_no)]*len(temp0), # needs to be an array the same size as temp
                    'lon': [lon]*len(temp0), # needs to be an array the same size as temp
                    'lat': [lat]*len(temp0), # needs to be an array the same size as temp
                    'date': [date]*len(temp0), # needs to be an array the same size as temp
                    'format': [format]*len(temp0), # needs to be an array the same size as temp
                    'notes': ['']*len(temp0), # needs to be an array the same size as temp
                    'temp': temp0,
                    'salt': salt0,
                    'p': p0
                    }
        # Build output data frame
        df = pd.DataFrame(out_dict)
        # Return all the relevant values
        return df
    else:
        return None

################################################################################

def read_ITP_data_file(file_path, file_name, instrmt, format, white_list):
    """
    Reads certain data from an ITP profile file
    Returns a pandas dataframe

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             string of the instrument that took the data in the file
    format              either 'cormat' or 'final'
    white_list          Optional list of profile numbers to actually load
    """
    # Make sure it isn't a 'sami' file instead of a 'grd' file
    if 'sami' in file_name:
        print('Skipping',file_name)
        return
    # Get just the subdirectory name, before the slash
    filename1 = file_name.split('/')[0]
    # Extract itp FloatID from filename1 assuming FloatID is the only number
    flt_id = int(''.join(filter(str.isdigit, filename1.split('_')[0])))
    # Convert that FloatID number to a string
    # instrmt = 'itp' + str(flt_id)
    instrmt = 'itp' + str(instrmt)
    # There is one ITP with ID 39_1, so need to distinguish
    if '_' in filename1:
        instrmt += 0.1
    # Get just the proper file name, after the slash
    filename2 = file_name.split('/')[-1]
    # Assuming filename format itpXgrdYYYY.dat where YYYY is always 4 digits
    #   works for `final` format above or `cormat` format corYYYY.mat
    try:
        prof_no = int(filename2[-8:-4])
    except:
        print('Skipping',instrmt,'file',filename2)
        return None
    # Check to make sure this one isn't on the black list
    if instrmt in black_list.keys():
        if prof_no in black_list[instrmt]:
            print('Skipping',instrmt,'profile',prof_no)
            return
    # If there's a white_list, check to see if this profile is on it
    if not isinstance(white_list, type(None)):
        if str(prof_no) not in white_list:
            return None
    # Load data based on itp file format
    if format == 'final':
        load_itp = load_final_itp
    elif format == 'cormat':
        load_itp = load_cormat_itp
    #
    # print('    Loading file',file_name)
    return load_itp(file_path, file_name, instrmt, prof_no)

def load_final_itp(file_path, file_name, instrmt, prof_no):
    """
    Loads the data from an ITP profile file in the `final` format
    Returns a pandas dataframe

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             The number of the ITP that took the profile measurement
    prof_no             The number identifying this specific profile
    """
    # Check instrument and profile number and skip if it's a down-cast
    #   Nevermind, `final` profiles are sorted, so there's no way to tell
    #   if it is an up or down cast
    # Read data file in as a pandas data object
    #   The arguments used are specific to how the ITP files are formatted
    #   Reading in the file one line at a time because the number of items
    #       on each line is inconsistent between files
    dat0 = pd.read_table(file_path+'/'+file_name, header=None, nrows=2, engine='python', delim_whitespace=True).iloc[1]
    # Extract certain data from the object, specific to how the files are formatted
    #   The date this profile was taken
    try:
        date = pd.to_datetime(float(dat0[1])-1, unit='D', origin=str(dat0[0]))
    except:
        date = None
    #   The latitude and longitude values where the profile was taken
    lon = float(dat0[2])
    lat = float(dat0[3])
    # Read in data from the file
    dat = pd.read_table(file_path+'/'+file_name,header=2,skipfooter=1,engine='python',delim_whitespace=True)
    # If it finds the correct column headers, put data into arrays
    if 'temperature(C)' and 'salinity' and '%pressure(dbar)' in dat.columns:
        temp0 = dat['temperature(C)'][:].values
        salt0 = dat['salinity'][:].values
        p0    = dat['%pressure(dbar)'][:].values
        out_dict = {'source': ['ITP']*len(temp0), # needs to be an array the same size as temp
                    'instrmt': [instrmt]*len(temp0), # needs to be an array the same size as temp
                    'prof_no': [str(prof_no)]*len(temp0), # needs to be an array the same size as temp
                    'lon': [lon]*len(temp0), # needs to be an array the same size as temp
                    'lat': [lat]*len(temp0), # needs to be an array the same size as temp
                    'date': [date]*len(temp0), # needs to be an array the same size as temp
                    'format': ['final']*len(temp0), # needs to be an array the same size as temp
                    'notes': ['']*len(temp0), # needs to be an array the same size as temp
                    'temp': temp0,
                    'salt': salt0,
                    'p': p0
                    }
        # Build output data frame
        df = pd.DataFrame(out_dict)
        # Return all the relevant values
        return df

def load_cormat_itp(file_path, file_name, instrmt, prof_no):
    """
    Loads the data from an ITP profile file in the `cormat` format
    Returns a pandas dataframe

    file_path           string of a file path to the containing directory
    file_name           string of the file name of a specific file
    instrmt             The number of the ITP that took the profile measurement
    prof_no             The number identifying this specific profile
    """
    # Load cormat file into dictionary with mat73
    #   (specific to version of MATLAB used to make cormat files)
    try:
        dat = mat73.loadmat(file_path+'/'+file_name)
    except:
        dat = io.loadmat(file_path+'/'+file_name)
    # print(dat)
    # exit(0)
    # Extract certain data from the object, specific to how the files are formatted
    #   The date this profile was taken, psdate: profile start or pedate: profile end
    date_MMDDYY = dat['psdate']
    #   The time this profile was taken, pstart: profile start or pstop: profile end
    time_HHMMSS = dat['pstart']
    # Sometimes, the date comes out as an array. In that case, take the first index
    if not isinstance(date_MMDDYY, str):
        date_MMDDYY = date_MMDDYY[0]
    if not isinstance(time_HHMMSS, str):
        time_HHMMSS = time_HHMMSS[0]
    try:
        date = datetime.datetime.strptime(date_MMDDYY+' '+time_HHMMSS, r'%m/%d/%y %H:%M:%S')
    except:
        date = None
    #
    # print('date:', date)
    # exit(0)
    #   The latitude and longitude values where the profile was taken
    lon = float(dat['longitude'])
    lat = float(dat['latitude'])
    # print('lon:',type(lon))
    # print('lat:',type(lat))
    # If it finds the correct column headers, put data into arrays
    if 'te_adj' in dat and 'sa_adj' in dat and 'pr_filt' in dat:
        temp0 = dat['te_adj'].flatten()
        salt0 = dat['sa_adj'].flatten()
        p0    = dat['pr_filt'].flatten()
        # print('temp0:',temp0.shape)
        # print('salt0:',salt0.shape)
        # print('p0   :',p0.shape)
        # Down-casts have an issue with the profiler wake, so only take profiles
        #   that were measured as the profiler was moving upwards
        if p0[0] < p0[-1]:
            # print('Skipping',instrmt,'profile',prof_no,'because it is a down cast')
            return None
        # else:
        #     print('prof:',prof_no,'goes from',p0[0],'to',p0[-1])
        out_dict = {'source': ['ITP']*len(temp0), # needs to be an array the same size as temp
                    'instrmt': [instrmt]*len(temp0), # needs to be an array the same size as temp
                    'prof_no': [str(prof_no)]*len(temp0), # needs to be an array the same size as temp
                    'lon': [lon]*len(temp0), # needs to be an array the same size as temp
                    'lat': [lat]*len(temp0), # needs to be an array the same size as temp
                    'date': [date]*len(temp0), # needs to be an array the same size as temp
                    'format': ['cormat']*len(temp0), # needs to be an array the same size as temp
                    'notes': ['']*len(temp0), # needs to be an array the same size as temp
                    'temp': temp0,
                    'salt': salt0,
                    'p': p0
                    }
        # Build output data frame
        df = pd.DataFrame(out_dict)
        # Return all the relevant values
        return df

################################################################################
################################################################################
# Admin functions for plotting
################################################################################
################################################################################

def make_plots(to_plot, filename=None):
    """
    Takes in a list of dictionaries, one for each subplot. Determines the needed
    arrangement of subplots, then passes one dictionary to each axis for plotting

    to_plot         A list of dictionaries, one for each subplot
                    Each dictionary contains the info to create each subplot
    """
    # Define number of rows and columns based on number of subplots
    #   key: number of subplots, value: (rows, cols, f_ratio, f_size)
    n_row_col_dict = {'1':[1,1, 0.8, 1.25], '2':[1,2, 0.5, 1.25],
                      '3':[1,3, 0.3, 1.40], '4':[2,2, 0.8, 1.50],
                      '5':[2,3, 0.5, 1.50], '6':[2,3, 0.5, 1.50]}
    # Figure out what layout of subplots to make
    n_subplots = len(to_plot)
    if n_subplots == 1:
        fig, ax = set_fig_axes([1], [1], fig_ratio=0.8, fig_size=1.25)
        xlabel, ylabel, plt_title, ax = make_plot(ax, to_plot[0], fig, 111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plt_title)
    elif n_subplots > 1 and n_subplots < 7:
        rows, cols, f_ratio, f_size = n_row_col_dict[str(n_subplots)]
        fig, axes = set_fig_axes([1]*rows, [1]*cols, fig_ratio=f_ratio, fig_size=f_size, share_y_axis=False)
        for i in range(n_subplots):
            if rows > 1:
                i_ax = (i//cols,i%cols)
            else:
                i_ax = i
            ax_pos = int(str(rows)+str(cols)+str(i+1))
            xlabel, ylabel, plt_title, ax = make_plot(axes[i_ax], to_plot[i], fig, ax_pos)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plt_title)
            # Label subplots a, b, c, ...
            ax.text(-0.1, 1.1, '('+string.ascii_lowercase[i]+')', transform=ax.transAxes, size=20)
        # Turn off unused axes
        if n_subplots < (rows*cols):
            for i in range(rows*cols-1, n_subplots-1, -1):
                axes[i//cols,i%cols].set_axis_off()
    else:
        print('Too many subplots')
        exit(0)
    #
    plt.tight_layout()
    #
    if filename != None:
        plt.savefig(filename, dpi=400)
    else:
        plt.show()

################################################################################

def make_plot(ax, plt_dict, fig, ax_pos):
    """
    Takes in a dictionary of plotting parameters and produces the plot as
    specified by those parameters. Returns the x and y labels

    ax              The axis on which to make the plot
    plt_dict        A dictionary containing the info to create this subplot
    fig             The figure in which ax is contained
    ax_pos          A tuple of the ax (rows, cols, linear number of this subplot)
    """
    # Load data into a pandas data frame and apply filters
    data = load_data(plt_dict)
    # Plot the data in the specified manner
    #   Returns the x and y labels for this axis
    xlabel, ylabel, plt_title, ax = plot_data(ax, data, plt_dict, fig, ax_pos)
    return xlabel, ylabel, plt_title, ax

################################################################################
################################################################################
# Functions to format plots
################################################################################
################################################################################

def set_fig_axes(heights, widths, fig_ratio=0.5, fig_size=1, share_x_axis=None, share_y_axis=None, prjctn=None):
    """
    Creates fig and axes objects based on desired heights and widths of subplots
    Ex: if widths=[1,5], there will be 2 columns, the 1st 1/5 the width of the 2nd

    heights      array of integers for subplot height ratios, len=rows
    widths       array of integers for subplot width  ratios, len=cols
    fig_ratio    ratio of height to width of overall figure
    fig_size     size scale factor, 1 changes nothing, 2 makes it very big
    share_x_axis bool whether the subplots should share their x axes
    share_y_axis bool whether the subplots should share their y axes
    projection   projection type for the subplots
    """
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(fig_ratio)
    # Find rows and columns of subplots
    rows = len(heights)
    cols = len(widths)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': heights,
                   'width_ratios': widths}
    # Determine whether to share x or y axes
    if share_x_axis == None and share_y_axis == None:
        if rows == 1 and cols != 1: # if only one row, share y axis
            share_x_axis = False
            share_y_axis = True
        elif rows != 1 and cols == 1: # if only one column, share x axis
            share_x_axis = True
            share_y_axis = False
        else:                       # otherwise, forget about it
            share_x_axis = False
            share_y_axis = False
    elif share_x_axis == False and share_y_axis == None:
        share_y_axis = False
        print('Set share_y_axis to', share_y_axis)
    elif share_x_axis == None and share_y_axis == False:
        share_x_axis = False
        print('Set share_x_axis to', share_x_axis)
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w*fig_size,h*fig_size), nrows=rows, ncols=cols, gridspec_kw=plot_ratios, sharex=share_x_axis, sharey=share_y_axis, subplot_kw=dict(projection=prjctn))
    # Set ticklabel format for all axes
    if (rows+cols)>2:
        for ax in axes.flatten():
            ax.ticklabel_format(style='sci', scilimits=(-3,3), useMathText=True)
    else:
        axes.ticklabel_format(style='sci', scilimits=(-3,3), useMathText=True)
    return fig, axes

################################################################################

def add_std_title(plt_dict, plt_title, data):
    """
    Adds in standard information to this subplot's title, as appropriate
    """
    plt_title += ': '
    # Find how many sources there are in the plot
    sources = data.source.unique()
    if len(sources) == 1 and len(plt_dict['data_sources']) > 3:
        plt_title += sources[0]
    else:
        # Note all the data sources in the title
        for source in plt_dict['data_sources']:
            plt_title += '-'.join(source)
            plt_title += ' '
    #
    return plt_title

################################################################################

def add_std_legend(ax, data, x_key):
    """
    Adds in standard information to this subplot's legend, as appropriate
    """
    # Add legend to report the total number of points and notes on the data
    n_pts_patch  = mpl.patches.Patch(color='none', label=str(len(data[x_key]))+' points')
    notes_string = ''.join(data.notes.unique())
    notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
    # Only add the notes_string if it contains something
    if len(notes_string) > 1:
        ax.legend(handles=[n_pts_patch, notes_patch])
    else:
        ax.legend(handles=[n_pts_patch])

################################################################################
################################################################################
# Main function for plotting
################################################################################
################################################################################

def plot_data(ax, data, plt_dict, fig, ax_pos):
    """
    Takes in arguments which determine how and what to plot on the given axis

    ax              The axis on which to make the plot
    data            A pandas dataframe of pre-filtered data
    plt_dict        A dictionary containing the info to create this subplot
    fig             The figure in which ax is contained
    ax_pos          A tuple of the ax (rows, cols, linear number of this subplot)
    """
    plot_type = plt_dict['plot_type']
    clr_map   = plt_dict['color_map']
    # Determine the x and y axes of the plot
    if plot_type == 'T-S':
        # Set the x and y axis labels
        xlabel, ylabel = r'Salinity (g/kg)', r'Temperature ($^\circ$C)'
        # Set the title
        plt_title = 'T-S'
        # Set the keys for x and y data arrays
        x_key = 'salt'
        y_key = 'temp'
    elif plot_type == 'res_vs_p':
        # Set the x and y axis labels
        xlabel, ylabel = r'Resolution (m)', r'Depth (m)'
        # Set the title
        plt_title = 'Resolution vs. Depth'
        # Sort the dataframe correctly: first by instrmt, then prof_no, then p
        #   then find the resolution (first differences in p)
        data = find_p_res(data)
        # Set the keys for x and y data arrays
        x_key = 'res'
        y_key = 'p'
    elif plot_type == 'res_hist':
        # Set the x and y axis labels
        xlabel, ylabel = r'Vertical Resolution (m)', r'Number of measurements'
        # Set the title
        plt_title = 'Resolution Histogram'
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Sort the dataframe correctly: first by instrmt, then prof_no, then p
        #   then find the resolution (first differences in p)
        data = find_p_res(data)
        res = data['res']
        # Find overall statistics
        median  = np.median(res)
        mean    = np.mean(res)
        std_dev = np.std(res)
        # Define the bins to use in the histogram, np.arange(start, stop, step)
        stop = mean + 3*std_dev
        step = stop / 50
        res_bins = np.arange(0, stop, step) #5, 0.03)
        # Plot the histogram
        ax.hist(res, bins=res_bins, color=std_clr)
        # Add legend to report overall statistics
        median_patch  = mpl.patches.Patch(color='none', label='Median:  '+'%.4f'%median)
        mean_patch    = mpl.patches.Patch(color='none', label='Mean:    ' + '%.4f'%mean)
        std_dev_patch = mpl.patches.Patch(color='none', label='Std dev: '+'%.4f'%std_dev)
        # If there are notes to add, put them in the legend
        notes_string = ''.join(data.notes.unique())
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            ax.legend(handles=[median_patch, mean_patch, std_dev_patch, notes_patch])
        else:
            ax.legend(handles=[median_patch, mean_patch, std_dev_patch])
        # Note: this option doesn't go through the colormap section because
        #   it doesn't deal with a scatter plot
        return xlabel, ylabel, plt_title, ax
    elif plot_type == 'date_hist':
        # Set the x and y axis labels
        xlabel, ylabel = r'Temporal Resolution (hours)', r'Number of measurements'
        # Set the title
        plt_title = 'Resolution Histogram'
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Sort the dataframe correctly: first by instrmt, then prof_no, then date
        #   then find the resolution (first differences in dates)
        data = find_date_res(data)
        # Make sure to convert the datetime objects to numbers for plotting
        res = data['res'].astype('timedelta64[h]')
        # Find overall statistics
        median  = np.median(res)
        mean    = np.mean(res)
        std_dev = np.std(res)
        # Define the bins to use in the histogram, np.arange(start, stop, step)
        stop = mean + 3*std_dev
        step = stop / 50
        res_bins = np.arange(0, stop, step) #5, 0.03)
        # Plot the histogram
        ax.hist(res, bins=res_bins, color=std_clr)
        # Format the numbers on the x axis
        # loc = mpl.dates.AutoDateLocator()
        # ax.xaxis.set_major_locator(loc)
        # ax.xaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
        # Add legend to report overall statistics
        median_patch  = mpl.patches.Patch(color='none', label='Median:  '+'%.4f'%median)
        mean_patch    = mpl.patches.Patch(color='none', label='Mean:    ' + '%.4f'%mean)
        std_dev_patch = mpl.patches.Patch(color='none', label='Std dev: '+'%.4f'%std_dev)
        # If there are notes to add, put them in the legend
        notes_string = ''.join(data.notes.unique())
        if len(notes_string) > 1:
            notes_patch  = mpl.patches.Patch(color='none', label=notes_string)
            ax.legend(handles=[median_patch, mean_patch, std_dev_patch, notes_patch])
        else:
            ax.legend(handles=[median_patch, mean_patch, std_dev_patch])
        # Note: this option doesn't go through the colormap section because
        #   it doesn't deal with a scatter plot
        return xlabel, ylabel, plt_title, ax
    elif plot_type == 'map':
        # Set the x and y axis labels
        xlabel, ylabel = '', ''
        # Set the title
        plt_title = 'Map of the Arctic'
        # Call a specialized function for plotting a map of the arctic
        ax, plt_title = plot_arctic_map(ax, data, plt_dict, plt_title, fig, ax_pos)
        # Note: this option doesn't go through the colormap section because
        #   it doesn't deal with a scatter plot
        return xlabel, ylabel, plt_title, ax
    elif plot_type == 'profiles':
        # Set the x and y axis labels
        xlabel, ylabel = r'Temperature ($^\circ$C)', r'Depth (m)'
        # Set the title
        plt_title = 'Profiles'
        # Call a specialized function for plotting individual profiles
        ax, plt_title = plot_profiles(ax, data, plt_dict, plt_title)
        # Note: this option doesn't go through the colormap section because
        #   it doesn't deal with a scatter plot
        return xlabel, ylabel, plt_title, ax
    else:
        # Did not provide a valid plot type
        print('Plot type',plot_type,'not valid')
        exit(0)
    #
    #
    #
    # Determine the color mapping to be used
    if clr_map == 'clr_all_same':
        # Plot every point the same color, size, and marker
        ax.scatter(data[x_key], data[y_key], color=std_clr, s=mrk_size, marker=std_marker, alpha=mrk_alpha)
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, data, x_key)
    elif clr_map == 'clr_by_source':
        # Make a list of each unique source, without sorting
        sources_to_plot = [plt_dict['data_sources'][0][0]]
        if len(plt_dict['data_sources']) > 1:
            for i in range(1, len(plt_dict['data_sources'])):
                source = plt_dict['data_sources'][i][0]
                if not source in sources_to_plot:
                    sources_to_plot.append(source)
        print('sources_to_plot',sources_to_plot)
        # Loop through each source
        i = 0
        lgnd_hndls = []
        for source in sources_to_plot:
            # Decide on the color, don't go off the end of the array
            my_clr = mpl_clrs[i%len(mpl_clrs)]
            # Get x and y data
            x_data = data[data.source == source][x_key]
            y_data = data[data.source == source][y_key]
            # Get format
            format_string = ''.join(data[data.source == source].format.unique())
            # Get notes
            notes_string = ''.join(data[data.source == source].notes.unique())
            # Get number of points
            n_pts_string = ' '+str(len(x_data))+' points'
            # Plot every point the same color, size, and marker
            ax.scatter(x_data, y_data, color=my_clr, s=mrk_size, marker=std_marker, alpha=mrk_alpha, label=source+format_string+notes_string+n_pts_string, zorder=(i+1))
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=source+n_pts_string))
            i += 1
        # Add legend with custom handles
        lgnd = ax.legend(handles=lgnd_hndls)
    elif clr_map == 'clr_by_instrmt':
        # Make a list of each unique instrument
        instrmts_to_plot = np.unique(np.array(data['instrmt']))
        # Loop through each instrument
        i = 0
        lgnd_hndls = []
        for instrmt in instrmts_to_plot:
            # Decide on the color, don't go off the end of the array
            my_clr = mpl_clrs[i%len(mpl_clrs)]
            # Get x and y data
            x_data = data[data.instrmt == instrmt][x_key]
            y_data = data[data.instrmt == instrmt][y_key]
            # Get format
            format_string = ''.join(data[data.instrmt == instrmt].format.unique())
            # Get notes
            notes_string = ''.join(data[data.instrmt == instrmt].notes.unique())
            # Get number of points
            n_pts_string = ' '+str(len(x_data))+' points'
            # Plot every point the same color, size, and marker
            ax.scatter(x_data, y_data, color=my_clr, s=mrk_size, marker=std_marker, alpha=mrk_alpha, label=instrmt+format_string+notes_string+n_pts_string)
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=instrmt+n_pts_string))
            i += 1
        # Add legend with custom handles
        lgnd = ax.legend(handles=lgnd_hndls)
    elif clr_map == 'clr_by_pf_no':
        # Get the profile numbers ready for plotting (strips out alphabet characters)
        pf_nos = [re.compile(r'[A-Z,a-z]').sub('', m) for m in data['prof_no'].values]
        # The color of each point corresponds to the number of the profile it came from
        heatmap = ax.scatter(data[x_key], data[y_key], c=list(map(int, pf_nos)), cmap=cmap_pf_no, s=mrk_size, marker=std_marker)
        # Create the colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('profile number')
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, data, x_key)
    elif clr_map == 'clr_by_p':
        # The color of each point corresponds to the pressure (depth) value of that point
        heatmap = ax.scatter(data[x_key], data[y_key], c=data['p'], cmap=cmap_p, s=mrk_size, marker=std_marker)
        # Create the colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('pressure (dbar) or depth (m)')
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, data, x_key)
    elif clr_map == 'clr_by_date':
        # The color of each point corresponds to the date that measurement was taken
        heatmap = ax.scatter(data[x_key], data[y_key], c=mpl.dates.date2num(data['date']), cmap=cmap_date, s=mrk_size, marker=std_marker)
        # Create the colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        # Format the numbers on the colorbar
        loc = mpl.dates.AutoDateLocator()
        cbar.ax.yaxis.set_major_locator(loc)
        cbar.ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
        cbar.set_label('date of measurement')
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, data, x_key)
    elif clr_map == 'density_hist':
        # Plot a density histogram where each grid box is colored to show how
        #   many points fall within it
        # Decide on what the limits of the colorbar should be
        clr_min = 0
        clr_max = 20
        clr_ext = 'max'        # adds arrow indicating values go past the bounds
        #                       #   use 'min', 'max', or 'both'
        # Make the 2D histogram, the number of bins really changes the outcome
        heatmap = ax.hist2d(data[x_key], data[y_key], bins=250, cmap=cmap_den_h, vmin=clr_min, vmax=clr_max)
        # `hist2d` returns a tuple, the index 3 of which is the mappable for a colorbar
        cbar = plt.colorbar(heatmap[3], ax=ax, extend=clr_ext)
        cbar.set_label('density of points')
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, data, x_key)
    else:
        # Did not provide a valid colormap
        print('Colormap',clr_map,'not valid')
        exit(0)
    #
    return xlabel, ylabel, plt_title, ax

################################################################################
################################################################################
# Functions to make other kinds of plots
################################################################################
################################################################################

def plot_arctic_map(ax, data, plt_dict, plt_title, fig, ax_pos):
    """
    Uses the given arguments to determine how to plot on a map of the Arctic
    for the given axis

    ax              The axis on which to make the plot
    data            A pandas dataframe of pre-filtered data
    plt_dict        A dictionary containing the info to create this subplot
    plt_title       A string of the title for this subplot
    fig             The figure in which ax is contained
    ax_pos          A tuple of the ax (rows, cols, linear number of this subplot)
    """
    # Plot data on map of the Arctic
    #   Set latitude and longitude extents
    #   I found these values by guess-and-check, there really isn't a good way
    #       to know beforehand what you'll actually get
    if map_extent == 'Canada_Basin':
        cent_lon = -140
        ex_N = 80
        ex_S = 69
        ex_E = -156
        ex_W = -124
    elif map_extent == 'Western_Arctic':
        cent_lon = -140
        ex_N = 80
        ex_S = 69
        ex_E = -165
        ex_W = -124
    else:
        cent_lon = 0
        ex_N = 90
        ex_S = 70
        ex_E = -180
        ex_W = 180
    # Remove the current axis
    ax.remove()
    # Replace axis with one that can be made into a map
    ax = fig.add_subplot(ax_pos, projection=ccrs.NorthPolarStereo(central_longitude=cent_lon))
    ax.set_extent([ex_E, ex_W, ex_S, ex_N], ccrs.PlateCarree())
    print(ax.get_extent(crs=ccrs.PlateCarree()))
    #   Add ocean first, then land. Otherwise the ocean covers the land shapes
    # ax.add_feature(cartopy.feature.OCEAN, color=clr_ocean)
    # ax.add_feature(cartopy.feature.LAND, color=clr_land, alpha=0.5)
    #   Add gridlines to show longitude and latitude
    gl = ax.gridlines(draw_labels=True, color=clr_lines, alpha=0.3, linestyle='--')
    #       x is actually all labels around the edge
    gl.xlabel_style = {'size':6, 'color':clr_lines}
    #       y is actually all labels within map
    gl.ylabel_style = {'size':6, 'color':clr_lines}
    #   Plotting the coastlines takes a really long time
    # ax.coastlines()
    #
    # Remove rows of the data frame with missing data for lon and lat
    data = data[data.lon.notnull() & data.lat.notnull()]
    # Create blank dataframe in which to collect profile info
    map_df = pd.DataFrame({'source':[], 'instrmt':[], 'prof_no':[], 'lon':[], 'lat':[], 'date':[]})
    # Get the unique profiles from the data for each instrument
    unique_instrmts = np.unique(np.array(data['instrmt']))
    for instrmt in unique_instrmts:
        # print('For instrument:',instrmt, type(instrmt))
        instrmt_df = data[data['instrmt'] == instrmt]
        # print(instrmt_df)
        pfs_to_plot = np.unique(np.array(instrmt_df['prof_no']))
        for prof_no in pfs_to_plot:
            # Get the longitude and latitude values
            pf_df_lon = instrmt_df[instrmt_df['prof_no']==prof_no].lon
            pf_df_lat = instrmt_df[instrmt_df['prof_no']==prof_no].lat
            # Just in case one profile has multiple lat and lon values
            lon = np.average(np.unique(np.array(pf_df_lon, dtype=float)))
            lat = np.average(np.unique(np.array(pf_df_lat, dtype=float)))
            # Get the source, date, format, and notes values
            pf_df_sou = instrmt_df[instrmt_df['prof_no']==prof_no].source
            pf_df_date= instrmt_df[instrmt_df['prof_no']==prof_no].date
            pf_df_form= instrmt_df[instrmt_df['prof_no']==prof_no].format
            pf_df_note= instrmt_df[instrmt_df['prof_no']==prof_no].notes
            # Just take the source, date, format, and notes values from
            #   the first entry, close enough
            source = np.unique(np.array(pf_df_sou))[0]
            date   = np.unique(np.array(pf_df_date))[0]
            format = np.unique(np.array(pf_df_form))[0]
            notes  = np.unique(np.array(pf_df_note))[0]
            # Add values to a dictionary
            pf_dict = {'source': source,
                       'instrmt': instrmt,
                       'prof_no': prof_no,
                       'lon': lon,
                       'lat': lat,
                       'date': date,
                       'format': format,
                       'notes': notes
                      }
            #
            # Add to the list of profile dataframes
            map_df = map_df.append(pf_dict, ignore_index=True)
        #
    #
    clr_map = plt_dict['color_map']
    # Determine the color mapping to be used
    if clr_map == 'clr_all_same':
        # Plot every point the same color, size, and marker
        ax.scatter(map_df['lon'], map_df['lat'], color=std_clr, s=map_mrk_size, marker=map_marker, alpha=mrk_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree())
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, map_df, 'lon')
    elif clr_map == 'clr_by_source':
        # Make a list of each unique source, without sorting
        sources_to_plot = [plt_dict['data_sources'][0][0]]
        if len(plt_dict['data_sources']) > 1:
            for i in range(1, len(plt_dict['data_sources'])):
                source = plt_dict['data_sources'][i][0]
                if not source in sources_to_plot:
                    sources_to_plot.append(source)
        print('sources_to_plot',sources_to_plot)
        # Loop through each source
        i = 0
        lgnd_hndls = []
        for source in sources_to_plot:
            pf_df = map_df[map_df.source == source]
            # Decide on the color, don't go off the end of the array
            my_clr = mpl_clrs[i%len(mpl_clrs)]
            # Get lon and lat data
            lon_data = pf_df.lon
            lat_data = pf_df.lat
            # Find the number of profiles for that instrument
            n_pfs_string = ' (' + str(pf_df.shape[0]) + ' profiles)'
            # Plot every point the same color, size, and marker
            ax.scatter(lon_data, lat_data, color=my_clr, s=map_mrk_size, marker=map_marker, alpha=mrk_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), label=source+n_pfs_string, zorder=(2-i))
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=source+n_pfs_string))
            i += 1
        # Add legend with custom handles
        lgnd = ax.legend(handles=lgnd_hndls)
    elif clr_map == 'clr_by_instrmt':
        # Loop through each instrument
        i = 0
        lgnd_hndls = []
        for instrmt in unique_instrmts:
            pf_df = map_df[map_df.instrmt == instrmt]
            # Decide on the color, don't go off the end of the array
            my_clr = mpl_clrs[i%len(mpl_clrs)]
            # Get lon and lat data
            lon_data = pf_df.lon
            lat_data = pf_df.lat
            # Find the number of profiles for that instrument
            n_pfs_string = ' (' + str(pf_df.shape[0]) + ' profiles)'
            # Plot every point the same color, size, and marker
            ax.scatter(lon_data, lat_data, color=my_clr, s=map_mrk_size, marker=map_marker, alpha=mrk_alpha, linewidths=map_ln_wid, transform=ccrs.PlateCarree(), label=instrmt+n_pfs_string)
            lgnd_hndls.append(mpl.patches.Patch(color=my_clr, label=instrmt+n_pfs_string))
            i += 1
        #
        # Add legend with custom handles
        lgnd = ax.legend(handles=lgnd_hndls)
    elif clr_map == 'clr_by_pf_no':
        # Get the profile numbers ready for plotting (strips out alphabet characters)
        pf_nos = [re.compile(r'[A-Z,a-z]').sub('', m) for m in map_df['prof_no'].values]
        # The color of each point corresponds to the number of the profile it came from
        heatmap = ax.scatter(map_df['lon'], map_df['lat'], c=list(map(int, pf_nos)), cmap=cmap_pf_no, s=map_mrk_size, marker=map_marker, linewidths=map_ln_wid, transform=ccrs.PlateCarree())
        # Create the colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('profile number')
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, map_df, 'lon')
    elif clr_map == 'clr_by_date':
        # The color of each point corresponds to the date that measurement was taken
        heatmap = ax.scatter(map_df['lon'], map_df['lat'], c=mpl.dates.date2num(map_df['date']), cmap=cmap_date, s=map_mrk_size, marker=map_marker, linewidths=map_ln_wid, transform=ccrs.PlateCarree())
        # Create the colorbar
        cbar = plt.colorbar(heatmap, ax=ax)
        # Format the numbers on the colorbar
        loc = mpl.dates.AutoDateLocator()
        cbar.ax.yaxis.set_major_locator(loc)
        cbar.ax.yaxis.set_major_formatter(mpl.dates.ConciseDateFormatter(loc))
        cbar.set_label('date of measurement')
        # Add title
        plt_title = add_std_title(plt_dict, plt_title, data)
        # Add legend
        add_std_legend(ax, map_df, 'lon')
    else:
        # Did not provide a valid colormap
        print('Colormap',clr_map,'not valid')
        exit(0)
    #
    # Return new axis so it's labels and title can be changed later
    return ax, plt_title

################################################################################

def plot_profiles(ax, data, plt_dict, plt_title):
    """
    Uses the given arguments to plot the individual profiles in data

    ax              The axis on which to make the plot
    data            A pandas dataframe of pre-filtered data
    plt_dict        A dictionary containing the info to create this subplot
    plt_title       A string of the title for this subplot
    """
    # Find the color map
    clr_map = plt_dict['color_map']
    # Create the twin axis for salinity
    ax1 = ax.twiny()
    ax1.set_xlabel(r'Salinity (g/kg)')
    # Change color of the ticks on both axes
    ax.tick_params(axis='x', colors=clr_temp)
    ax1.tick_params(axis='x', colors=clr_salt)
    # Get the unique profiles from the data for each instrument
    unique_instrmts = np.unique(np.array(data['instrmt']))
    profile_dictionaries = []
    for instrmt in unique_instrmts:
        instrmt_df = data[data['instrmt'] == instrmt]
        pfs_to_plot = np.unique(np.array(instrmt_df['prof_no']))
        for prof_no in pfs_to_plot:
            # Get the longitude and latitude values
            pf_temp = np.array(instrmt_df[instrmt_df['prof_no']==prof_no].temp.values)
            pf_salt = np.array(instrmt_df[instrmt_df['prof_no']==prof_no].salt.values)
            pf_p    = np.array(instrmt_df[instrmt_df['prof_no']==prof_no].p.values)
            # Get format
            format_string = ''.join(instrmt_df.format.unique())
            # Get notes
            notes_string = ''.join(instrmt_df.notes.unique())
            # Create dictionary for this profile
            pf_dict = {'instrmt': instrmt,
                       'prof_no': prof_no,
                       'format': format_string,
                       'notes': notes_string,
                       'temp': pf_temp,
                       'salt': pf_salt,
                       'p': pf_p}
            #
            profile_dictionaries.append(pf_dict)
        #
    #
    # Plot each profile
    if len(profile_dictionaries) > 15:
        print("You are trying to plot",len(profile_dictionaries),"profiles.")
        print("That is too many. Try less than 15")
        exit(0)
    for i in range(len(profile_dictionaries)):
        data = profile_dictionaries[i]
        # Decide on marker and line styles, don't go off the end of the array
        mkr     = mpl_mrks[i%len(mpl_mrks)]
        l_style = l_styles[i%len(l_styles)]
        # Adjust the starting points of each subsequent profile
        if i == 0:
            # Find upper and lower temperature and salinity bounds, for reference points
            temp_low  = min(data['temp'])
            temp_high = max(data['temp'])
            salt_low  = min(data['salt'])
            salt_high = max(data['salt'])
            # Define spans for the temperature and salinity profile
            t_span = temp_high - temp_low
            s_span = salt_high - salt_low
            # Define temperature and salinity arrays to plot
            #   Shift the salinity profile over a little bit
            temp = data['temp']
            salt = data['salt'] + s_span/5
            # Adjust lower temperature and salinity bounds
            temp_low  = temp_low - t_span/5
            salt_low  = salt_low - s_span/3
            # Adjust upper temperature and salinity bounds
            temp_high = max(temp)
            salt_high = max(salt)
        else:
            # Define temperature and salinity arrays to plot
            temp = data['temp'] - min(data['temp']) + temp_high
            salt = data['salt'] - min(data['salt']) + salt_high
            # Adjust upper temperature and salinity bounds
            temp_high = max(temp)
            salt_high = max(salt)
            # Define spans for the temperature and salinity profile
            t_span = temp_high - min(data['temp'])
            s_span = salt_high - min(data['salt'])
        # Plot every temperature point the same color, size, and marker
        pf_label = data['instrmt'] + data['format'] + data['notes'] + '-' + str(data['prof_no'])
        ax.plot(temp, -data['p'], color=clr_temp, linestyle=l_style, label=pf_label)
        # Plot every salinity point the same color, size, and marker
        ax1.plot(salt, -data['p'], color=clr_salt, linestyle=l_style)
        # Apply any colormapping, if appropriate
        if clr_map == 'clr_by_p':
            # Plot every temperature point the same color, size, and marker
            ax.scatter(temp, -data['p'], color=clr_temp, s=pf_mrk_size, marker=mkr)
            # Plot every salinity point the same color, size, and marker
            ax1.scatter(salt, -data['p'], color=clr_salt, s=pf_mrk_size, marker=mkr)
        #
    #
    # Adjust bounds on axes
    ax.set_xlim([temp_low, temp_high + (t_span/5)])
    ax1.set_xlim([salt_low, salt_high + (s_span/5)])
    # Add legend
    lgnd = ax.legend()
    # Need to change the marker size for each label in the legend individually
    for hndl in lgnd.legendHandles:
        hndl._sizes = [lgnd_mrk_size]
    #
    # Return new axis so it's labels and title can be changed later
    return ax, plt_title

################################################################################

def find_p_res(data):
    """
    Finds the difference between each sequential pressure measurement in each
    profile in the data and stores it as a new column called 'res'

    df      A pandas DataFrame with the following columns:
        instrmt     A string of the instrmt name that took the profile
        prof_no     The profile number
        temp        An array of temperature values
        salt        An array of salinity values
        p           An array of depth values (in m)
    """
    # Add a new column for the resolution values
    data['res'] = None
    # Create an empty list to add each modified profile to
    output_list = []
    # Loop across each instrument
    instrmts = np.unique(np.array(data['instrmt']))
    for instrmt in instrmts:
        # Find the data for just that instrument
        data_instrmt = data[data.instrmt == instrmt]
        # Loop across each profile
        pfs = np.unique(np.array(data['prof_no']))
        for pf in pfs:
            # Find the data just for that profile
            data_pf = data_instrmt[data_instrmt.prof_no == pf]
            # Sort that profile by the pressure (depth) values
            data_sorted = data_pf.sort_values(by='p')
            # Add first difference of depth values to data frame
            data_sorted['res'] = abs(data_sorted['p'].diff())
            # Add that dataframe to the list
            output_list.append(data_sorted)
        #
    #
    # Combine all the modified profile data frames into one
    data = pd.concat(output_list)
    # Remove rows of the data frame with missing data
    #   Note: only apply to res because 'format' will often be
    #       set to a null value, for exmaple with AIDJEX data
    data = data[data.res.notnull()]
    return data

################################################################################

def find_date_res(data):
    """
    Finds the difference between each sequential date measurement of each
    profile in the data and stores it as a new column called 'res'

    df      A pandas DataFrame with the following columns:
        instrmt     A string of the instrmt name that took the profile
        prof_no     The profile number
        temp        An array of temperature values
        salt        An array of salinity values
        p           An array of depth values (in m)
    """
    # Create an empty list to add each modified profile to
    output_list = []
    # Get the unique profiles from the data for each instrument
    unique_instrmts = np.unique(np.array(data['instrmt']))
    for instrmt in unique_instrmts:
        # Create blank dataframe in which to collect profile info
        date_df = pd.DataFrame({'source':[], 'instrmt':[], 'prof_no':[], 'date':[], 'res':[], 'format':[], 'notes':[]})
        # print('For instrument:',instrmt, type(instrmt))
        instrmt_df = data[data['instrmt'] == instrmt]
        # print(instrmt_df)
        pfs_to_plot = np.unique(np.array(instrmt_df['prof_no']))
        for prof_no in pfs_to_plot:
            # print('    For profile:',prof_no, type(prof_no))
            # Get the source, date, format, and notes values
            pf_df_sou = instrmt_df[instrmt_df['prof_no']==prof_no].source
            pf_df_date= instrmt_df[instrmt_df['prof_no']==prof_no].date
            pf_df_form= instrmt_df[instrmt_df['prof_no']==prof_no].format
            pf_df_note= instrmt_df[instrmt_df['prof_no']==prof_no].notes
            # Just take the source, date, format, and notes values from
            #   the first entry, close enough
            source = np.unique(np.array(pf_df_sou))[0]
            date   = np.unique(np.array(pf_df_date))[0]
            format = np.unique(np.array(pf_df_form))[0]
            notes  = np.unique(np.array(pf_df_note))[0]
            # Add values to a dictionary
            pf_dict = {'source': source,
                       'instrmt': instrmt,
                       'prof_no': prof_no,
                       'date': date,
                       'res': None,
                       'format': format,
                       'notes': notes
                      }
            #
            # Add to the list of profile dataframes
            date_df = date_df.append(pf_dict, ignore_index=True)
        #
        # Sort that instrument's dataframe by the date values
        data_sorted = date_df.sort_values(by=['date'])
        # Add first difference of date values to data frame
        data_sorted['res'] = abs(data_sorted['date'].diff())
        # Add that dataframe to the list
        output_list.append(data_sorted)
    #
    # Combine all the modified profile data frames into one
    df = pd.concat(output_list)
    # Remove rows of the data frame with missing data
    #   Note: only apply to res because 'format' will often be
    #       set to a null value, for exmaple with AIDJEX data
    df = df[df.res.notnull()]
    # print(df)
    # exit(0)
    return df

################################################################################
