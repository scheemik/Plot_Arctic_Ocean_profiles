"""
This is a script which will plot many different kinds of plots using the
functions in the `helper_functions.py` script

made by: Mikhail Schee (June 2022)
"""

# For custom plotting functions
import helper_functions as hf

# Common filters
staircase_range = [200,300]
test_range = [220,240]

################################################################################
############### Define groups of sources I will commonly use ###################
################################################################################

# All 4 AIDJEX stations
all_AIDJEX           = [
                            ('AIDJEX', 'BigBear'),
                            ('AIDJEX', 'BlueFox'),
                            ('AIDJEX', 'Caribou'),
                            ('AIDJEX', 'Snowbird')
                        ]

# All the ITPs who's data I was able to download. Last updated in February 2022
all_ITPs              = [
                            ('ITP', '1', 'cormat'),
                            ('ITP', '2', 'cormat'),
                            ('ITP', '3', 'cormat'),
                            ('ITP', '4', 'cormat'),
                            ('ITP', '5', 'cormat'),
                            ('ITP', '6', 'cormat'),
                            ('ITP', '7', 'cormat'),
                            ('ITP', '8', 'cormat'),
                            ('ITP', '9', 'cormat'),
                            ('ITP', '10', 'cormat'),
                            ('ITP', '11', 'cormat'),
                            ('ITP', '12', 'cormat'),
                            ('ITP', '13', 'cormat'),
                            ('ITP', '14', 'cormat'),
                            ('ITP', '15', 'cormat'),
                            ('ITP', '16', 'cormat'),
                            ('ITP', '17', 'cormat'),
                            ('ITP', '18', 'cormat'),
                            ('ITP', '19', 'cormat'),
                            # ('ITP', '20', 'cormat'), # no profiles recorded
                            ('ITP', '21', 'cormat'),
                            ('ITP', '22', 'cormat'),
                            ('ITP', '23', 'cormat'),
                            ('ITP', '24', 'cormat'),
                            ('ITP', '25', 'cormat'),
                            ('ITP', '26', 'cormat'),
                            ('ITP', '27', 'cormat'),
                            ('ITP', '28', 'cormat'),
                            ('ITP', '29', 'cormat'),
                            ('ITP', '30', 'cormat'),
                            # ('ITP', '31', 'cormat'), # not in Arctic
                            ('ITP', '32', 'cormat'),
                            ('ITP', '33', 'cormat'),
                            ('ITP', '34', 'cormat'),
                            ('ITP', '35', 'cormat'),
                            ('ITP', '36', 'cormat'),
                            ('ITP', '37', 'cormat'),
                            ('ITP', '38', 'cormat'),
                            # ('ITP', '39', 'cormat'), # not in Arctic
                            # ('ITP', '40', 'cormat'), # not in Arctic
                            ('ITP', '41', 'cormat'),
                            ('ITP', '42', 'cormat'),
                            ('ITP', '43', 'cormat'),
                            # ('ITP', '44', 'cormat'), # no profiles recorded
                            # ('ITP', '45', 'cormat'), # not in Arctic
                            # ('ITP', '46', 'cormat'), # not in Arctic
                            ('ITP', '47', 'cormat'),
                            ('ITP', '48', 'cormat'),
                            ('ITP', '49', 'cormat'),
                            # ('ITP', '50', 'cormat'), # no profiles recorded
                            ('ITP', '51', 'cormat'),
                            ('ITP', '52', 'cormat'),
                            ('ITP', '53', 'cormat'),
                            ('ITP', '54', 'cormat'),
                            ('ITP', '55', 'cormat'),
                            ('ITP', '56', 'cormat'),
                            ('ITP', '57', 'cormat'),
                            ('ITP', '58', 'cormat'),
                            ('ITP', '59', 'cormat'),
                            ('ITP', '60', 'cormat'),
                            ('ITP', '61', 'cormat'),
                            ('ITP', '62', 'cormat'),
                            ('ITP', '63', 'cormat'),
                            ('ITP', '64', 'cormat'),
                            ('ITP', '65', 'cormat'),
                            # ('ITP', '66', 'cormat'), # no profiles available
                            # ('ITP', '67', 'cormat'), # no profiles available
                            ('ITP', '68', 'cormat'),
                            ('ITP', '69', 'cormat'),
                            ('ITP', '70', 'cormat'),
                            # ('ITP', '71', 'cormat'), # no `cormat` version
                            ('ITP', '72', 'cormat'),
                            ('ITP', '73', 'cormat'),
                            ('ITP', '74', 'cormat'),
                            ('ITP', '75', 'cormat'),
                            ('ITP', '76', 'cormat'),
                            ('ITP', '77', 'cormat'),
                            ('ITP', '78', 'cormat'),
                            ('ITP', '79', 'cormat'),
                            ('ITP', '80', 'cormat'),
                            ('ITP', '81', 'cormat'),
                            ('ITP', '82', 'cormat'),
                            ('ITP', '83', 'cormat'),
                            ('ITP', '84', 'cormat'),
                            ('ITP', '85', 'cormat'),
                            ('ITP', '86', 'cormat'),
                            ('ITP', '87', 'cormat'),
                            ('ITP', '88', 'cormat'),
                            ('ITP', '89', 'cormat'),
                            ('ITP', '90', 'cormat'),
                            ('ITP', '91', 'cormat'),
                            ('ITP', '92', 'cormat'),
                            # ('ITP', '93', 'cormat'), # no `cormat` version
                            ('ITP', '94', 'cormat'),
                            ('ITP', '95', 'cormat'),
                            # ('ITP', '96', 'cormat'), # dosen't exist
                            ('ITP', '97', 'cormat'),
                            ('ITP', '98', 'cormat'),
                            ('ITP', '99', 'cormat'),
                            ('ITP', '100', 'cormat'),
                            ('ITP', '101', 'cormat'),
                            ('ITP', '102', 'cormat'),
                            ('ITP', '103', 'cormat'),
                            ('ITP', '104', 'cormat'),
                            ('ITP', '105', 'cormat'),
                            # ('ITP', '106', 'cormat'), # doesn't exist
                            ('ITP', '107', 'cormat'),
                            ('ITP', '108', 'cormat'),
                            ('ITP', '109', 'cormat'),
                            ('ITP', '110', 'cormat'),
                            ('ITP', '111', 'cormat'),
                            # ('ITP', '112', 'cormat'), # no `final` version
                            ('ITP', '113', 'cormat'),
                            ('ITP', '114', 'cormat'),
                            # ('ITP', '115', 'cormat'), # no `cormat` version
                            ('ITP', '116', 'cormat')
                        ]

################################################################################
################ Define the parameters of the plots to make ####################
################################################################################

# Each element in this list is a dictionary defining what will be plotted
#   in the subplot that corresponds to that index of the list

to_plot = [
    {
        'data_sources': [
            # ('ITP', '1', 'cormat')
            ('ITP', '2', 'cormat')
            # ('AIDJEX', 'BigBear'),
            # ('AIDJEX', 'BlueFox'),
            # ('AIDJEX', 'Caribou'),
            # ('AIDJEX', 'Snowbird')
        ],
        'filtering_types': [
            {
             'p_range': staircase_range
             # 'white_list': {'ITP': {'2': ['1', '3']}}
             # 'white_list': {'ITP': {'2': ['1', '3'], '3': []}, 'AIDJEX': {'Snowbird': ['7']}}
            }
        ],
        'plot_type':
            # 'T-S'
            # 'res_vs_p'
            # 'res_hist'
            # 'date_hist'
            'map'
            # 'profiles'
        ,
        'color_map':
            'clr_all_same'
            # 'clr_by_source'
            # 'clr_by_instrmt'
            # 'clr_by_pf_no'
            # 'clr_by_p'
            # 'clr_by_date'
            # 'density_hist'
    }
]

################################################################################

################################################################################
# Main execution of code

hf.make_plots(to_plot, filename=None)
