#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle


def get_team_data(teams_source, year, scrape_data):
    """
       Returns team-specific data.
       'year' specificies which year of data to collect.
       'scrape_data' will load existing data if set to False, but will scrape data from the web if set to True
    """

    #grab logistical data about specific year (team names, sources, etc.)
    sources, teams, team_dict = get_logistics(teams_source, year)

    ## run if scraping new data
    if scrape_data:

        # scrape data from the web
        dataframes = webscrape_data(year, sources, teams)

    ## run if loading existing 
    else:

        dataframes = load_frames(teams=teams, year=year)

    return dataframes, teams, team_dict


def get_logistics(teams_source, year):
    """
        Takes in 'teams_source' which has urls and team names for each season.
        The 'year' argument is used to specify which year of data to grab.
        Function returns the year, url sources, team names, and a team dictionary.
        The dictionary maps each team name to an integer
    """
    
    # Grabs logistical data about the college basketball teams
    sources = teams_source[year].dropna()                  # website sources for each team
    teams = teams_source[year + '_Teams'].dropna()         # names of teams 
    team_nums = list(range(len(teams))) # idx numbers for each team
    team_dict = {teams[num]: num for num in team_nums}
    
    return sources, teams, team_dict


def load_frames(teams=[], year='2016', multiple=False):
    """
        Optional arguments are the list of team_names (needed if only loading one season),
        the 'year' for which to grab the data, and 'multiple'. 'multiple' is "True" if 
        getting data for multiple seasons and "False" otherwise.
        
        If grabbing multiple seasons, function returns one dataframe for all games.
        If grabbing one season, function returns a list of dataframes for that season.
        
        All data comes from pre-processed files that were created using data from the web.
    """
    
    if multiple:
        
        # loads one dataframe that contains all games
        dataframes = pd.read_csv('./Final_Dataframes/Teams_Only.csv')
        
        dataframes['Date'] = pd.to_datetime(dataframes['Date'])
    
    else:
        
        # loads in a separate dataframe for each team
        #dataframes = [pd.read_csv('./Team_Dataframes/' + year + '/Team_Only/' + teams[ii] + '.csv') for ii in range(len(teams))]
        dataframes = []
        for ii in range(len(teams)):
            
            with open('./Team_Dataframes/' + year + '/Team_Only/' + teams[ii] + '.pickle', 'rb') as f: 
                
                dataframes.append(pickle.load(f))
        
        # converts the 'Date' column to a datetime 
        #for df in dataframes:

            # Convert 'Date' to datetime for evaluation later in runtime
            #df['Date'] = pd.to_datetime(df['Date'])
    
    print('DONE LOADING FRAMES', year)
    
    return dataframes
    
    
def webscrape_data(year, sources, teams): 
    """
        Takes in 'year', 'sources', and 'teams'. Year is the year of the data, sources are the urls to
        navigate to, and teams is the list of team names.
        Function returns a list of dataframes for each team in that season
    """
    
    print('BEGINNING WEB SCRAPE', year)
    
    dataframes = [] # list of dataframes

    # Loops through each team listed in the sources list of urls
    for ii, src in enumerate(sources):
        
        soup = BeautifulSoup(requests.get(src).text, "html.parser")
        tags = soup.find_all('td') # grab only tags with 'td' which grabs the data table from the website

        data = []
        row = []
        # loops through extracted tag
        for idx, tr in enumerate(tags):
            row.append(tr.text) # appends text of the tag

            # Given that there are 39 columns of data in the table,
            # a check is perfromed on if the tag is the last tag
            # in the row
            if (idx + 1)%39 == 0: 

                # if the opponent team name ('idx 3'), does not an 'a' href,
                # then that team is not a Division 1 school and will not have 
                # data of its own in the dataframe. Because the program will use
                # data from the opponent, all data from non-Division 1 schools
                # is removed.
                if tags[idx - 36].find('a'):
                    data.append(row) # row of data is appended to data list if opponent is Division 1
                row = []
        
        # Conference String
        conf = soup.find_all('p')[2].find('a').text
        
        # Removes cancelled game between Hampton and Morgan State
        # Score was 2-0 at the time of cancellation
        if (year == '2018') and ((ii == 112) or (ii == 191)):
            if ii == 112:
                data = data[:26] + data[27:]
            else:
                data = data[:24] + data[25:]

        # dataframe is created and key columns are renamed to better represent the data
        df = pd.DataFrame(data).rename(columns = {0: 'Date', 1: 'Loc', 2: 'Opp_name', 3: 'W', 4: 'Score', 5: 'Opp_score', 6: 'FG', 7: 'FGA', 8: 'FG_Pct', 9: 'Threes', 10: 'ThreesA', 11: 'Threes_Pct',
                                                  12: 'FT', 13: 'FTA', 14: 'FT_Pct', 15: 'ORB', 16: 'TRB', 17: 'AST', 18: 'STL', 19: 'BLK', 20: 'TOV', 21: 'PF', 23: 'Opp_FG', 24: 'Opp_FGA', 25: 'Opp_FG_Pct',
                                                  26: 'Opp_Threes', 27: 'Opp_ThreesA', 28: 'Opp_Threes_Pct', 29: 'Opp_FT', 30: 'Opp_FTA', 31: 'Opp_FT_Pct', 32: 'Opp_ORB', 33: 'Opp_TRB', 34: 'Opp_AST',
                                                  35: 'Opp_STL', 36: 'Opp_BLK', 37: 'Opp_TOV', 38: 'Opp_PF'})

        # datatypes of numerical columns are changed to floats 
        try:
            df = df.astype({key: float for key in df.keys() if key not in ['Date', 'Loc', 'Opp_name', 'W', 22]})
        except ValueError: # handles value errors (by looking through the data it was determined there are a few instances where a cell was left blank)
            for col in df: # iterates through each column in dataframe
                float_vals = []
                for val in df[col]: # iterates through each value in column
                    try:
                        float_vals.append(float(val)) # converts to float if possible
                    except ValueError:   # used for values that cannot be converted to float
                        if col not in ['Date', 'Loc', 'Opp_name', 'W', 22]:  # checks to make sure it wasn't supposed to be a String
                            float_vals.append(np.nan) # replace blank space with NaN
                        else:
                            float_vals.append(val) # leaves value as is

                df[col] = pd.Series(float_vals) # replaces column with new "float_values"

        # Convert 'Date' to datetime for evaluation later in runtime
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate running difference of dates to get days of rest
        df['Rest_Days'] = df['Date'].diff().dt.days

        # The result column is added which represents how much the team lost or won by in that game
        # This will be the y-vector that the prediction model uses.
        df['result'] = df['Score'] - df['Opp_score']
        
        # The 'Win' column is 1 if the team won the game and 0 otherwise.
        # This will be another y-vector that the model predicts
        df['Win'] = np.where(df['result'] > 0, 1, 0)

        # Keep track of the Game Number
        df['Game_Number'] = list(range(1, len(df) + 1))

        # Calcultes the mean score of each game ( (team_score + opp_score) / 2 )
        df['Mean_Score'] = df[['Score', 'Opp_score']].mean(axis=1)
        
        # Create Dummy Variables for Home and Away
        location = pd.get_dummies(df['Loc'])
        if location.shape[1] == 3:
            location.columns = ['Home', 'Away', 'Neutral']
        else: # needed for teams that don't have any neutral-site games
            location.columns = ['Home', 'Away'] 

        # concatenate original dataframe to new location dummy dataframe
        df = pd.concat((df, location), axis = 1)
        
        # Add Home and Away columns
        df['Comb_Loc'] = df['Home'] + (-1 * df['Away'])

        # # Create running means of key statistics.
        # # Running means represent the means entering the game
        # df['Avg_Score'], df['Opp_Avg_Score'], df['Avg_Threes'], df['Opp_Avg_Threes'], df['Avg_FG_Pct'], df['Opp_Avg_FG_Pct'], df['Avg_Mean_Score'], df['Home_Pct'] = \
        # running_means([df['Score'], df['Opp_score'], df['Threes'], df['Opp_Threes'], df['FG_Pct'], df['Opp_FG_Pct'], df['Mean_Score'], df['Comb_Loc']])
        
        # # Scale Home Pct
        # df['Home_Pct'] = (df['Home_Pct'] ** 2) * df['Game_Number']
        
        # # Percentage of points scored by 3-pointers
        # df['Avg_Threes'] = (3*df['Avg_Threes'])/df['Avg_Score'] - threes_pct
        # df['Opp_Avg_Threes'] = (3*df['Opp_Avg_Threes'])/df['Opp_Avg_Score'] - threes_pct
        
        # # Calculates the average value of vicotry/loss
        # df['Avg_Result'] = df['Avg_Score'] - df['Opp_Avg_Score']

        # # Calculates the percent margin of victory/loss as percentage of mean scores
        # df['Pct_Margin'] = df['Avg_Result'] / df['Avg_Mean_Score'] 

        # # create a column that contains the average season result in every row
        # df['Avg_Result_Fin'] = [df['Score'].mean() - df['Opp_score'].mean()] * len(df)
        # df['Opp_Avg_FG_Pct_Fin'] = [df['Opp_FG_Pct'].mean()] * len(df)
        # df['Pct_Margin_Fin'] = [df['Avg_Result_Fin'][0] / (df['Mean_Score'].mean())] * len(df)

        avg_prefixes = ['SMA_', 'EMA_', 'Last3Avg_', 'Prev_']
        
        # Calculate Moving Averages for Key Metrics
        test_feats = ['Score', 'Opp_score', 'Threes', 'Opp_Threes', 'FG_Pct', 'Opp_FG_Pct', 'Mean_Score', 'Comb_Loc']
        df = mov_avgs(df, test_feats)

        # Adjust Percentage of games played at homes
        test_feats = ['Comb_Loc']
        df = home_pcts(df, test_feats, avg_prefixes)

        # Convert Number of Threes to Pct of Points Scored by 3s
        test_feats = ['Threes', 'Opp_Threes']
        df = three_pt_pcts(df, test_feats, avg_prefixes)

        # Grab Average Game Results
        df = avg_results(df, avg_prefixes)

        # Calculates the percent margin of victory/loss as percentage of mean scores
        df = pct_margins(df, avg_prefixes)

        # Calculate Finals Columns
        df = create_fins(df, avg_prefixes)
        
        # Create conference column
        df['Conference'] = [conf] * len(df)
        
        # saves dataframe
        df.to_csv('./Team_Dataframes/' + year + '/Team_Only/' + teams[ii] + '.csv', index=False)
        
        # Save dataframe to pickle file
        with open('./Team_Dataframes/' + year + '/Team_Only/' + teams[ii] + '.pickle', 'wb') as f:
            pickle.dump(df, f)

        # add dataframe to list of dataframes
        dataframes.append(df)
        
        # Show progress
        if ii%20 == 0:
            print(ii)
    
    print('DONE WITH DATA INTAKE', year)
    return dataframes


# Calculate simple moving average and exponentially weighted moving average 
def mov_avgs(df, test_feats):
    len_test = len(df)
    for col in df[test_feats]:
        sma = df[col].expanding().mean()
        df['SMA_' + col + '_Fin'] = [list(sma)[-1]] * len_test
        df['SMA_' + col] = sma.shift(1)
        ema = df[col].ewm(span=len_test).mean()
        df['EMA_' + col + '_Fin'] = [list(ema)[-1]] * len_test
        df['EMA_' + col] = ema.shift(1)
        last_3 = df[col].rolling(3).mean()
        df['Last3Avg_' + col + '_Fin'] = [list(last_3)[-1]] * len_test
        df['Last3Avg_' + col] = last_3.shift(1)
        prev = df[col].rolling(1).mean()
        df['Prev_' + col + '_Fin'] = [list(prev)[-1]] * len_test
        df['Prev_' + col] = prev.shift(1)
        
    return df


def home_pcts(df, test_feats, prfxs):
    game_numbers = df['Game_Number']
    for col in df[test_feats]:
        for prfx in prfxs:
            df[prfx + col] = (df[prfx + col] ** 2) * game_numbers
            
    return df


def three_pt_pcts(df, test_feats, prfxs):
    threes_pct = 0.31352436210388857

    # Loop through each column of interest in the dataframe
    for col in df[test_feats]:
        # Loop through the different averages
        for prfx in prfxs:
            if col == 'Threes':
                df[prfx + col] = (3*df[prfx + col])/df[prfx+'Score'] - threes_pct
            else:
                df[prfx + col] = (3*df[prfx + col])/df[prfx+'Opp_score'] - threes_pct
    
    return df


def avg_results(df, prfxs):
    # Loop through the different averages
    for prfx in prfxs:
        df[prfx + 'Result'] = df[prfx + 'Score'] - df[prfx + 'Opp_score']
    return df


def pct_margins(df, prfxs):
    # Loop through the different averages
    for prfx in prfxs:
        df[prfx + 'Pct_Margin'] = df[prfx + 'Result'] / df[prfx + 'Mean_Score']
    return df


def create_fins(df, prfxs):
    # Loop through each column of interest in the dataframe
    for prfx in prfxs:
        df[prfx + 'Result_Fin'] = df[prfx + 'Score_Fin'] - df[prfx + 'Opp_score_Fin']
        df[prfx + 'Pct_Margin_Fin'] = df[prfx + 'Result_Fin'][0] / df[prfx + 'Mean_Score_Fin'][0]
    return df


def running_means(columns):
    """
        Function takes in columns of data and returns the running means of
        those columns.
    """
    
    means = []
    
    for feature in columns:
        
        running_mean = []
        
        for ii in range(len(feature)):
            running_mean.append(feature[:ii].mean())
            
        means.append(running_mean)
    
    return means


def combine_data(dataframes, team_dict, teams, year='2016'):
    """
    Takes in a list of dataframes
    Function looks at team's opponents' dataframes and concatenates the team's dataframe
    with data from the opponents.
    Returns a list of concatenated dataframes for each team
    """
    
    print('BEGINNING COMBINING DATA', year)
    
    # Initialize a list of dataframes that will combine a team's dataframe with data from its opponents
    combined_df = []
    next_games = []
    
    # Dictionary of columns to convert
    # columns_dict = {'result': 'result2', 'Mean_Score': 'Mean_Score2', 'Opp_Avg_Score': 'Opp_Avg_Score2', 'Avg_Score': 'Avg_Score2',
    #                 'Avg_Mean_Score': 'Avg_Mean_Score2', 'Avg_Result': 'Avg_Result2', 'Pct_Margin': 'Pct_Margin2', 'Home': 'Home2',
    #                 'Away': 'Away2', 'Neutral': 'Neutral2', 'Date': 'Date2', 'Loc': 'Loc2', 'Opp_name': 'Opp_name2', 'Win': 'Win2',
    #                 'Opp_score': 'Opp_score2', 'Score': 'Score2', 'Threes': 'Threes2', 'Opp_Threes': 'Opp_Threes2', 'W': 'W2',
    #                 'Avg_Threes': 'Avg_Threes2', 'Opp_Avg_Threes': 'Opp_Avg_Threes2', 'Game_Number': 'Game_Number2', 'Home_Pct': 'Home_Pct2',
    #                 'Avg_Result_Fin': 'Avg_Result_Fin2', 'Opp_Avg_FG_Pct': 'Opp_Avg_FG_Pct2', 'Comb_Loc': 'Comb_Loc2', 'Rest_Days': 'Rest_Days2',
    #                 'Avg_FG_Pct': 'Avg_FG_Pct2', 'Opp_Avg_FG_Pct_Fin': 'Opp_Avg_FG_Pct_Fin2', 'Pct_Margin_Fin': 'Pct_Margin_Fin2'}
    columns_dict = {'result': 'result2', 'Mean_Score': 'Mean_Score2', 'SMA_Opp_Score': 'SMA_Opp_Score2', 'EMA_Opp_Score': 'EMA_Opp_Score2', 'Last3Avg_Opp_Score': 'Last3Avg_Opp_Score2',
                'Prev_Opp_Score': 'Prev_Opp_Score2', 'SMA_Score': 'SMA_Score2', 'EMA_Score': 'EMA_Score2', 'Last3Avg_Score': 'Last3Avg_Score2', 'Prev_Score': 'Prev_Score2',
                'SMA_Mean_Score': 'SMA_Mean_Score2', 'EMA_Mean_Score': 'EMA_Mean_Score2', 'Last3Avg_Mean_Score': 'Last3Avg_Mean_Score2', 'Prev_Mean_Score': 'Prev_Mean_Score2', 
                'SMA_Result': 'SMA_Result2', 'EMA_Result': 'EMA_Result2', 'Last3Avg_Result': 'Last3Avg_Result2', 'Prev_Result': 'Prev_Result2', 'SMA_Pct_Margin': 'SMA_Pct_Margin2',
                'EMA_Pct_Margin': 'EMA_Pct_Margin2', 'Last3Avg_Pct_Margin': 'Last3Avg_Pct_Margin2', 'Prev_Pct_Margin': 'Prev_Pct_Margin2', 'Home': 'Home2', 'Away': 'Away2', 
                'Neutral': 'Neutral2', 'Date': 'Date2', 'Loc': 'Loc2', 'Opp_name': 'Opp_name2', 'Win': 'Win2', 'Opp_score': 'Opp_score2', 'Score': 'Score2', 'Threes': 'Threes2',
                'Opp_Threes': 'Opp_Threes2', 'W': 'W2', 'SMA_Threes': 'SMA_Threes2', 'EMA_Threes': 'EMA_Threes2', 'Last3Avg_Threes': 'Last3Avg_Threes2', 'Prev_Threes': 'Prev_Threes2',
                'SMA_Opp_Threes': 'SMA_Opp_Threes2', 'EMA_Opp_Threes': 'EMA_Opp_Threes2', 'Last3Avg_Opp_Threes': 'Last3Avg_Opp_Threes2', 'Prev_Opp_Threes': 'Prev_Opp_Threes2', 
                'Game_Number': 'Game_Number2', 'SMA_Comb_Loc': 'SMA_Comb_Loc2', 'EMA_Comb_Loc': 'EMA_Comb_Loc2', 'Last3Avg_Comb_Loc': 'Last3Avg_Comb_Loc2', 'Prev_Comb_Loc': 'Prev_Comb_Loc2',
                'SMA_Result_Fin': 'SMA_Result_Fin2', 'EMA_Result_Fin': 'EMA_Result_Fin2', 'Last3Avg_Result_Fin': 'Last3Avg_Result_Fin2', 'Prev_Result_Fin': 'Prev_Result_Fin2',
                'SMA_Opp_FG_Pct': 'SMA_Opp_FG_Pct2', 'EMA_Opp_FG_Pct': 'EMA_Opp_FG_Pct2', 'Last3Avg_Opp_FG_Pct': 'Last3Avg_Opp_FG_Pct2', 'Prev_Opp_FG_Pct': 'Prev_Opp_FG_Pct2',
                'Comb_Loc': 'Comb_Loc2', 'Rest_Days': 'Rest_Days2', 'SMA_FG_Pct': 'SMA_FG_Pct2', 'EMA_FG_Pct': 'EMA_FG_Pct2', 'Last3Avg_FG_Pct': 'Last3Avg_FG_Pct2', 'Prev_FG_Pct': 'Prev_FG_Pct2', 
                'SMA_Opp_FG_Pct_Fin': 'SMA_Opp_FG_Pct_Fin2', 'EMA_Opp_FG_Pct_Fin': 'EMA_Opp_FG_Pct_Fin2', 'Last3Avg_Opp_FG_Pct_Fin': 'Last3Avg_Opp_FG_Pct_Fin2',
                'Prev_Opp_FG_Pct_Fin': 'Prev_Opp_FG_Pct_Fin2', 'SMA_Pct_Margin_Fin': 'SMA_Pct_Margin_Fin2', 'EMA_Pct_Margin_Fin': 'EMA_Pct_Margin_Fin2', 'Last3Avg_Pct_Margin_Fin': 'Last3Avg_Pct_Margin_Fin2',
                'Prev_Pct_Margin_Fin': 'Prev_Pct_Margin_Fin2'}

    keys = list(columns_dict.keys())

    # loop through each dataframe (team)
    for ii, df in enumerate(dataframes):
        
        next_game = []
        
        next_game.append(df['SMA_Result_Fin'][0])

        # initialize a list that will represent the strength of schedule of the team entering the game
        # initialized with NaN because there would be no previous opponents entering the first game
        sos = [np.nan]
        sos_all = [np.nan] # uses final average result from each previous opponent
        sos_nc = [np.nan] # team's non-conference SOS

        # initialize a list that will represent the strength of schedule of the team's previous opponents
        # entering the game
        # initialized with NaN because there would be no previous opponents entering the first game
        prev_sos = [np.nan]
        prev_sos_all = [np.nan] # uses final average result from each previous opponent
        prev_sos_nc = [np.nan] # team's non-conference SOS
        
        # Non-conference game statistics
        avg_result_nc = [np.nan]
        nc_games = []
        opps_avg_result_nc = []

        # Initializes a list that will represent the strength of schedule of the opponent entering the game
        # This list is not initialized with NaN because the opponent may have already played a game before
        # the team's first game
        opp_sos =[]
        opp_sos_nc = []
        opp_prev_opp_sos = []

        # list that will contain data of opponent
        opp_data = []

        # names of all opponents
        opp_names = df['Opp_name']
        
        name_check = teams[ii]
        
        # Conference of team
        conf_check = df['Conference'][0]

        # loops through every game, specifically by enumerating the opponent name column
        for idx, name in enumerate(opp_names):
            
            date_check = df['Date'][idx]
            
            # will loop through each team's opponent prior to current game
            # first checks if idx is greater than 0 because there are no games
            # before the first game
            if idx > 0:
                
                # list of all previous opponents
                prev_opps = opp_names[:idx]

                # list that will store average win/loss results for each previous opponent
                avg_results = []
                sos_all_results = [] # stores the final average score for each previous opponent
                sos_nc_games = []
                sos_nc_games_fin = []

                # lists that will store average win/loss results for each previous opponent's previous opponents
                prev_prev_avg_results = []
                prev_sos_all_results = []
                prev_sos_nc_games = []
                fin_prev_sos_results = []

                # loop through each previous opponent
                for jj, prev_name in enumerate(prev_opps):
                    # previous opponent dataframe
                    prev_df = dataframes[team_dict[prev_name]]
                    
                    # Previous opponent conference
                    prev_opp_conf = prev_df['Conference'][0]
                    
                    len_prev = len(prev_df)

                    # loops through each game for previous opponent
                    for idx3, date2 in enumerate(prev_df['Date']):
                        if date2 >= date_check:
                            prev_avg_result = prev_df['SMA_Result'][idx3]
                            prev_opp_prev = prev_df['Opp_name'][:idx3]
                            break
                        elif idx3 == (len_prev - 1):
                            prev_avg_result = prev_df['SMA_Result_Fin'][0]
                            prev_opp_prev = prev_df['Opp_name']
                            break     

                    ## Checks to see if conferences are different
                    ## Only appends "nc" games if conferences are different
                    if conf_check != prev_opp_conf:
                        sos_nc_games.append(prev_avg_result)
                        sos_nc_games_fin.append(prev_df['SMA_Result_Fin'][0])

                    ## append strength of schedule data
                    avg_results.append(prev_avg_result)
                    sos_all_results.append(prev_df['SMA_Result_Fin'][0])

                    # loops through the previous opponents of the previous opponent
                    # to find the SOS of previous opponents
                    for prev_opp in prev_opp_prev:
                        prev2_df = dataframes[team_dict[prev_opp]]
                        
                        len_prev2 = len(prev2_df)

                        for kk, date_kk in enumerate(prev2_df['Date']):
                            if date_kk >= date_check:
                                #prev_prev_avg_result = prev2_df['result'][:kk].mean()
                                prev_prev_avg_result = prev2_df['SMA_Result'][kk]
                                break
                            elif kk == (len_prev2 - 1):
                                #prev_prev_avg_result = prev2_df['result'].mean()
                                prev_prev_avg_result = prev2_df['SMA_Result_Fin'][0]
                                break   

                        ## Checks to see if conferences are different
                        ## Only appends "nc" games if conferences are different
                        if prev_opp_conf != prev2_df['Conference'][0]:
                            prev_sos_nc_games.append(prev_prev_avg_result)

                        ## append strength of schedule data
                        prev_prev_avg_results.append(prev_prev_avg_result)
                        prev_sos_all_results.append(prev2_df['SMA_Result_Fin'][0])

                # take means of SOS data to provide singular value for each game
                prev_sos.append(np.array(prev_prev_avg_results).mean())
                prev_sos_all.append(np.array(prev_sos_all_results).mean())
                #prev_sos_nc.append(np.array(prev_sos_nc_games).mean())

                # take means of SOS data to provide singular value for each game
                sos.append(np.array(avg_results).mean())
                sos_all.append(np.array(sos_all_results).mean())
                sos_nc.append(np.array(sos_nc_games).mean())

            opp_df = dataframes[team_dict[name]] # opponent's dataframe
            
            # Opponent Conference
            opp_conf = opp_df['Conference'][0]
            opp_nc_games = []
            opp_avg_result_nc = [np.nan]
            
            # Adds NC results
            if conf_check != opp_conf:
                nc_games.append(df['Score'][idx] - df['Opp_score'][idx])
            if idx < (len(df) - 1):
                avg_result_nc.append(np.mean(nc_games))
            else:
                fin_avg_result_nc = np.mean(nc_games)
 

            # loops through each game in opponent's dataframe to find the matching date which
            # would represent the same game
            for idx2, (date, opp) in enumerate(zip(opp_df['Date'], opp_df['Opp_name'])):    # added the name check to account for few times where teams played twice in a day
                if (date == date_check) and (opp == name_check):
                    opp_data.append(opp_df.loc[idx2]) # appends data from opponent's game
                    break

            # will loop through each previous opponent of that game's opponent prior to current game
            # first checks if idx2 is greater than 0 because there are no games
            # before the first game
            if idx2 > 0:

                # list of opponent's previous opponents before current game
                opp_prev_opps = opp_df['Opp_name'][:idx2]

                # list that will store average win/loss results for each opponent of current game's opponent
                opp_avg_results = []
                opp_sos_nc_games = []
                opp_prev_opp_prev_results = []

                # loop through each previous opponent of opponent
                for kk, opp_prev_name in enumerate(opp_prev_opps):
                    
                    # previous opponent's opponent dataframe
                    opp_prev_df = dataframes[team_dict[opp_prev_name]]
                    
                    # Adds NC2 results
                    if opp_conf != opp_prev_df['Conference'][0]:
                        opp_nc_games.append(opp_df['Score'][kk] - opp_df['Opp_score'][kk])
                    
                    len_opp_prev = len(opp_prev_df)
                    
                    # checks the dates of games of the opponent's previous opponent 
                    # will look at games prior to current game date
                    for idx4, date3 in enumerate(opp_prev_df['Date']):
                        if date3 >= date_check:
                            opp_prev_avg_result = opp_prev_df['SMA_Result'][idx4]
                            opp_prev_opp_prev = opp_prev_df['Opp_name'][:idx4] # opponent's opponent's previous opponents 
                            break
                        elif idx4 == (len_opp_prev - 1):
                            opp_prev_avg_result = opp_prev_df['SMA_Result_Fin'][0]
                            opp_prev_opp_prev = opp_prev_df['Opp_name']
                            break
                    
                    opp_prev_opp_prev_result = []
                    fin_prev_sos = []
                        
                    for name2 in opp_prev_opp_prev:

                        opp_prev_opp_prev_df = dataframes[team_dict[name2]]
                        
                        len_opp_prev_opp_prev = len(opp_prev_opp_prev_df)

                        for ll, date_ll in enumerate(opp_prev_opp_prev_df['Date']):

                            if date_ll >= date_check:
                                opp_prev_opp_prev_result.append(opp_prev_opp_prev_df['SMA_Result'][ll])
                                break
                            elif ll == (len_opp_prev_opp_prev - 1):
                                opp_prev_opp_prev_result.append(opp_prev_opp_prev_df['SMA_Result_Fin'][0])

                    opp_prev_opp_prev_results.append(np.mean(opp_prev_opp_prev_result))
                    
                    # appends the average result of opponent's opponent's previous games
                    opp_avg_results.append(opp_prev_avg_result)
                    
                    # Only keep games where conferences are different
                    if opp_conf != opp_prev_df['Conference'][0]:
                        opp_sos_nc_games.append(opp_prev_avg_result)

                # appends the average results of all opponent's previous opponents previous games
                opp_sos.append(np.array(opp_avg_results).mean())
                opp_sos_nc.append(np.array(opp_sos_nc_games).mean())
                opp_prev_opp_sos.append(np.mean(opp_prev_opp_prev_results))
                opps_avg_result_nc.append(np.mean(opp_nc_games))
            else:
                opp_sos.append(np.nan)
                opp_sos_nc.append(np.nan)
                opp_prev_opp_sos.append(np.nan)
                opps_avg_result_nc.append(np.nan)

        # Converts the opponents' data to a dataframe and then renames the columns to avoid duplication of column names and 
        # provide clarity as to what the different columns represent
        opp_data_df = pd.DataFrame(opp_data, index=list(range(len(opp_data))))
        
        keep_dict = {}
        cols = opp_data_df.columns
        # Only convert columns that are in the opp_data_df
        for key in keys:   
            if key in cols:
                keep_dict[key] = columns_dict[key]
        
        opp_data_df = opp_data_df.rename(columns=keep_dict)[list(keep_dict.values())]

        if len(nc_games) > 0:
            next_game.append(fin_avg_result_nc)
        else:
            next_game.append(0)
        next_game.append(np.mean(sos_all_results + [opp_df['SMA_Result_Fin'][0]]))
        if len(nc_games) > 0:
            next_game.append(np.mean(sos_nc_games_fin))
        else:
            next_game.append(0)
        next_game.append(df['SMA_Opp_FG_Pct_Fin'][0])
        next_game.append(df['SMA_Pct_Margin_Fin'][0])
        next_game.append((df['Comb_Loc'].mean() ** 2) * (len(df) + 1))
        next_game.append(df['Mean_Score'].mean())
        next_game.append(opp_names)
        
        # Non-Conference Games
        df['Avg_Result_NC'] = avg_result_nc
        df['Avg_Result_NC2'] = opps_avg_result_nc
        
        # Converts the strength of schedule lists to dataframes that can be appended
        sos_df = pd.DataFrame()
        sos_df['SOS'] = sos        # Strength of schedule using all previous opponents
        sos_df['SOS_NC'] = sos_nc  # Strength of schedule using only non-conference opponentes (first 10 opponents)
        sos_df['Opp_SOS'] = opp_sos # Opponent's SOS using all previous opponents
        sos_df['Opp_SOS_NC'] = opp_sos_nc
        sos_df['Prev_SOS'] = prev_sos
        sos_df['SOS_Fin'] = [sos_df['SOS'][len(sos_df) - 1]] * len(sos_df) # column that contains the final SOS
        sos_df['SOS_All'] = sos_all
        sos_df['SOS_All_Fin'] = [sos_df['SOS_All'][len(sos_df) - 1]] * len(sos_df)
        sos_df['Opp_Prev_SOS'] = opp_prev_opp_sos

        opp_data_df['Scoring_Pace_Diff'] = (df['SMA_Mean_Score'] - opp_data_df['SMA_Mean_Score2'])/2

        # Looks at the relative percentage of points scored and given up by 3-pointers and multiplies that
        # by the factor of the oppponent
        team_score = df['SMA_Threes'] * opp_data_df['SMA_Opp_Threes2']
        opp_score = df['SMA_Opp_Threes'] * opp_data_df['SMA_Threes2']
        opp_data_df['Matchup_Comp'] = team_score / (abs(team_score) ** 0.5) 
        opp_data_df['Opp_Matchup_Comp'] = opp_score / (abs(opp_score) ** 0.5) 

        # combines original dataframe with data from game opponents, SOS of team, and SOS of opponents
        combined_df.append(pd.concat([df, opp_data_df, sos_df], axis=1))

        # saves dataframe
        combined_df[ii].to_csv('./Team_Dataframes/' + year + '/Team_Combined/' + teams[ii] + '.csv', index = False)
        
        # Save dataframe to pickle file
        with open('./Team_Dataframes/' + year + '/Team_Combined/' + teams[ii] + '.pickle', 'wb') as f:
            pickle.dump(combined_df[ii], f)

        if ii%20 == 0:
            print(ii)
            
        next_games.append(next_game)
    
    with open('next_g_' + year + '.pickle', 'wb') as f:
        pickle.dump(next_games, f)
    
    pd.concat((combined_df)).to_csv('./Team_Dataframes/' + year + 'Combined.csv', index=False)
    
    with open('./Team_Dataframes/' + year + 'Combined.pickle', 'wb') as f:
        pickle.dump(pd.concat((combined_df)), f)
    
    # list of concatenated dataframes
    return combined_df, next_games


def print_null(y):
    print('NULL STATISTICS')
    print('Standard Error:', np.std(y) / np.sqrt(len(y)))
    print('Standard Deviation:', np.std(y))
    print('Mean-squared Error:', mean_squared_error(y, [0]*len(y)))
    print('Percent Win/Loss Prediction: 50 %')


def home_null(data):
    """
        Function takes in a dataframe and returns the percent accuracy of choosing the winner
        if the home team is chosen to win for each game.
    """
    
    # always prediction home team wins
    home_preds = np.where(data['Away'] == 0, 1, 0)
    
    # initialize correct predictions
    correct = 0
    
    # correct predictions will be 0 or 2 because a if both the prediction and 
    # win result are 0, that means it accurately predicted a loss, and if both
    # the prediction and actual are 1, that means it accurately predicted a win
    for pred, win in zip(home_preds, data['Win']):

        if (pred + win) == 0 or (pred + win) == 2:
            correct += 1

    return (correct/len(home_preds)) * 100



def get_linear_stats(X, y, features, opp_features, n_runs=10, n_folds=10, print_out=True, pickled=False):
    """
        Takes in X and y data and returns Linear Model statistics.
        Returns Standard Error, Standard Deviation, Mean Squared Error,
        Cross-Validation Score, and statsmodels OLS results.
        
        n_runs is the number of training runs to perform. An increase in n_runs
        provides more stable results.
        
        n_folds is the number of folds to create in the cross-validation test
    """
    
    # Initialized Linear Regression model
    lr = LinearRegression(fit_intercept=False)
    
    # number for features in model
    num_features = len(features)
    
    # initializes statistics that will be averaged
    MSEs = []
    SE = []
    stds = []
    scores = []
    pct_acc = []
    cv = []
    coefs = []
    
    # Number of training runs
    # Increase in runs stabilizes results
    for ii in range(n_runs):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        
        # Uses only features
        X_train_tm = X_train[features]
        X_test_tm = X_test[features]
        
        lr.fit(X_train_tm, y_train)

        y_pred1 = lr.predict(X_test_tm)
        
        # Uses only opp_features
        X_train_opp = X_train[opp_features]
        X_test_opp = X_test[opp_features]
        
        y_pred2 =  -1 * lr.predict(X_test_opp)
        
        # Difference between predicted results and test data
        y_pred = np.mean([y_pred1, y_pred2], axis = 0)
        resid = y_pred - y_test
        
        # Append statistics
        SE.append(np.std(resid, ddof=num_features) / np.sqrt(np.size(resid)))
        stds.append(np.std(resid, ddof=num_features))
        MSEs.append(mean_squared_error(y_test, y_pred))
        scores.append(lr.score(X_test_tm, y_test))
        
         # Show percent accuracy of predicting win/loss
        num_correct = 0
        for pred, act in zip(y_pred, y_test):
            if pred*act > 0:
                num_correct += 1
        pct_acc.append(num_correct/len(y_test))
        
        coefs.append(lr.coef_)
        
        cv.append(cross_val_score(LinearRegression(fit_intercept=False), X[features], y, cv=KFold(n_folds, shuffle=True)).mean())
   
    
    # Develop OLS model
    model = sm.OLS(y_train, X_train_tm, missing='drop') # sm.add_constant(X)
    results = model.fit()
    
    # Pickle Linear Model Data
    if pickled:
        with open('lin_reg_coef.pickle', 'wb') as f:
            pickle.dump(np.mean(coefs, axis=0), f)
        with open('lin_mse.pickle', 'wb') as f:
            pickle.dump(np.mean(MSEs), f)
    
    if print_out:
        print('\n\nLINEAR STATISTICS')
        print('Standard Error:', np.mean(SE))
        print('Standard Deviation:', np.mean(stds))
        print('Mean-squared Error:', np.mean(MSEs))
        print('Percent Win/Loss Prediction: ', np.mean(pct_acc)*100, '%')
        print('Cross-validation score:', np.mean(cv))
        print(results.summary())
    
    return np.mean(SE), np.mean(stds), np.mean(MSEs), np.mean(cv), results, np.mean(pct_acc)*100, np.mean(coefs, axis=0), lr



def get_logistic_stats(X, y, features, opp_features, n_runs=10, n_folds=10, print_out=True, pickled=False):
    """
        Takes in X and y data and returns Logistic Model statistics.
        Returns Percent Accuracy.
        
        n_runs is the number of training runs to perform. An increase in n_runs
        provides more stable results.
        
        n_folds is the number of folds to create in the cross-validation test
    """
    
    # Initialized Logistic Regression model
    lr1 = LogisticRegression(max_iter=10000)
    lr2 = LogisticRegression(max_iter=10000)
    
    # initializes statistics that will be averaged
    cv = []
    coefs = []
    
    sc = StandardScaler()
    X_cv = sc.fit_transform(X)
    
    # Number of training runs
    # Increase in runs stabilizes results
    for ii in range(n_runs):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        
        sc = StandardScaler()
        X_train_tm = sc.fit_transform(X_train[features])
        X_test_tm = sc.transform(X_test[features])
        
        lr1.fit(X_train_tm, y_train)

        # Uses only opp_features
        X_train_opp = sc.transform(X_train[opp_features])
        X_test_opp = sc.transform(X_test[opp_features])
        
        lr2.fit(X_train_opp, y_train)
        
        avg_coefs = np.mean([lr1.coef_, -1 * lr2.coef_], axis=0)

        coefs.append(avg_coefs)
        
        cv.append(cross_val_score(LogisticRegression(max_iter=10000), X_cv, y, cv=StratifiedKFold(n_folds, shuffle=True), scoring='accuracy').mean())
    
    # Pickle Logistic Model Data
    if pickled:
        with open('log_reg_coef.pickle', 'wb') as f:
            pickle.dump(np.mean(coefs, axis=0), f)
        with open('log_reg_sc.pickle', 'wb') as f:
            pickle.dump(sc, f)

    if print_out:
        print('\n\nLOGISITC STATISTICS')
        print('Score: ', np.mean(cv)*100, '%')
        #print('Average Coefficients:', np.mean(coefs, axis=0))
        logs = lr1.predict(sc.transform(X[features]))
        print('Classification Report\n', classification_report(y, logs))
    
    return np.mean(cv)*100, np.mean(coefs, axis=0), lr1, sc
 

def log_best_fit(X, X_opp, sc, avg_log_coefs):

    predictions = []
    probabilities = []
    
    logit1 = np.exp(sc.transform(X).dot(np.transpose(avg_log_coefs)))
    logit2 = np.exp(sc.transform(X_opp).dot(np.transpose(avg_log_coefs)))
    prob1 = logit1 / (1 + logit1)
    prob2 = logit2 / (1 + logit2)
    
    probs = (prob1 + (1 - prob2)) / 2
    
    for prob in probs:
        
        if prob > 0.5:
            prediction = 1
        else:
            prediction = 0
        
        predictions.append(prediction)
        probabilities.append(prob[0])
    
    return predictions, probabilities
 
    
def get_rf_stats(X, y, y2, features, opp_features, class_n_estimators=188, class_max_depth=9, class_max_features=5,
                 regr_n_estimators=188, regr_max_depth=9, regr_max_features=5, n_runs=2, print_out=True, pickled=False):
    """
        Takes in X and y data and returns RandomForest model statistics for both Regression and Classification.
        
        Optional tuning parameters (n_estimators, max_depth, and max_features) can be passed in for both regression and classification.
        
        Returns both models as well as MSE for regression and accuracy for classification.
    """
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    
    # Uses only features
    X_train_tm = X_train[features]
    X_test_tm = X_test[features]

    # initialize RandomForest Regressor with hyperparameters
    rf_regr = RandomForestRegressor(n_estimators=regr_n_estimators, max_depth=regr_max_depth, max_features=regr_max_features)

    # Fit on the training data and predict on the test data
    rf_regr.fit(X_train_tm, y_train)
    y_pred1 = rf_regr.predict(X[features])
    print(y_pred1[:50])
    
    # Uses only opp_features
    X_train_opp = X_train[opp_features]
    X_test_opp = X_test[opp_features]
        
    y_pred2 =  -1 * rf_regr.predict(X[opp_features])
    print(y_pred2[:50])
    
    # Difference between predicted results and test data
    y_pred = np.mean([y_pred1, y_pred2], axis = 0)
    
    # split data into training and testing sets
    X_train, X_test, y2_train, y2_test = train_test_split(X, y2, stratify=y2)
    
    # Scale data for use in RandomForest classification   
    sc = StandardScaler()
    X_train_tm = sc.fit_transform(X_train[features])
    X_test_tm = sc.transform(X_test[features])

    # initialize RandomForest Classifier with hyperparameters
    rf_class = RandomForestClassifier(n_estimators=class_n_estimators, max_depth=class_max_depth, max_features=class_max_features)

    # Fit on the training data and predict on the test data
    rf_class.fit(X_train_tm, y2_train)
    y2_pred1 = rf_class.predict(X[features])
    print(y2_pred1[:50])
    y2_pred2 =  (-1 * rf_class.predict(X[opp_features])) + 1
    print(y2_pred2[:50], '\n')
    y2_pred = np.mean([y2_pred1, y2_pred2], axis = 0)
    
    # Grab the mse (regression) and accuracy (classification) metrics from the two models
    #regr_mse = mean_squared_error(y_test, y_pred)
    #class_acc = accuracy_score(y2_test, y2_pred)
    
    mses = []
    accs = []
    Xsc = sc.fit_transform(X[features])
    
    for ii in range(n_runs):
        
        # get cv mean squared error and append to list - eventually will take mean of list
        mses.append(-cross_val_score(RandomForestRegressor(n_estimators=regr_n_estimators, max_depth=regr_max_depth, max_features=regr_max_features), \
                                     X[features], y, cv=KFold(10, shuffle=True), scoring='neg_mean_squared_error').mean())

        
        # get cv accuracy and append to list - eventually will take mean of list
        accs.append(cross_val_score(RandomForestClassifier(n_estimators=class_n_estimators, max_depth=class_max_depth, max_features=class_max_features), \
                                     Xsc, y2, cv=StratifiedKFold(10, shuffle=True), scoring='accuracy').mean())


        #### FOR USE WHEN TUNING PARAMETERS
        #params = {'n_estimators': range(130, 141), 'max_depth': range(5, 12), 'max_features': range(7,12)}
        #cvtree = GridSearchCV(rf_class, params)
        #cvtree.fit(X, y)
    
    # Save RF Values
    if pickled:   
        with open('rf_mse.pickle', 'wb') as f:
            pickle.dump(np.mean(mses), f)
        with open('rf_class.pickle', 'wb') as f:
            pickle.dump(rf_class, f)
        with open('rf_reg.pickle', 'wb') as f:
            pickle.dump(rf_regr, f)
    
    if print_out:
        print('\n\nRANDOM FOREST STATISTICS')
        print('Points Spread MSE (Regressor):', np.mean(mses))
        print('Percent Win/Loss Prediction (Classifier): ', np.mean(accs) * 100, '%')
        print('Classification Report\n', classification_report(y2, rf_class.predict(Xsc)))
    
    # return statistics
    return np.mean(mses), np.mean(accs) * 100, rf_regr, rf_class, y_pred, y2_pred


def combine_models(X, y, y2, features, opp_features, MSE, rf_mse, class_n_estimators=188, class_max_depth=9, class_max_features=5, 
                   regr_n_estimators=188, regr_max_depth=9, regr_max_features=5, n_runs=10, print_out=True, pickled=False):
    
    lin_weight = 1 - (MSE/ (MSE+rf_mse))
    rf_regr_weight = 1 - (rf_mse / (MSE+rf_mse))
    
    av_mses = []
    av_accs = []
    y_tests = []
    y2_tests = []
    y_preds = []
    lin_y2_preds = []
    y2_preds = []
    y2_probs = []
    log_y2_probs = []
    rf_y2_probs = []

    for ii in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

        # Uses only features
        X_train_tm = X_train[features]
        X_test_tm = X_test[features]

        # initialize RandomForest Regressor with hyperparameters
        rf_regr_pred = RandomForestRegressor(n_estimators=regr_n_estimators, max_depth=regr_max_depth, max_features=regr_max_features)

        # Fit on the training data and predict on the test data
        rf_regr_pred.fit(X_train_tm, y_train)
        y_pred1 = rf_regr_pred.predict(X_test_tm)

        # Uses only opp_features
        X_train_opp = X_train[opp_features]
        X_test_opp = X_test[opp_features] 
        y_pred2 =  -1 * rf_regr_pred.predict(X_test_opp)

        # Difference between predicted results and test data
        rf_y_pred = np.mean([y_pred1, y_pred2], axis=0)
        #rf_ys.append(rf_y_pred)

        # Initialized Linear Regression model
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_train_tm, y_train)
        y_pred1 = lr.predict(X_test_tm)
        y_pred2 =  -1 * lr.predict(X_test_opp)
        lin_y_pred = np.mean([y_pred1, y_pred2], axis = 0)
        #lin_ys.append(lin_y_pred)
        
        # Take average of regression ys
        #y_av = np.mean([rf_y_pred, lin_y_pred], axis=0)
        y_av = (rf_y_pred * rf_regr_weight) + (lin_weight * lin_y_pred)

        Xsc_train, Xsc_test, y2_train, y2_test = train_test_split(X, y2, stratify=y2)
        # Uses only features
        Xsc_train_tm = Xsc_train[features]
        Xsc_test_tm = Xsc_test[features]
        Xsc_train_opp = Xsc_train[opp_features]
        Xsc_test_opp = Xsc_test[opp_features] 

        # Linear Win/Loss Predictions
        y_pred1 = lr.predict(Xsc_test_tm)
        y_pred2 =  -1 * lr.predict(Xsc_test_opp)
        lin_y2_pred = np.mean([y_pred1, y_pred2], axis = 0)
        lin_y2_pred = np.where(lin_y2_pred > 0, 1, 0)

        sc = StandardScaler()
        Xsc_train_tm = sc.fit_transform(Xsc_train_tm)
        Xsc_test_tm = sc.transform(Xsc_test_tm)

        lr = LogisticRegression(max_iter=10000)
        lr.fit(Xsc_train_tm, y2_train)
        y2_pred1 = lr.predict(Xsc_test_tm)
        y2_pred2 = lr.predict(Xsc_test_opp)
        y2_prob1 = lr.predict_proba(Xsc_test_tm)[:, 1]
        y2_prob2 = lr.predict_proba(Xsc_test_opp)[:, 0]
        log_y2_prob = (y2_prob1 + y2_prob2) / 2
        log_y2_probs.append(log_y2_prob)

        log_y2_pred = np.where(log_y2_prob > 0.5, 1, 0)

        rf_class_pred = RandomForestClassifier(n_estimators=class_n_estimators, max_depth=class_max_depth, max_features=class_max_features)
        rf_class_pred.fit(Xsc_train_tm, y2_train)
        rf_y2_pred1 = rf_class_pred.predict(Xsc_test_tm)
        rf_y2_pred2 = rf_class_pred.predict(Xsc_test_opp)
        rf_y2_prob1 = rf_class_pred.predict_proba(Xsc_test_tm)[:, 1]
        rf_y2_prob2 = rf_class_pred.predict_proba(Xsc_test_opp)[:, 0]
        rf_y2_prob = (rf_y2_prob1 + rf_y2_prob2) / 2

        rf_y2_pred = np.where(rf_y2_prob > 0.5, 1, 0)

        y2_av = np.mean([rf_y2_pred, lin_y2_pred, log_y2_pred], axis=0)
        y2_av = np.where(y2_av > 0.5, 1, 0)
        y2_prob = np.mean([log_y2_prob, rf_y2_prob], axis=0)
        rf_y2_probs.append(rf_y2_prob)

        y_tests += list(y_test)
        y2_tests += list(y2_test)
        y_preds += list(y_av)
        y2_preds += list(y2_av)
        y2_probs += list(y2_prob)
        lin_y2_preds += list(lin_y2_pred)

        mse = mean_squared_error(y_test, y_av)
        av_mses.append(mse) 

        av_accs.append(accuracy_score(y2_test, y2_av))

        if ii%20 == 0:
            print(ii)
    
    # Save Average MSE and Accuracy
    if pickled:
         with open('av_mse.pickle', 'wb') as f:
             pickle.dump(np.mean(av_mses), f)
         with open('acc_av.pickle', 'wb') as f:
             pickle.dump(np.mean(av_accs), f)
    
    if print_out:
        print('\n\nModel Combination MSE:', np.mean(av_mses))
        print('Model Combination Accuracy:', np.mean(av_accs))
    
    return np.mean(av_mses), np.mean(av_accs), y_tests, y2_tests, y_preds, lin_y2_preds, y2_preds, y2_probs, log_y2_probs, rf_y2_probs


def create_pct_odds(y2_probs, y2_tests, val_range, pickled=False):
    
    rounded_y2_probs = pd.DataFrame(np.round(y2_probs, 2), columns=['y_prob'])
    y2_tests = pd.DataFrame(y2_tests, columns=['y2_test'])

    preds_and_results = pd.concat((rounded_y2_probs, y2_tests), axis=1)
    
    vals = np.round(val_range, 2)

    dicts = {val: [] for val in vals}
    
    for ii, result in enumerate(preds_and_results['y2_test']):
        prediction = preds_and_results['y_prob'].iloc[ii]
        if prediction in vals:
            dicts[prediction].append(result)
    
    win_pcts = []
    
    for val in vals:
        wins = 0
        for val2 in dicts[val]:
            if val2 > 0:
                wins += 1
        win_pcts.append(wins/len(dicts[val]))
    
    # Pickle Distribution
    if pickled:
        with open('pct_dists.pickle', 'wb') as f:
            pickle.dump(win_pcts, f)
        with open('pct_val_range.pickle', 'wb') as f:
            pickle.dump(val_range, f)
    
    return win_pcts



def create_pts_spread_odds(y_preds, y_tests, val_range, pickled=False):
    
    rounded_y_preds = pd.DataFrame(np.round(y_preds, 0), columns = ['y_pred'])
    y_tests = pd.DataFrame(y_tests, columns = ['y_test'])

    preds_and_results = pd.concat((rounded_y_preds, y_tests), axis=1)
    
    vals = np.round(val_range, 0)

    dicts = {val: [] for val in vals}

    for ii, result in enumerate(preds_and_results['y_test']):
        prediction = preds_and_results['y_pred'].iloc[ii]
        if prediction in val_range:
            dicts[prediction].append(result)

    #plt.hist(dicts[21], bins=20);

    prob_dists = []

    for jj in vals:

        probs = []

        for kk in vals:

            count = 0

            len_dict = len(dicts[kk])

            if len_dict > 0:
                for val in dicts[kk]:

                    if val < jj:
                        count += 1

                probs.append(count/len_dict)
        
        prob_dists.append(probs)
    
    # Pickle Distribution
    if pickled:
        with open('prob_dists.pickle', 'wb') as f:
            pickle.dump(prob_dists, f)
        with open('spread_val_range.pickle', 'wb') as f:
            pickle.dump(val_range, f)
    
    return prob_dists


def create_pct_models(val_range, win_pct, pickled=False):
    xs = np.round(val_range, 2)
    
    model = np.polyfit(xs, win_pct, 5)
     
    # Pickle distribution models      
    if pickled:
        with open('pct_model.pickle', 'wb') as f:
            pickle.dump(model, f)
    
    return model, xs


def create_models(val_range, prob_dists, pickled=False):
    xs = np.round(val_range, 0)
    models = []
    for x2 in xs:
        coefs_fit = np.polyfit(xs, prob_dists[int(x2)-int(xs[0])], 7)
        models.append(coefs_fit)
     
    # Pickle distribution models      
    if pickled:
        with open('dist_models.pickle', 'wb') as f:
            pickle.dump(models, f)
    
    return models, xs



def get_pickles(year):

    with open('log_reg_coef.pickle', 'rb') as f:
        log_reg_coef = pickle.load(f)   
    with open('log_reg_sc.pickle', 'rb') as f:
        sc = pickle.load(f)
    with open('lin_reg_coef.pickle', 'rb') as f:
        lin_reg_coef = pickle.load(f)    
    with open('rf_mse.pickle', 'rb') as f:
        rf_mse = pickle.load(f)
    with open('lin_mse.pickle', 'rb') as f:
        lin_mse = pickle.load(f)    
    with open('av_mse.pickle', 'rb') as f:
        av_mse = pickle.load(f)    
    with open('rf_class.pickle', 'rb') as f:
        rf_class = pickle.load(f)
    with open('rf_reg.pickle', 'rb') as f:
        rf_reg = pickle.load(f)        
    with open('dist_models.pickle', 'rb') as f:
        dist_models = pickle.load(f)
    with open('val_range.pickle', 'rb') as f:
        val_range = pickle.load(f)
    with open('pct_model.pickle', 'rb') as f:
        pct_model = pickle.load(f)
    with open('pct_dists.pickle', 'rb') as f:
        pct_dists = pickle.load(f)
    with open('pct_val_range.pickle', 'rb') as f:
        pct_val_range = pickle.load(f)
    with open('next_g_' + year + '.pickle', 'rb') as f:
        next_g = pickle.load(f)
        
    return log_reg_coef, sc, lin_reg_coef, rf_mse, lin_mse, av_mse, rf_class, rf_reg, dist_models, val_range, pct_model, pct_dists, pct_val_range, next_g



def show_confidence(final_pts_spread, final_probs, dist_models, pct_model, val_range, pct_val_range, matchups):
    for jj, (spread, prob) in enumerate(zip(final_pts_spread, final_probs)):

        confidence, win_pct = get_confidence(dist_models, spread, val_range, pct_model, prob, pct_val_range)
        print(matchups[jj][0], spread)
        print(pd.DataFrame(confidence))
        print(win_pct)
        # Look at pct chance of winning
        #prob_win = ((2/3) * final_probs[jj]) + ((1/3) * confidence[25][0])
        prob_win = ((2/3) * win_pct) + ((1/3) * confidence[25][0])

        if prob_win > 0.5:
            odds = -(100*prob_win)/(1-prob_win)
        else:
            odds = (100/prob_win) - 100

        print(matchups[jj][0], prob_win, odds)
        print('\n\n')

        # Start with Team 2
        confidence, win_pct = get_confidence(dist_models, -spread, val_range, pct_model, 1-prob, pct_val_range)
        print(matchups[jj][1], -spread)
        print(pd.DataFrame(confidence))
        print(win_pct)
        # Look at pct chance of winning
        prob_win = ((2/3) * (win_pct)) + ((1/3) * confidence[25][0])
        if prob_win > 0.5:
            odds = -(100*prob_win)/(1-prob_win)
        else:
            odds = (100/prob_win) - 100

        print(matchups[jj][1], prob_win, odds)
        print('\n\n')



def get_next_games(next_g, combined_df, team_dict):
    """
        Takes in list of next game data. Loops through opponents and grabs the previous opponents
        of all opponents. Takes Final average score of all opponents' previous opponents to calculate
        Prev_SOS. Returns entire next game list
    """
    
    # loop through each team in list
    for next_game in next_g:
        
        # grab opponents names which is in last column
        prev_opps = next_game[-1]
        scores = []
        
        # loop through each opponent
        for team in prev_opps:
            
            # dataframe of opponent
            df = combined_df[team_dict[team]]
            
            # loop through each of opponent's previous opponents
            for team2 in df['Opp_name']:
                
                # opponent's previous opponent dataframe
                df2 = combined_df[team_dict[team2]]
                
                # append final average score
                scores.append(df2['Avg_Result_Fin'][0])

        next_game.append(np.mean(scores))
        
    return next_g


def get_rankings(teams, next_g, team_dict, lin_reg_coef, lin_mse, rf_mse, rf_reg, print_out):

    rankings = [teams[0]]

    for hh in range(1, len(teams)):
        # Team to be ranked
        team = teams[hh]
        if hh%20 == 0:
            print(hh)
        for gg in range(len(rankings)):

            matchups = [[team, rankings[gg], 2]]

            matchup_vals = create_matchups(matchups, next_g, team_dict)

            lin_preds = lin_regr_predict(lin_reg_coef, matchup_vals, matchups, print_out=print_out)
            rf_reg_preds = rf_regr_predict(rf_reg, matchup_vals, matchups, print_out=print_out)

            lin_weight = 1 - (lin_mse/ (lin_mse+rf_mse))
            rf_regr_weight = 1 - (rf_mse / (lin_mse+rf_mse))

            final_pts_spread = [(lin_weight * lin_pred) + (rf_regr_weight * rf_reg_pred[0]) for lin_pred, rf_reg_pred in zip(lin_preds, rf_reg_preds)]
            if final_pts_spread[0] > 0:
                rankings = rankings[:gg] + [team] + rankings[gg:]
                break

            if gg == (len(rankings) - 1):
                rankings += [team]
                
    return rankings



def create_matchups(matchups, next_g, team_dict):

    matchup_vals = []

    for matchup in matchups:
        team1 = matchup[0]
        vals1 = next_g[team_dict[team1]]

        team2 = matchup[1]
        vals2 = next_g[team_dict[team2]]

        if matchup[2] == 1:
            team1_home = 1
            team1_away = 0
            team2_home = 0
            team2_away = 1
        elif matchup[2] == 0:
            team1_home = 0
            team1_away = 1
            team2_home = 1
            team2_away = 0
        else:
            team1_home = 0
            team1_away = 0
            team2_home = 0
            team2_away = 0

        team1_vals = [vals1[0], vals2[0], vals1[1], vals2[1], team1_home, team1_away, vals1[2], vals2[2], vals1[-1], vals2[-1], vals1[3], vals2[3],
                      vals1[4], vals2[4], vals1[5], vals2[5], vals1[6], vals2[6], (vals1[7] - vals2[7])/2]
        team2_vals = [vals2[0], vals1[0], vals2[1], vals1[1], team2_home, team2_away, vals2[2], vals1[2], vals2[-1], vals1[-1], vals2[3], vals1[3],
                      vals2[4], vals1[4], vals2[5], vals1[5], vals2[6], vals1[6], (vals2[7] - vals1[7])/2]
        matchup_vals.append([team1_vals, team2_vals])
        
    return matchup_vals



def regr_predict(models, games, matchups):
    """
        Takes in list of predictive regression models and data to make predictions upon.
        
        Prints out predictions for each model passed in.
    """
    for ii, game in enumerate(games):
        # loop through each of the models passed
        for model in models:

            # make a prediction based on the model
            prediction1 = model.predict([game[0]])
            prediction2 = -1 * model.predict([game[1]])
            prediction = (prediction1 + prediction2) / 2

            # print model and prediction
            print(f"{model} model points spread: {matchups[ii][0]} {round(prediction[0], 4)} to {matchups[ii][1]} ({prediction1, prediction2})")
        print()
        


def class_predict(models, games, sc, matchups):
    """
        Takes in list of predictive classification models and data to make predictions upon.
        
        Prints out predictions for each model passed in.
    """
    for ii, game in enumerate(games):
        # loop through each of the models passed
        for model in models:

            # make a prediction based on the model
            prediction1 = model.predict(sc.transform([game[0]]))
            prediction2 = model.predict(sc.transform([game[1]]))
            prediction = prediction1 + prediction2
            probability1 = model.predict_proba(sc.transform([game[0]]))[0]
            probability2 = model.predict_proba(sc.transform([game[1]]))[0]
            print(prediction1, probability1, prediction2, probability2)

            # print model and prediction
            if prediction == 1:
                if prediction1 == 1:
                    probability = (probability1[1] + probability2[0]) / 2
                    print(f"{model} model prediction: {matchups[ii][0]} Win ({round(probability * 100, 4)}% probability)")
                else:
                    probability = (probability1[0] + probability2[1]) / 2
                    print(f"{model} model prediction: {matchups[ii][0]} Loss ({round(probability * 100, 4)}% probability)")
            else:
                print(f"Undeterminded Winner: (Win probability: {(probability1[1] + probability2[0]) / 2})")
        print()



def lin_regr_predict(avg_lin_coef, games, matchups, print_out=True):
    """
        Takes in list of predictive regression models and data to make predictions upon.
        
        Prints out predictions for each model passed in.
    """
    predictions = []
    
    for ii, game in enumerate(games):
        # loop through each of the models passed

        # make a prediction based on the model
        prediction1 = np.array(game[0]).dot(avg_lin_coef)
        prediction2 = -1 * np.array(game[1]).dot(avg_lin_coef)
        prediction = (prediction1 + prediction2) / 2
        
        if print_out:
            # print model and prediction
            print(f"Linear Regression model points spread: {matchups[ii][0]} {round(prediction, 4)} to {matchups[ii][1]} ({prediction1, prediction2})")
            print()
        
        predictions.append(prediction)
        
    return predictions
 

 
def log_regr_predict(avg_log_coef, games, sc, matchups, print_out=True):
    """
        Takes in list of predictive regression models and data to make predictions upon.
        
        Prints out predictions for each model passed in.
    """
    predictions = []
    probabilities = []
    
    for ii, game in enumerate(games):
        # loop through each of the models passed

        # make a prediction based on the model
        logit1 = np.exp(sc.transform([game[0]]).dot(np.transpose(avg_log_coef))[0][0])
        probability1 = (logit1) / (1 + logit1)
        logit2 = np.exp(sc.transform([game[1]]).dot(np.transpose(avg_log_coef))[0][0])
        probability2 = (logit2) / (1 + logit2)

        # print model and prediction
        probability = (probability1 + (1 - probability2)) / 2
        if probability > 0.5:
            prediction = 1
            if print_out:
                print(f"Logistic Regression model prediction: {matchups[ii][0]} Win ({round(probability * 100, 4)}% probability)")
        else:
            probability = probability
            prediction = 0
            if print_out:
                print(f"Logistic Regression model prediction: {matchups[ii][0]} Loss ({round(probability * 100, 4)}% probability)")
        if print_out:
            print()
        
        predictions.append(prediction)
        probabilities.append(probability)
    
    return predictions, probabilities
  

  
def rf_regr_predict(rf_reg, games, matchups, print_out=True):
    """
        Takes in list of predictive regression models and data to make predictions upon.
        
        Prints out predictions for each model passed in.
    """
    predictions = []
    
    for ii, game in enumerate(games):
        # loop through each of the models passed

        # make a prediction based on the model
        prediction1 = rf_reg.predict([game[0]])
        prediction2 = -1 * rf_reg.predict([game[1]])
        prediction = (prediction1 + prediction2) / 2

        # print model and prediction
        if print_out:
            print(f"Random Forest model points spread: {matchups[ii][0]} {round(prediction[0], 4)} to {matchups[ii][1]} ({prediction1, prediction2})")
            print()
        
        predictions.append(prediction)
        
    return predictions


def rf_class_predict(rf_class, games, sc, matchups, print_out=True):
    """
        Takes in list of predictive classification models and data to make predictions upon.
        
        Prints out predictions for each model passed in.
    """
    predictions = []
    probabilities = []
    
    for ii, game in enumerate(games):

        # make a prediction based on the model
        #prediction1 = rf_class.predict(sc.transform([game[0]]))
        #prediction2 = rf_class.predict(sc.transform([game[1]]))
        #prediction = prediction1 + prediction2
        probability1 = rf_class.predict_proba(sc.transform([game[0]]))[0][1]
        probability2 = rf_class.predict_proba(sc.transform([game[1]]))[0][1]
        probability = (probability1 + (1 - probability2)) / 2

        if probability > 0.5:
            prediction = 1
            if print_out:
                print(f"Random Forest Classification model prediction: {matchups[ii][0]} Win ({round(probability * 100, 4)}% probability)")
        else:
            probability = probability
            prediction = 0
            if print_out:
                print(f"Random Forest Classification model prediction: {matchups[ii][0]} Loss ({round(probability * 100, 4)}% probability)")
        if print_out:
            print()
        
        predictions.append(prediction)
        probabilities.append(probability)
        
    return predictions, probabilities



def get_confidence(models, prediction, xs, pct_model, prob, pct_val_range):

    pcts = []

    for ii, model in enumerate(models):

        pct = 1 - (model[0]*(prediction**7) + model[1]*(prediction**6) + model[2]*(prediction**5) + model[3]*(prediction**4) + model[4]*(prediction**3) + model[5]*(prediction**2) + model[6]*prediction + model[7])

        if pct > 0.5:
            odds = -(100*pct)/(1-pct)
        else:
            odds = (100/pct) - 100

        pcts.append((pct, ii+xs[0], odds))

    win_pct = (pct_model[0]*(prob**5) + pct_model[1]*(prob**4) + pct_model[2]*(prob**3) + pct_model[3]*(prob**2) + pct_model[4]*prob + pct_model[5])
    
    return pcts, win_pct




def plot_params(SEs, STDs, MSEs, CVs, pct_accs, avg_lin_coefss):

    """
        Takes in model statistics: standard error, standard deviation, mean squared error,
        cross-validation score, percent accuracies, and linear coefficients. Each statistic
        should be a list of values generated for specific game numbers throughout the season.
        
        Function plots the values a function of the game number.
    """
    
    plt.plot(list(range(len(STDs))), STDs)


# In[ ]:




