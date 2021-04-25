import pandas as pd 
import numpy as np 

def get_date(dataframe, state, item="mask_date"):
    state = dataframe.loc[dataframe['state'] == state]
    
    date = np.array(state[item])[0].split('-')[::-1]
    return date[0] + '-' + date[1] + '-' + date[2]

def stateData(dataframe, state):
    # Create a copy
    dataframe2 = dataframe.copy()
    # set the index to be this and don't drop
    dataframe2.set_index(keys=['state'], drop=False,inplace=True)
    return dataframe2.loc[dataframe2.state==state]

def mask_date_to_all_states(state, all_states_file, stay_at_home_file, 
                            item="mask_date"):
    # all inputs are strings
    # files must be paths
    stay_at_home_df = pd.read_csv(stay_at_home_file)
    st = stateData(pd.read_csv(all_states_file), state)
    return st.loc[st['date'] == get_date(stay_at_home_df, state, item=item)]

# def get_all_dates(state, all_states_file, stay_at_home_file, time_data_file):
    
#     time_dataframe = pd.read_csv(time_data_file)
#     # get dataframe entries
    
#     mask_date_on_row = mask_date_to_all_states(state, all_states_file, stay_at_home_file, 
#                             item="mask_date")
#     mask_date_off_row = mask_date_to_all_states(state, all_states_file, stay_at_home_file, 
#                             item="mask_end_date")
    
#     stay_home_row = mask_date_to_all_states(state, all_states_file, stay_at_home_file, 
#                             item="stay_at_home_start")
#     go_out_row = mask_date_to_all_states(state, all_states_file, stay_at_home_file, 
#                             item="stay_at_home_end")
    
#     # extract time indices for each event
#     bools = time_dataframe["date"] == np.array(mask_date_on_row["date"])[0]
#     mask_on_index = len(bools) - np.arange(len(bools))[bools]
    
#     bools = time_dataframe["date"] == np.array(mask_date_off_row["date"])[0]
#     mask_off_index = len(bools) - np.arange(len(bools))[bools]

#     bools = time_dataframe["date"] == np.array(stay_home_row["date"])[0]
#     stay_at_home_on = len(bools) - np.arange(len(bools))[bools]
    
#     bools = time_dataframe["date"] == np.array(go_out_row["date"])[0]
#     stay_at_home_off = len(bools) - np.arange(len(bools))[bools]
    
#     return {"mask_on": mask_on_index[0], 
#             "mask_off": mask_off_index[0], 
#             "stay_home": stay_at_home_on[0],
#             "go_out": stay_at_home_off[0]}

if __name__ == "__main__":
    
    state = "new-jersey"
    all_events = get_all_dates("NJ", "all-states-history.csv", 'stay_at_home_and_masks.csv', "{}-history.csv".format(state))
    print(all_events)

