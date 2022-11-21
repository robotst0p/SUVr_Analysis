def retrieve_feature_names(rfe_bool, data):
    feat_index = []
    feature_list = []

    for i in range(0, len(rfe_bool)):
        if (rfe_bool[i] == True):
            feat_index.append(i)

    for column_num in range(0, len(data.columns)):
        if column_num in feat_index:
            feature_list.append(data.columns[column_num])

    return feature_list

def feature_vote(feature_list, rfe_features, threshold, data):
    #threshold is percentage value
    voting_thresh = round(27*threshold)
    
    voting_list = []
    final_features = []
    
    voting_dict = {}
    
    for feature in feature_list:    
        if (feature not in voting_list):
            voting_dict[feature] = feature_list.count(feature)
            voting_list.append(feature)
            
    for feature, vote in voting_dict.items():
        if (vote >= voting_thresh):
            final_features.append(feature)
            
    data = data.drop(final_features, axis=1)
    
    for feature in data.columns:
        voting_dict[feature] = 0
            
    return (final_features, voting_dict)
    