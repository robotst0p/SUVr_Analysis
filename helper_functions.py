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

    