import pandas as pd
import scipy.sparse as sp
import os


def create_URM_matrix(ratings_df):
    URM_all = sp.csr_matrix((ratings_df["data"].values,
                             (ratings_df["user_id"].values, ratings_df["item_id"].values)))

    return URM_all


def create_ICM_matrix(dataframe):
    csr_matrix = sp.csr_matrix((dataframe["data"].values,
                                (dataframe["item_id"].values, dataframe["feature_id"].values)))

    return csr_matrix


def combine_matrices(URM: sp.csr_matrix, ICM: sp.csr_matrix):
    stacked_URM = sp.vstack([URM, ICM.T])
    stacked_URM = sp.csr_matrix(stacked_URM)

    stacked_ICM = sp.csr_matrix(stacked_URM.T)
    return stacked_URM, stacked_ICM


def load_URM():
    interactions_df = load_data_interactions()

    # Make watched = 1
    interactions_df.loc[interactions_df['data'] == 0, "data"] = 1

    # Drop duplicates
    interactions_df.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)

    URM_all = create_URM_matrix(interactions_df)

    return URM_all


def load_data():
    interactions_df = load_data_interactions()
    length_df = load_data_length()
    type_df = load_data_type()

    interactions_df.drop_duplicates()

    interactions_df.loc[interactions_df['data'] == 0, "data"] = 1

    # Remove cold items
    length_df = length_df[length_df.item_id.isin(interactions_df.item_id)]
    type_df = type_df[type_df.item_id.isin(interactions_df.item_id)]

    # FEATURES
    all_features_indices = pd.concat([length_df["feature_id"], type_df["feature_id"]], ignore_index=True)
    mapped_id, original_id = pd.factorize(all_features_indices.unique())

    print("Unique features: {}".format(len(original_id)))

    features_original_ID_to_index = pd.Series(mapped_id, index=original_id)

    length_df["feature_id"] = length_df["feature_id"].map(features_original_ID_to_index)
    type_df["feature_id"] = type_df["feature_id"].map(features_original_ID_to_index)

    URM_all = create_URM_matrix(interactions_df)
    ICM_length = create_ICM_matrix(length_df)
    ICM_type = create_ICM_matrix(type_df)

    ICM_all = sp.hstack([ICM_type, ICM_length])

    return URM_all, ICM_type, ICM_length, ICM_all


def load_data_interactions():
    if os.path.exists("../data/interactions_and_impressions.csv"):
        print('interactions_and_impressions found!')
        return pd.read_csv(
            "../data/interactions_and_impressions.csv",
            sep=",",
            names=["user_id", "item_id", "impressions", "data"],
            header=0,
            dtype={"user_id": int, "item_id": int, "impressions": str, "data": int})
    else:
        print("interactions_and_impressions not found.")
        return None


def load_data_length():
    if os.path.exists("../data/data_ICM_length.csv"):
        print('data_ICM_length found!')
        return pd.read_csv("../data/data_ICM_length.csv",
                           sep=",",
                           names=["item_id", "feature_id", "data"],
                           header=0,
                           dtype={"item_id": int, "feature_id": int, "data": int})
    else:
        print("data_ICM_length not found.")
        return None


def load_data_type():
    if os.path.exists("../data/data_ICM_type.csv"):
        print('data_ICM_type found!')
        return pd.read_csv("/Users/redaellimattia/Desktop/RecSysCompetition/Competition/data/data_ICM_type.csv",
                           sep=",",
                           names=["item_id", "feature_id", "data"],
                           header=0,
                           dtype={"item_id": int, "feature_id": int, "data": int})
    else:
        print("data_ICM_type not found.")
        return None


def load_users_for_submission():
    if os.path.exists("../data/data_target_users_test.csv"):
        print('data_target_users_test found!')
        return pd.read_csv(
            "/Users/redaellimattia/Desktop/RecSysCompetition/Competition/data/data_target_users_test.csv",
            names=['user_id'],
            header=0,
            dtype={"user_id": int})
    else:
        print("data_target_users_test not found.")
        return None


def create_submission(recommender):
    users_df = load_users_for_submission()
    submission = []
    for user_id in users_df["user_id"].values:
        submission.append((user_id, recommender.recommend(user_id_array=user_id, cutoff=10)))

    return submission


def write_submission(submission, file_name):
    with open("../submissions/" + file_name + ".csv",
              "w") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submission:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")
