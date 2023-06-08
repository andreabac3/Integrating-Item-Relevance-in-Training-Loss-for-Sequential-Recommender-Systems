from multiprocessing.sharedctypes import Value
import pickle as pkl
import os
import json
# import wget
import pandas as pd
import numpy as np
from ast import literal_eval
import datetime
from scipy import stats


def preprocess_dataset(data_cfg, project_folder, experiment_id):
    dataset_name = data_cfg["name"]
    min_rating = data_cfg["min_rating"]
    min_items_per_user = data_cfg["min_items_per_user"]
    min_users_per_item = data_cfg["min_users_per_item"]
    keep_time = data_cfg["keep_time"] if "keep_time" in data_cfg else False
    split_method = data_cfg["split_method"]

    if split_method == "leave_one_out" and min_items_per_user < 3 + data_cfg["test_num_samples"]:
        print(
            'Need at least 3+test_num_samples ratings per user for input, train, validation and test: min_items_per_user --> 3+test_num_samples')
        min_items_per_user = 3 + data_cfg["test_num_samples"]

    # Check if data exists; otherwise, download it
    data_folder = os.path.join(project_folder, "data")
    raw_data_folder = os.path.join(data_folder, "raw")
    dataset_raw_folder = os.path.join(raw_data_folder, dataset_name)
    # maybe_download_raw_dataset(dataset_raw_folder, dataset_name)
    maybe_preprocess_raw_dataset(dataset_raw_folder, dataset_name)

    df = load_ratings_df(dataset_raw_folder, dataset_name)

    df = make_implicit(df, min_rating)

    df = filter_by_frequence(df, min_items_per_user, min_users_per_item)

    df, dcts = densify_index(df)

    complete_set = df_to_sequences(df, keep_time)

    print_stats(complete_set, keep_time)

    train_set, val_set, test_set = split_df(complete_set, data_cfg)

    dataset = {'train': train_set,
               'val': val_set,
               'test': test_set}
    for k, v in dcts.items():
        dataset[k] = v

    # processed_data_folder = os.path.join(data_folder,"processed")
    # if not os.path.isdir(processed_data_folder):
    #     os.makedirs(processed_data_folder)

    # dataset_loc = os.path.join(processed_data_folder,str(experiment_id)+".pkl")
    # with open(dataset_loc,'wb') as f:
    #     pkl.dump(dataset, f)

    # GPT PREPROCESS
    # extra_dataset, item_id_to_title = extra_preprocess(dataset, dataset_raw_folder, dataset_name, keep_time)

    # dataset_loc = os.path.join(processed_data_folder,str(experiment_id)+".json")
    # with open(dataset_loc,'w') as f:
    #     json.dump(extra_dataset, f, indent=4, sort_keys=True)

    # file_loc = os.path.join(processed_data_folder,"item_id_to_title_"+str(experiment_id)+".json")
    # with open(file_loc,'w') as f:
    #     json.dump(item_id_to_title, f, indent=4, sort_keys=True)


def extra_preprocess(dataset, dataset_raw_folder, dataset_name, keep_time):
    item_info = load_item_info(dataset_raw_folder, dataset_name)
    item_id_to_title = pd.Series(item_info.iloc[:, 1].values, index=item_info.iloc[:, 0]).to_dict()
    title_to_item_id = {v: k for k, v in item_id_to_title.items()}

    genre_info = load_genre_info(dataset_raw_folder, dataset_name)
    id_to_genre = pd.Series(genre_info.iloc[:, 0].values, index=genre_info.iloc[:, 1]).to_dict()
    item_id_to_genre = {row[0]: [id_to_genre[g_id] for g_id in np.where(row[-19 + 1 * (dataset_name == "ml-1m"):])[0]]
                        for _, row in item_info.iterrows()}

    user_info = load_user_info(dataset_raw_folder, dataset_name)
    user_id_to_age = pd.Series(user_info["age"].values, index=user_info.iloc[:, 0]).to_dict()
    user_id_to_gender = pd.Series(user_info["gender"].values, index=user_info.iloc[:, 0]).to_dict()
    user_id_to_gender = {x: y for x, y in user_id_to_gender.items()}
    user_id_to_job = pd.Series(user_info["occupation"].values, index=user_info.iloc[:, 0]).to_dict()
    user_info_dict = {"gender": user_id_to_gender, "age": user_id_to_age, "job": user_id_to_job}

    # inverse smap
    new_item_id_to_original = {v: k for k, v in dataset["smap"].items()}

    new_dataset = extract_gpt_data(dataset, new_item_id_to_original, item_id_to_title, item_id_to_genre, user_info_dict,
                                   keep_time)

    return new_dataset, item_id_to_title


def extract_gpt_data(dataset, new_item_id_to_original, item_id_to_title, item_id_to_genre, user_info_dict, keep_time):
    new_seqs = {}
    for user, seq in dataset["train"].items():
        if keep_time:
            x = [new_item_id_to_original[i] for i in seq[0]] + [new_item_id_to_original[dataset["val"][user][0][0]]]
        else:
            x = [new_item_id_to_original[i] for i in seq] + [
                new_item_id_to_original[dataset["val"][user][0]]]  # not sure about this line
        y = dataset["test"][user]  # [0]
        if keep_time:
            time_y = datetime.datetime.fromtimestamp(y[1][0])
            time_y = {"year": time_y.year, "month": time_y.month, "day": time_y.day}
            y = y[0]  # [0]
        else:
            time_y = None
        y = [new_item_id_to_original[i] for i in y]
        genre_y = [item_id_to_genre[i] for i in y]
        user_info = {}
        for key, dct in user_info_dict.items():
            user_info["user_" + key] = dct[user]

        user_id = user  # dataset["umap"][user]
        str_x = [item_id_to_title[i] for i in x]
        str_y = [item_id_to_title[i] for i in y]
        new_seqs[user_id] = {"int_v": {"seq": x, "next": y}, "str_v": {"seq": str_x, "next": str_y},
                             "next_time": time_y, "next_genres": genre_y, **user_info}
    return new_seqs


# def maybe_download_raw_dataset(dataset_raw_folder, dataset_name):
#     print(dataset_raw_folder)
#     if os.path.isdir(dataset_raw_folder) and all(os.path.isfile(os.path.join(dataset_raw_folder,filename)) for filename in get_all_files_per_dataset(dataset_name)):
#         print('Raw data already exists. Skip downloading')
#         return
#     else:
#         raise NotImplementedError

# def get_all_files_per_dataset(dataset_name):
#     if dataset_name == "ml-1m":
#         return ['README', 'movies.dat', 'ratings.dat', 'users.dat']
#     elif dataset_name == "ml-100k":
#         return ['README', 'u.data', 'u.genre', 'u.info', 'u.info', 'u.occupation', 'u.user']
#     elif dataset_name == "steam":
#         return ['australian_user_reviews.json']
#     else:
#         raise NotImplementedError

def maybe_preprocess_raw_dataset(dataset_raw_folder, dataset_name):
    print(dataset_raw_folder)
    if os.path.isdir(dataset_raw_folder) and all(
            os.path.isfile(os.path.join(dataset_raw_folder, filename)) for filename in
            get_rating_files_per_dataset(dataset_name)):
        print('Ratings data already exists. Skip pre-processing')
        return
    else:
        preprocess_specific(dataset_raw_folder, dataset_name)


def get_rating_files_per_dataset(dataset_name):
    if dataset_name == "ml-1m":
        return ['ratings.dat']
    elif dataset_name == "ml-100k":
        return ['u.data']
    elif dataset_name == "steam":
        return ['australian_user_reviews.csv']
    elif dataset_name == "amazon_beauty":
        return ['All_Beauty.csv']
    elif dataset_name == "amazon_videogames":
        return ['Video_Games.csv']
    elif dataset_name == "amazon_toys":
        return ['Toys_and_Games.csv']
    elif dataset_name == "amazon_cds":
        return ['CDs_and_Vinyl.csv']
    elif dataset_name == "amazon_music":
        return ['Digital_Music.csv']
    elif dataset_name == "foursquare-nyc":
        return ['dataset_TSMC2014_NYC.txt']
    elif dataset_name == "foursquare-tky":
        return ['dataset_TSMC2014_TKY.txt']
    else:
        raise NotImplementedError


def preprocess_specific(dataset_raw_folder, dataset_name):
    if dataset_name == "steam":
        file_path = os.path.join(dataset_raw_folder,
                                 'australian_user_reviews.json')  # IT'S NOT A JSON... (NOR jsonl: single quotes instead of doubles)
        all_reviews = []
        with open(file_path, "r") as f:
            for line in f:
                line_dict = literal_eval(line)
                user_id = line_dict['user_id']
                for review_dict in line_dict['reviews']:
                    item_id = review_dict['item_id']
                    rating = review_dict['recommend'] * 1
                    timestamp = review_dict['posted'][7:-1]  # removing "Posted " and "."
                    try:
                        timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(timestamp, "%B %d, %Y"))
                    except ValueError:
                        timestamp = -1
                    timestamp = int(timestamp)
                    all_reviews.append((user_id, item_id, rating, timestamp))

        all_reviews = pd.DataFrame(all_reviews)
        all_reviews.to_csv(os.path.join(dataset_raw_folder, 'australian_user_reviews.csv'), header=False, index=False)
    elif "amazon" in dataset_name:  # structure should be the same for all Amazon (2018) datasets
        if dataset_name == "amazon_beauty":
            orig_file_name = 'All_Beauty'
        elif dataset_name == "amazon_videogames":
            orig_file_name = 'Video_Games'
        elif dataset_name == "amazon_toys":
            orig_file_name = 'Toys_and_Games'
        elif dataset_name == "amazon_cds":
            orig_file_name = 'CDs_and_Vinyl'
        elif dataset_name == "amazon_music":
            orig_file_name = 'Digital_Music'

        file_path = os.path.join(dataset_raw_folder,
                                 orig_file_name + '.json')  # IT'S NOT A JSON... (NOR jsonl: single quotes instead of doubles)
        all_reviews = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.replace('"verified": true,', '"verified": True,').replace('"verified": false,',
                                                                                      '"verified": False,')
                line_dict = literal_eval(line)
                user_id = line_dict['reviewerID']
                item_id = line_dict['asin']
                rating = float(line_dict['overall'])
                timestamp = line_dict['unixReviewTime']
                all_reviews.append((user_id, item_id, rating, timestamp))

        all_reviews = pd.DataFrame(all_reviews)
        all_reviews.to_csv(os.path.join(dataset_raw_folder, orig_file_name + '.csv'), header=False, index=False)
    else:
        raise NotImplementedError


def load_ratings_df(dataset_raw_folder, dataset_name):
    if dataset_name == "ml-1m":
        file_path = os.path.join(dataset_raw_folder, 'ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder, 'u.data')
        df = pd.read_csv(file_path, sep='\t', header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif "amazon" in dataset_name or dataset_name == "steam":
        if dataset_name == "steam":
            orig_file_name = 'australian_user_reviews'
        elif dataset_name == "amazon_beauty":
            orig_file_name = 'All_Beauty'
        elif dataset_name == "amazon_videogames":
            orig_file_name = 'Video_Games'
        elif dataset_name == "amazon_toys":
            orig_file_name = 'Toys_and_Games'
        elif dataset_name == "amazon_cds":
            orig_file_name = 'CDs_and_Vinyl'
        elif dataset_name == "amazon_music":
            orig_file_name = 'Digital_Music'
        file_path = os.path.join(dataset_raw_folder, orig_file_name + '.csv')
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df
    elif "foursquare" in dataset_name:
        if dataset_name == "foursquare-nyc":
            filename = 'dataset_TSMC2014_NYC.txt'
        elif dataset_name == "foursquare-tky":
            filename = 'dataset_TSMC2014_TKY.txt'
        file_path = os.path.join(dataset_raw_folder, filename)
        df = pd.read_csv(file_path, sep='\t', header=None, encoding='latin-1')
        df.columns = ['uid', 'sid', "s_cat", "s_cat_name", "latitude", "longitude", "timezone_offset", "UTC_time"]
        df["rating"] = 1  # there are no ratings
        df["timestamp"] = df["UTC_time"].apply(
            lambda x: datetime.datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y").timestamp())

        return df
    else:
        raise NotImplementedError


def load_item_info(dataset_raw_folder, dataset_name):
    ML_genres = np.array(["Action", "Adventure", "Animation",
                          "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                          "Thriller", "War", "Western"])

    if dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder, 'u.item')
        df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
        df.columns = ["movie id", "movie title", "release date", "video release date",
                      "IMDb URL", "unknown", *ML_genres]
        df["movie title"] = make_title_better(df["movie title"])
    elif dataset_name == "ml-1m":
        file_path = os.path.join(dataset_raw_folder, 'movies.dat')
        df = pd.read_csv(file_path, sep='::', header=None, encoding='latin-1')
        vecs = []
        for _, row in df.iterrows():
            row_genres = row.iloc[-1].split("|")
            vec = np.zeros(len(ML_genres)).astype(int)
            for genre in row_genres:
                vec[np.where(ML_genres == genre)[0]] = 1
            vecs.append(vec)
        df = df.join(pd.DataFrame(vecs), rsuffix="Cat")
        df.drop("2", axis=1, inplace=True)
        df.columns = ["movie id", "movie title", *ML_genres]
        df["movie title"] = make_title_better(df["movie title"])
    else:
        raise NotImplementedError
    return df


def make_title_better(lst):
    new_titles = []
    for complete_title in lst:
        # if "," in complete_title:
        # print(complete_title)
        year = complete_title.split("(")
        title = "(".join(year[:-1]).strip()
        year = year[-1][:-1]
        if "(" in title:
            title2 = title.split("(")
            title = "(".join(title2[:-1]).strip()
            title2 = title2[-1][:-1]
            titles = [title, title2]
        else:
            titles = [title]

        articles = ["A", "An", "The", "Il", "L'", "La", "Le", "Les", "O", "Das", "Der", "Die", "Det"]
        for i, tlt in enumerate(titles):
            if len(tlt) != 0:
                for article in articles:
                    app = 2 + len(article)
                    if tlt[-app:] == ", " + article:
                        if "'" in article: app -= 1
                        titles[i] = article + " " + tlt[:-app]
                # if "," in complete_title and tlt==titles[i]: print(tlt,titles[i])
        title = titles[0]
        for tlt in [*titles[1:], year]:
            title += " (" + tlt + ")"
        # if "," in complete_title: print(title)
        new_titles.append(title)
    return new_titles


def load_genre_info(dataset_raw_folder, dataset_name):
    if dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder, 'u.genre')
        df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
        df.columns = ['genre', 'gid']
    elif dataset_name == "ml-1m":
        df = load_genre_info(os.path.join(*dataset_raw_folder.split("/")[:-1], "ml-100k"), "ml-100k")
        df = df.iloc[1:]
        df["gid"] = df["gid"] - 1
        df.reset_index(drop=True, inplace=True)
    else:
        raise NotImplementedError
    return df


def load_user_info(dataset_raw_folder, dataset_name):
    if dataset_name == "ml-100k":
        file_path = os.path.join(dataset_raw_folder, 'u.user')
        df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1')
        df.columns = ["user id", "age", "gender", "occupation", "zip code"]
        df.replace({"gender": {"M": "Male", "F": "Female"}}, inplace=True)
    elif dataset_name == "ml-1m":
        file_path = os.path.join(dataset_raw_folder, 'users.dat')
        df = pd.read_csv(file_path, sep='::', header=None, encoding='latin-1')
        df.columns = ["user id", "gender", "age", "occupation", "zip code"]
        df.replace({"age": {1: "Under 18",
                            18: "18-24",
                            25: "25-34",
                            35: "35-44",
                            45: "45-49",
                            50: "50-55",
                            56: "56+"}}, inplace=True)
        df.replace({"gender": {"M": "Male", "F": "Female"}}, inplace=True)
        df.replace({"occupation": {0: "other or not specified", 1: "academic/educator", 2: "artist",
                                   3: "clerical/admin", 4: "college/grad student", 5: "customer service",
                                   6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker",
                                   10: "K-12 student",
                                   11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
                                   15: "scientist",
                                   16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
                                   19: "unemployed", 20: "writer"}}, inplace=True)
    else:
        raise NotImplementedError
    return df


def make_implicit(df, min_rating):
    print('Turning into implicit ratings >=', min_rating)
    df = df[df['rating'] >= min_rating]
    return df


def filter_by_frequence(df, min_items_per_user, min_users_per_item):
    if min_users_per_item > 0:
        print('Filtering by minimum number of users per item:', min_users_per_item)
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_users_per_item]
        df = df[df['sid'].isin(good_items)]

    if min_items_per_user > 0:
        print('Filtering by minimum number of items per user:', min_items_per_user)
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_items_per_user]
        df = df[df['uid'].isin(good_users)]

    return df


def densify_index(df, vars=["uid", "sid"], map_names=["umap", "smap"]):
    print('Densifying index')
    # if keep_time: vars.append("timestamp") #actually, in this way it can work with timestep just in the input set (not in the time interval!)
    maps = {}
    for map_name, var in zip(map_names, vars):
        maps[map_name] = {u: i + 1 for i, u in enumerate(set(df[var]))}  # Probably not a great way to name maps
        df[var] = df[var].map(maps[map_name])
    return df, maps


def df_to_sequences(df, keep_time):
    df_group_by_user = df.groupby('uid')
    user2items = df_group_by_user.apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
    complete_set = user2items.to_dict()
    if keep_time:
        user2times = df_group_by_user.apply(lambda d: list(d.sort_values(by='timestamp')['timestamp'])).to_dict()
        complete_set = {u: [seq, user2times[u]] for u, seq in complete_set.items()}
    return complete_set


def print_stats(complete_set, keep_time):
    print("NUM USERS:", len(complete_set))

    if keep_time:
        print("NUM ITEMS:", len(set(np.concatenate([seq for u, (seq, times) in complete_set.items()]))))
    else:
        print("NUM ITEMS:", len(set(np.concatenate([seq for u, seq in complete_set.items()]))))

    if keep_time:
        lens = [len(seq) for u, (seq, times) in complete_set.items()]
    else:
        lens = [len(seq) for u, seq in complete_set.items()]
    print("AVERAGE LEN:", np.mean(lens))
    print("MEDIAN LEN:", np.median(lens))
    print("MODE LEN:", stats.mode(lens))
    print("STD LEN:", np.std(lens))
    print("MIN/MAX LEN:", np.min(lens), np.max(lens))

    if keep_time:
        print("NUM INTERACTIONS:", np.sum([len(seq) for u, (seq, times) in complete_set.items()]))
    else:
        print("NUM INTERACTIONS:", np.sum([len(seq) for u, seq in complete_set.items()]))


def split_df(complete_set, cfg):
    print('Splitting:', cfg["split_method"])
    if cfg["split_method"] == 'leave_one_out':
        train_set, val_set, test_set = {}, {}, {}
        if "keep_time" in cfg and cfg["keep_time"]:
            for user, (seq, times) in complete_set.items():
                train_set[user], val_set[user], test_set[user] = [seq[:-1 - cfg["test_num_samples"]],
                                                                  times[:-1 - cfg["test_num_samples"]]], [seq[-1 - cfg[
                    "test_num_samples"]:-cfg["test_num_samples"]], times[-1 - cfg["test_num_samples"]:-cfg[
                    "test_num_samples"]]], [seq[-cfg["test_num_samples"]:], times[-cfg["test_num_samples"]:]]
        else:
            for user, seq in complete_set.items():
                train_set[user], val_set[user], test_set[user] = seq[:-1 - cfg["test_num_samples"]], seq[-1 - cfg[
                    "test_num_samples"]:-cfg["test_num_samples"]], seq[-cfg["test_num_samples"]:]
    elif cfg["split_method"] == 'hold_out':
        # Generate user indices
        np.random.seed(cfg["seed"])
        permuted_index = np.random.permutation(list(complete_set.keys()))

        test_size = int(cfg["test_size"] * len(permuted_index))
        val_size = int(cfg["val_size"] * (len(permuted_index) - test_size))

        train_users = permuted_index[: -val_size - test_size]
        val_users = permuted_index[-val_size - test_size: -test_size]
        test_users = permuted_index[-test_size:]

        train_set = {u: complete_set[u] for u in train_users}
        val_set = {u: complete_set[u] for u in val_users}
        test_set = {u: complete_set[u] for u in test_users}
    else:
        raise NotImplementedError

    return train_set, val_set, test_set


if __name__ == "__main__":
    data_cfg = {"name": "amazon_videogames",  # ml-1m #ml-100k  #foursquare-nyc #foursquare-tky
                "min_rating": 0,
                "min_items_per_user": 0,  # needed for split train/val/test
                "min_users_per_item": 0,
                "split_method": "leave_one_out",  # leave_one_out #hold_out,
                "test_num_samples": 1,
                "keep_time": True
                }
    experiment_id = f'{data_cfg["name"]}_{str(data_cfg["test_num_samples"])}'

    preprocess_dataset(data_cfg, "../", experiment_id)
