""" Imports """
import json
import pandas as pd
import os

""" Data cleaning functions """


def parse_obj(obj):
    for key in obj:
        if isinstance(obj[key], str):
            obj[key] = obj[key].encode("latin_1").decode("utf-8")
        elif isinstance(obj[key], list):
            obj[key] = list(
                map(
                    lambda x: x
                    if type(x) != str
                    else x.encode("latin_1").decode("utf-8"),
                    obj[key],
                )
            )
        pass
    return obj


def load_all_messages(path):
    """
    Load all json files from a specified folder into a pandas dataframe
    """
    files = sorted([f for f in os.listdir(path) if f.endswith(".json")])
    dfs = []
    for file in files:
        file = open(path + file, encoding="utf8")
        data = json.load(file, object_hook=parse_obj)
        df_temp = pd.json_normalize(data["messages"])
        dfs.append(df_temp)
    df = pd.concat(dfs)
    return df


def clean_data(df, exclude_members=None):
    """
    Clean a dataset
    """
    # We want a usable time stamp
    df["date_time"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

    # Standardize text
    df["content"] = df["content"].str.lower()

    # Keep only text data
    df.drop(
        columns=[
            "timestamp_ms",
            "gifs",
            "is_unsent",
            "photos",
            "type",
            "videos",
            "audio_files",
            "sticker.uri",
            "call_duration",
            "share.link",
            "share.share_text",
            "users",
            "files",
        ],
        inplace=True,
    )

    df["year"] = df["date_time"].dt.year
    df["hour"] = df["date_time"].dt.hour  # Hour of the day
    df["weekday"] = df["date_time"].dt.weekday  # Day of the week
    df["month"] = df["date_time"].apply(
        lambda x: str(x.year) + ("%0*d" % (2, x.month))
    )  # Month (with year)

    # Exclude certain participing people (optional)
    if exclude_members is None:
        exclude_members = []
    df = df[~df["sender_name"].isin(exclude_members)]

    df["content"] = df.content.fillna("")

    return df


def parse_reactions(df):
    """
    Get the reaction of each member
    """
    df = df.copy()

    members = list(
        set(
            ", ".join(
                df.loc[df.reactions.notnull(), "reactions"]
                .apply(lambda x: ", ".join([r["actor"] for r in x]))
                .unique()
            ).split(", ")
        )
    )

    for person in members:
        new_col = "reaction_" + person
        df[new_col] = ""
        df.loc[df.reactions.notnull(), new_col] = df.loc[
            df.reactions.notnull(), "reactions"
        ].apply(lambda x: "".join([r["reaction"] for r in x if r["actor"] == person]))

    del df["reactions"]

    return df


def prepare_data(path):
    """
    Reads in json message files from specified folder and returns cleaned dataframe.
    """
    df = clean_data(load_all_messages(path))
    df = parse_reactions(df)
    return df
