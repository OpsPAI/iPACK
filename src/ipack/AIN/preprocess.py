from collections import defaultdict


class Preprocessor:
    def __init__(self, feature_used, topic_level=3):
        self.feature_used = feature_used
        self.topic_level = topic_level

    def transform(self, df):
        for datatype in self.feature_used:
            self.process(df, self.feature_used[datatype])
        return df

    def process(self, df, feature_used):
        if (
                "Category" in feature_used["sparse"]
                or "Category_L1" in feature_used["sparse"]
        ):
            df = self.__split_topics(df, feature_used)
        if (
                "OwningComponent" in feature_used["sparse"]
                or "Team_1" in feature_used["sparse"]
        ):
            df = self.__split_team(df, feature_used)
        return df

    def __split_topics(self, df, feature_used):
        assert "Category" in df.columns
        topic_columns = [f"Category_L{i + 1}" for i in range(self.topic_level)]
        df[topic_columns] = df["Category"].str.split("\\", expand=True)[
            [i for i in range(self.topic_level)]
        ]
        feature_used["sparse"] += topic_columns
        if "Category" in feature_used["sparse"]:
            feature_used["sparse"].remove("Category")
        feature_used["sparse"] = list(set(feature_used["sparse"]))
        self.feature_used["SR"] = feature_used
        return df

    def __split_team(self, df, feature_used):
        assert "OwningComponent" in df.columns
        team_columns = [f"Team_{i + 1}" for i in range(2)]
        df[team_columns] = df["OwningComponent"].str.split("\\", expand=True)[
            [i for i in range(2)]
        ]

        feature_used["sparse"] += team_columns
        if "OwningComponent" in feature_used["sparse"]:
            feature_used["sparse"].remove("OwningComponent")
        feature_used["sparse"] = list(set(feature_used["sparse"]))
        self.feature_used["Incident"] = feature_used
        return df

    def get_feature_names(self, merge_incident_sr=False):
        if not merge_incident_sr:
            return self.feature_used
        else:
            merged_dict = defaultdict(list)
            for key in self.feature_used["SR"]:
                merged_dict[key] += list(set(self.feature_used["SR"][key] + self.feature_used["Incident"][key]))
            return merged_dict
