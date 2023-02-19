import re
import string
import hashlib
import pandas as pd
import logging
from ipack.parser import Drain
from tqdm import tqdm


class AlertDrain:
    def __init__(self, hashid, savePath, depth, st, regex):
        self.hashid = hashid
        self.savePath = savePath
        self.depth = depth
        self.st = st
        self.regex = regex

    def generate_tokens(self, message):
        """Function to generate a list of tokens and splitters"""
        punc = re.sub("[*<>\.\(\)\-\/\\\_]", "", string.punctuation)
        splitters = "\s\\" + "\\".join(punc)
        splitter_regex = re.compile("([{}]+)".format(splitters))
        tokens = re.split(splitter_regex, message)
        return [item for item in tokens if item]

    def __generate_hashid(self, df):
        df = df.assign(hashid=df[self.hashid].fillna("").agg("-".join, axis=1))
        return df

    def parse(self, df, col, drop_duplicate=False):
        logging.info("Parsing {} lines.".format(df.shape[0]))
        if drop_duplicate:
            df_dedup = df.drop_duplicates(subset=[col])
        else:
            df_dedup = df
        logging.info("After depulication {} lines.".format(df_dedup.shape[0]))

        df_dedup = self.__generate_hashid(df_dedup)

        logging.info("{} hashids to parse.".format(df_dedup["hashid"].nunique()))
        resdf = []
        for hs in tqdm(df_dedup["hashid"].unique()):
            partition = df_dedup[df_dedup["hashid"] == hs]
            parser = Drain.LogParser(
                outdir="",
                depth=self.depth,
                st=self.st,
                rex=self.regex,
                keep_para=False,
            )
            resdf.append(parser.parse("chunk", partition, col=col))
        resdf = pd.concat(resdf)
        logging.info("Finished.")

        content2tmplt = dict(zip(resdf[col], resdf["EventTemplate"]))
        df["EventTemplate"] = df[col].map(content2tmplt).fillna("")
        df["EventId"] = df["EventTemplate"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8]
        )

        return df
