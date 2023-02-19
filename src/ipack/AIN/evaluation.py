import csv

def output_df(df, outpath):
    df["TicketId"] = df["TicketId"].map(lambda x: "'" + str(x))
    df.to_csv(outpath, index=False, quoting=csv.QUOTE_ALL)


def get_eval_df(test_data, return_dict):
    prob_list = []
    for item in return_dict:
        prob_list.extend(item["prob"].numpy().reshape(-1))
    test_data["prob"] = prob_list
    return test_data
