import logging
from tqdm import tqdm


def assign_cluster(df_predicted, graph=None, use_graph=False, prob_thershold=0.3):
    if use_graph:
        logging.info("Using clustering to merge SRs..")

    ticketid_list = df_predicted["TicketId"].unique()
    srid2clusterid = {}

    for srid in tqdm(ticketid_list):
        srdf = df_predicted[df_predicted["TicketId"] == srid]

        top_rows = (
            srdf[srdf["prob"] >= prob_thershold]
            .sort_values(by=["prob"], ascending=False)
            .reset_index()
        )
        if top_rows.shape[0] > 0:
            row_dict = dict(top_rows.loc[0])
            eventid = row_dict["EventId"]
            windowid = row_dict["WindowId"]

            srid2clusterid[srid] = "{}-{}".format(windowid, eventid)
            if use_graph:
                outage_graph = graph[windowid]
                if eventid in outage_graph.nodes():
                    srid2clusterid[srid] = "Incident-{}".format(windowid)
        else:
            srid2clusterid[srid] = "Non-Incident"

    df_predicted = df_predicted.assign(
        clusterid=df_predicted["TicketId"].map(lambda x: srid2clusterid[x])
    )
    return df_predicted
