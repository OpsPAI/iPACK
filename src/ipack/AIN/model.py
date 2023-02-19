import logging
import torch
import pytorch_lightning as pl
from torch import nn
from itertools import combinations, chain, product
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


def flatten_all_feats(feat_names):
    return sorted(list(chain(*feat_names.values())))


class InnerProductLayer(nn.Module):
    """ output: product_sum_pooling (bs x 1),
                Bi_interaction_pooling (bs * dim),
                inner_product (bs x f2/2),
                elementwise_product (bs x f2/2 x emb_dim)
    """

    def __init__(self, combs=None, output="product_sum_pooling"):
        super(InnerProductLayer, self).__init__()
        self._output_type = output
        if output not in [
            "product_sum_pooling",
            "Bi_interaction_pooling",
            "inner_product",
            "elementwise_product",
        ]:
            raise ValueError(
                "InnerProductLayer output={} is not supported.".format(output)
            )

        p, q = combs
        self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
        self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)

    def forward(self, feature_emb):
        if self._output_type in ["product_sum_pooling", "Bi_interaction_pooling"]:
            sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feature_emb ** 2, dim=1)  # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self._output_type == "Bi_interaction_pooling":
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self._output_type == "elementwise_product":
            emb1 = torch.index_select(feature_emb, 1, self.field_p)
            emb2 = torch.index_select(feature_emb, 1, self.field_q)
            return emb1 * emb2


class AIN(nn.Module):
    def __init__(
        self,
        combs=None,
        embedding_dim=10,
        attention_dropout=[0, 0],
        attention_dim=10,
        use_attention=True,
        **kwargs,
    ):
        super(AIN, self).__init__()
        self.use_attention = use_attention

        self.product_layer = InnerProductLayer(
            combs=combs, output="elementwise_product"
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.weight_p = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout[0])
        self.dropout2 = nn.Dropout(attention_dropout[1])
        # self.model_to_device()

    def forward(self, feature_emb):
        elementwise_product = self.product_layer(feature_emb)  # bs x f(f-1)/2 x dim
        if self.use_attention:
            attention_weight = self.attention(elementwise_product)
            attention_weight = self.dropout1(attention_weight)
            attention_sum = torch.sum(attention_weight * elementwise_product, dim=1)
            attention_sum = self.dropout2(attention_sum)
            ain_out = self.weight_p(attention_sum)
        else:
            ain_out = (
                torch.flatten(elementwise_product, start_dim=1)
                .sum(dim=-1)
                .unsqueeze(-1)
            )
        return ain_out, attention_weight


class AINMatcher(pl.LightningModule):
    def __init__(
        self,
        vocab_size_dict,
        sr_feat_names,
        incident_feat_names,
        comb_type="cross",
        embedding_dim=8,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.sr_feat_names = sr_feat_names
        self.incident_feat_names = incident_feat_names

        self.all_feat_names = flatten_all_feats(sr_feat_names) + flatten_all_feats(
            incident_feat_names
        )
        num_fields = len(self.all_feat_names)

        if comb_type == "cross":
            self.comb_names = list(
                product(
                    list(chain(*sr_feat_names.values())),
                    list(chain(*incident_feat_names.values())),
                )
            )
        else:
            self.comb_names = list(combinations(self.all_feat_names, 2))

        p, q = list(zip(*self.comb_names))
        p = [self.all_feat_names.index(item) for item in p]
        q = [self.all_feat_names.index(item) for item in q]
        combs = (p, q)

        self.ain_layer = AIN(
            num_fields=num_fields, combs=combs, embedding_dim=embedding_dim,
        )

        self.embedder = nn.ModuleDict(
            {
                feat: nn.Sequential(
                    nn.Embedding(vocab_size, embedding_dim), nn.Dropout(0.1)
                )
                for feat, vocab_size in vocab_size_dict.items()
            }
        )

        self.bert_mapping = nn.ModuleDict(
            {
                text_name: nn.Sequential(
                    nn.Linear(768, embedding_dim), nn.Tanh(), nn.Dropout(0.1)
                )
                for text_name in self.incident_feat_names["text"]
                + self.sr_feat_names["text"]
            }
        )

        self.linear_clf = nn.Sequential(
            nn.Linear(len(self.all_feat_names) * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False),
            nn.Sigmoid(),
        )

        self.save_hyperparameters()
        self.loss_fn = nn.BCELoss()
        self.reset_parameters()

    def reset_parameters(self):
        logging.info("Initing AIN parameters")

        def reset_param(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        torch.nn.init.normal_(v.weight, std=1e-3)
                    if type(v) == nn.Linear:
                        nn.init.xavier_normal_(v.weight)
                        if v.bias is not None:
                            v.bias.data.fill_(0)

        self.apply(reset_param)

    def embedding_feats(self, input_dict, featname_dict):
        feat2embedding = {}

        for feat in featname_dict["sparse"]:
            embedder = self.embedder[feat]
            embedding = embedder(input_dict[feat].unsqueeze(1).long())
            feat2embedding[feat] = embedding

        if "text" in featname_dict:
            for textname in featname_dict["text"]:
                text_embedding = self.bert_mapping[textname](
                    input_dict[textname].float()
                )
                feat2embedding[textname] = text_embedding.unsqueeze(1)
        return feat2embedding

    def forward(self, batch):

        sr_dict, incident_dict, label = batch

        feat_embedding = {}
        feat_embedding.update(
            self.embedding_feats(incident_dict, self.incident_feat_names)
        )
        feat_embedding.update(self.embedding_feats(sr_dict, self.sr_feat_names))

        pair_embedding = torch.cat(
            [feat_embedding[feat] for feat in self.all_feat_names], dim=1
        )

        ain_out, attention_weight = self.ain_layer(pair_embedding)
        logit = ain_out

        prob = logit.sigmoid()
        loss = self.loss_fn(prob.view(-1), label)
        return {
            "loss": loss,
            "prob": prob,
            "label": label,
            "logit": logit,
            "attention_weight": attention_weight,
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        return_dict = self.forward(batch)
        self.log("train_loss", return_dict["loss"])
        return return_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        return optimizer
