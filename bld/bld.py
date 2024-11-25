import copy
import torch
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import *

class BLD(torch.nn.Module):
    def __init__(self, encoder, predictor, classifier, fc):
        super().__init__()
        self.main_encoder = encoder
        self.mlp_predictor = predictor
        self.aux_encoder = copy.deepcopy(encoder)
        self.aux_encoder.reset_parameters()
        self.classifier = classifier
        self.fc = fc
        for param in self.aux_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return list(self.main_encoder.parameters()) + list(self.mlp_predictor.parameters()) \
               + list(self.classifier.parameters()) + list(self.fc.parameters())

    @torch.no_grad()
    def update_aux_network(self, tau):
        for param_q, param_k in zip(self.main_encoder.parameters(), self.aux_encoder.parameters()):
            param_k.data.mul_(tau).add_(param_q.data, alpha=1. - tau)

    def forward(self, main_x, aux_x):
        main_h = self.main_encoder(main_x)
        main_p = self.mlp_predictor(main_h)
        main_cls = self.classifier(main_h.detach())
        logits = F.softmax(self.fc(main_cls))

        with torch.no_grad():
            aux_h = self.aux_encoder(aux_x).detach()
        return main_p, aux_h, logits

def predict_unlabeled_nodes(p1, aux_h2, p2, aux_h1):
    loss1 = 2 - 2*cosine_similarity(p1, aux_h2, dim=-1).mean()
    loss2 = 2 - 2*cosine_similarity(p2, aux_h1, dim=-1).mean()
    return (loss1 + loss2)/2

def predict_positive_nodes(p1, p2, positive_nodes):
    loss1 = _predict_positive(p1, positive_nodes)
    loss2 = _predict_positive(p2, positive_nodes)
    loss = loss1 + loss2
    return loss/2

def _predict_positive(p, seed_nodes):
    pos_centroid = p[seed_nodes].mean(dim=0)
    pos_loss = 2 - 2*cosine_similarity(p[seed_nodes], pos_centroid, dim=-1).mean()
    return pos_loss

def find_reliable_negative_nodes(p1, labeled_positive_nodes):
    pos_centroid1 = p1[labeled_positive_nodes].mean(dim=0)
    # distance measure
    similarities = cosine_similarity(p1, pos_centroid1, dim=-1)
    similarities[labeled_positive_nodes] = float('inf')
    tk = torch.topk(similarities, k=labeled_positive_nodes.size(0), largest=False, dim=0)
    negative_nodes = tk.indices.reshape(-1)
    return negative_nodes

# def update_psd_target(soft_psd_label, p, train_node, labeled_positive_nodes, y):
#     pos_centroid = p[labeled_positive_nodes].mean(dim=0)
#     pos_similarities = cosine_similarity(p, pos_centroid, dim=-1)
#     neg_idx = torch.min(pos_similarities, dim=0)[1]
#     neg_centroid = p[neg_idx]
#     neg_similarities = cosine_similarity(p, neg_centroid, dim=-1) # not effective
#
#     similarities = torch.stack([neg_similarities, pos_similarities], dim=1).cuda()
#     psd_lbl = torch.max(similarities, dim=1)[1]
#     train_psd_lbl = psd_lbl[train_node]       # require accurate labels
#     ry = y[train_node]
#     train_psd_lbl[-labeled_positive_nodes.size(0):]=1
#
#     # indices = torch.arange(train_psd_lbl.size(0)).cuda()
#     indices = np.arange(train_psd_lbl.size(0))
#     soft_psd_label[indices, train_psd_lbl.cpu().detach().numpy()] += 0.1
#     # soft_psd_label[:] = soft_psd_label / soft_psd_label.sum(dim=1, keepdim=True)
#     soft_psd_label[:] = soft_psd_label / np.sum(soft_psd_label, axis=1, keepdims=True)
#     return soft_psd_label

def update_psd_target(soft_psd_label, p, train_node, labeled_positive_nodes, y, pred, lbd=1):
    pos_centroid = p[labeled_positive_nodes].mean(dim=0)
    pos_similarities = cosine_similarity(p, pos_centroid, dim=-1)
    boundary = pos_similarities[labeled_positive_nodes].min()
    # a = pos_similarities[train_node]
    # tr_l = y[train_node]
    train_psd_lbl = (pos_similarities[train_node]>lbd*boundary).long()
    # print((tr_l!=train_psd_lbl).sum())

    # indices = torch.arange(train_psd_lbl.size(0)).cuda()
    indices = np.arange(train_psd_lbl.size(0))
    soft_psd_label[indices, train_psd_lbl.cpu().detach().numpy()] += 0.01
    # soft_psd_label[:] = soft_psd_label / soft_psd_label.sum(dim=1, keepdim=True)
    soft_psd_label[:] = soft_psd_label / np.sum(soft_psd_label, axis=1, keepdims=True)

    soft_psd_label[indices, pred.cpu().detach().numpy()] += 0.01
    # soft_psd_label[:] = soft_psd_label / soft_psd_label.sum(dim=1, keepdim=True)
    soft_psd_label[:] = soft_psd_label / np.sum(soft_psd_label, axis=1, keepdims=True)
    return soft_psd_label

def update_psd_target_with_recall(soft_psd_label, p, train_node, un_train_nodes, labeled_positive_nodes, y, pred, lbd=1):
    pos_centroid = p[labeled_positive_nodes].mean(dim=0)
    pos_similarities = cosine_similarity(p, pos_centroid, dim=-1)
    boundary = pos_similarities[labeled_positive_nodes].min()
    # a = pos_similarities[train_node]
    # tr_l = y[train_node]
    train_psd_lbl = (pos_similarities[train_node]>lbd*boundary).long()
    un_train_psd_lbl = (pos_similarities[un_train_nodes]>lbd*boundary).long()
    zone_idf_score = f1_score(y[un_train_nodes].cpu(), un_train_psd_lbl.cpu(), pos_label=1)
    # print((tr_l!=train_psd_lbl).sum())

    # indices = torch.arange(train_psd_lbl.size(0)).cuda()
    indices = np.arange(train_psd_lbl.size(0))
    soft_psd_label[indices, train_psd_lbl.cpu().detach().numpy()] += 0.01
    # soft_psd_label[:] = soft_psd_label / soft_psd_label.sum(dim=1, keepdim=True)
    soft_psd_label[:] = soft_psd_label / np.sum(soft_psd_label, axis=1, keepdims=True)

    u_size = un_train_nodes.size(0)
    a = np.argmax(soft_psd_label[:u_size],axis=1)
    c = torch.tensor(soft_psd_label,requires_grad=False)[:u_size].max(1)[1]
    t = y[un_train_nodes].cpu()
    accumulate_idf_f1score = f1_score(t, c , pos_label=1)
    accumulate_idf_accscore = accuracy_score(t, c)

    soft_psd_label[indices, pred.cpu().detach().numpy()] += 0.01
    # soft_psd_label[:] = soft_psd_label / soft_psd_label.sum(dim=1, keepdim=True)
    soft_psd_label[:] = soft_psd_label / np.sum(soft_psd_label, axis=1, keepdims=True)

    return soft_psd_label, zone_idf_score, accumulate_idf_f1score, accumulate_idf_accscore
