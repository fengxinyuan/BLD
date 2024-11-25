from torch.optim import AdamW
from bld import *
from load_dataset_and_preprocess import *
from sklearn.metrics import *
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    time_start = time.time()
    parser = argparse.ArgumentParser(description='load parameter for running and evaluating \
                                                 BLD learning')
    parser.add_argument('--dataset', '-d', type=str, default='cora',
                        help='Data set to be used')
    parser.add_argument('--positive_index', '-c', type=int, default=2,
                        help='Index of label to be used as positive')
    parser.add_argument('--sample_seed', '-s', type=int, default=1,
                        help='random seed for sample labeled positive from all positive nodes')
    parser.add_argument('--epoch', '-e', type=int, default=5000,
                        help='epoch for training')
    parser.add_argument('--train_pct', '-p', type=float, default=0.3,
                        help='Percentage of positive nodes to be used as training positive')
    parser.add_argument('--val_pct', '-v', type=float, default=0.0,
                        help='Percentage of positive nodes to be used as evaluating positive')
    parser.add_argument('--test_pct', '-t', type=float, default=1.00,
                        help='Percentage of unknown nodes to be used as test set')
    parser.add_argument('--hidden_size', '-l', type=int, default=32,
                        help='Size of hidden layers')
    parser.add_argument('--output_size', '-o', type=int, default=16,
                        help='Dimension of output representations')
    parser.add_argument('--result_path', '-r', type=str, default='./result/',
                        help='Path of result file')
    args = parser.parse_args()
    recall_result_path = args.result_path + "bld/" + "bld_" + args.dataset + str(args.train_pct) + "_recall.txt"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pu dataset
    data = load_dataset(args.dataset)
    data = make_pu_dataset(data, pos_index=[args.positive_index], sample_seed=args.sample_seed,
                           train_pct=args.train_pct, val_pct=args.val_pct,test_pct=args.test_pct)
    data = data.to(device)
    dataset = [data]

    # prepare augment
    drop_edge_p_1, drop_feat_p_1, drop_edge_p_2, drop_feat_p_2 = agmt_dict[args.dataset]
    augment_1 = augment_graph(drop_edge_p_1, drop_feat_p_1)
    augment_2 = augment_graph(drop_edge_p_2, drop_feat_p_2)

    # build BLD networks
    input_size = data.x.size(1)
    encoder = GCNEncoder(input_size, args.hidden_size, args.output_size)
    predictor = MLP_Predictor(args.output_size, args.hidden_size, args.output_size)
    classifier = MLP_Predictor(args.output_size, args.hidden_size, args.output_size)
    fc = torch.nn.Linear(args.output_size, 2, bias=False)
    model = BLD(encoder, predictor, classifier, fc).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=5e-4, weight_decay=1e-5)
    positive_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)
    val_nodes = data.val_mask.nonzero(as_tuple=False).view(-1)
    un_train_nodes = data.un_train_mask.nonzero(as_tuple=False).view(-1)

    train_node = torch.cat([un_train_nodes, positive_nodes], axis=0).cuda()
    un_soft_label = np.ones((un_train_nodes.size(0), 2)) * [0.5, 0.5]
    pos_soft_label = np.ones((positive_nodes.size(0), 2)) * [0.0, 1.0]
    soft_psd_label = np.concatenate((un_soft_label, pos_soft_label))

    def learn_repersentations(epoch):
        model.train()
        # forward
        optimizer.zero_grad()
        g1, g2 = augment_1(data), augment_2(data)
        p1, aux_h2, logits1 = model(g1, g2)
        p2, aux_h1, logits2 = model(g2, g1)

        pred1 = logits1[train_node].max(1)[1]
        pred2 = logits2[train_node].max(1)[1]
        if epoch > 200:
            un_train_nodes_pred = logits1[un_train_nodes]
            prob_pos = torch.index_select(un_train_nodes_pred, dim=1, index=torch.tensor([1]).cuda())
            tk = torch.topk(prob_pos, k=20, largest=True, dim=0)
            pred_pos_index = un_train_nodes[tk.indices.reshape(-1)]
            temp_pos= torch.cat([positive_nodes, pred_pos_index], dim=0)
            # b = data.y[pred_pos_index]
        else:
            temp_pos = positive_nodes

        # losses of predicting positive nodes
        positive_loss = predict_positive_nodes(p1, p2, temp_pos)
        # losses of predicting unlabeled nodes
        unlabeld_loss = predict_unlabeled_nodes(p1, aux_h2.detach(), p2, aux_h1.detach())

        # softmax loss
        update_psd_target(soft_psd_label, p1, train_node, positive_nodes, data.y, pred1)
        s_loss1 = -(torch.log(logits1[train_node]) * torch.tensor(soft_psd_label,
                                                                  requires_grad=False).cuda()).sum(dim=1).mean()

        update_psd_target(soft_psd_label, p2, train_node, positive_nodes, data.y, pred2)
        s_loss2 = -(torch.log(logits2[train_node]) * torch.tensor(soft_psd_label,
                                                                  requires_grad=False).cuda()).sum(dim=1).mean()

        s_loss = (s_loss1 + s_loss2) / 2

        loss = positive_loss + unlabeld_loss
        loss = loss + s_loss
        loss.backward()
        # update main network
        optimizer.step()
        # update axulary network
        model.update_aux_network(0.005)

    def test():
        model.eval()
        _,_,logits = model(data, data)
        pred = logits[data.test_mask].max(1)[1]  # predicted labels for test vertics, stored in tensor form: tensor[1,3,2,5,6,...]
        y_tr = data.y[data.test_mask]
        score = f1_score(y_tr.cpu(), pred.cpu(), pos_label=1)
        print(score)
        return score

    # learn representation, select reliable negatives, and
    data.y_psd_neg = data.y.clone().to(device)  # for recording selecting unlabeled as negative
    for epoch in range(1, 1500+1):
        learn_repersentations(epoch)
        if epoch % 20 == 0:
            score = test()
            print(score)

    time_end = time.time()
    print('time cost', time_end - time_start, 's')