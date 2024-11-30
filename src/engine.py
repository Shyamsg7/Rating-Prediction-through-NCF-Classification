#HAVE TO CHANGE THIS FILE FOR CLASSIFICATION


import torch
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_optimizer
# from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        # self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback

        #HAVE TO CHANGE THIS FOR CLASSIFICATION
        self.crit = torch.nn.CrossEntropyLoss()
        # self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        correct = 0
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        logits = self.model(users, items)
        loss = self.crit(logits, ratings)  # CrossEntropyLoss handles class labels directly
        loss.backward()
        self.opt.step()
        _, predictions = torch.max(logits, dim=1)
        # predictions = torch.argmax(logits, dim=0)
        correct += (predictions == ratings).sum().item()
        p=len(ratings)
        # total += labels.size(0)
        return loss.item() , correct ,p
        # ratings_pred = self.model(users, items)
        # loss = self.crit(ratings_pred.view(-1), ratings)
        # loss.backward()
        # self.opt.step()
        # loss = loss.item()
        # return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_preds=0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            # rating = rating.float()
            loss , correct ,p  = self.train_single_batch(user, item, rating)
            # print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
            total_correct += correct
            total_preds+=p
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
        print('[Training Epoch {}] Loss = {:.4f} Accuracy = {:.4f}'.format(epoch_id+1, total_loss / len(train_loader) , total_correct/total_preds))
        return total_loss / len(train_loader) , total_correct/total_preds

    # CHANGE THIS FOR CLASSIFICATION
    def evaluate(self, test_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model!'
        self.model.eval()
        correct = 0
        total = 0
        total_loss=0
        with torch.no_grad():
            # for batch in tqdm(test_loader, desc=f'Evaluating Epoch {epoch_id+1}'):
            for batch in test_loader:
                users, items, labels = batch[0], batch[1], batch[2]
                if self.config['use_cuda']:
                    users, items, labels = users.cuda(), items.cuda(), labels.cuda()
                logits = self.model(users, items)
                loss = self.crit(logits, labels)
                _, predictions = torch.max(logits, dim=1)
                # predictions = torch.argmax(logits, dim=0)
                correct += (predictions == labels).sum().item()
                total_loss+=loss.item()
                total += labels.size(0)
        accuracy = correct / total
        self._writer.add_scalar('performance/accuracy', accuracy, epoch_id)
        print('[Evaluation Epoch {}] Loss = {:.4f} Accuracy = {:.4f}'.format(epoch_id+1, total_loss / len(test_loader) , accuracy))
        # print(f'[Evaluating Epoch {epoch_id+1}] Accuracy = {accuracy:.4f}')
        return total_loss/len(test_loader), accuracy
    
    # def evaluate(self, evaluate_data, epoch_id):
    #     assert hasattr(self, 'model'), 'Please specify the exact model !'
    #     self.model.eval()
    #     with torch.no_grad():
    #         test_users, test_items = evaluate_data[0], evaluate_data[1]
    #         negative_users, negative_items = evaluate_data[2], evaluate_data[3]
    #         if self.config['use_cuda'] is True:
    #             test_users = test_users.cuda()
    #             test_items = test_items.cuda()
    #             negative_users = negative_users.cuda()
    #             negative_items = negative_items.cuda()

    #     if self.config['use_bachify_eval'] == False:    
    #         test_scores = self.model(test_users, test_items)
    #         negative_scores = self.model(negative_users, negative_items)
    #     else:
    #         test_scores = []
    #         negative_scores = []
    #         bs = self.config['batch_size']
    #         for start_idx in range(0, len(test_users), bs):
    #             end_idx = min(start_idx + bs, len(test_users))
    #             batch_test_users = test_users[start_idx:end_idx]
    #             batch_test_items = test_items[start_idx:end_idx]
    #             test_scores.append(self.model(batch_test_users, batch_test_items))
    #         for start_idx in tqdm(range(0, len(negative_users), bs)):
    #             end_idx = min(start_idx + bs, len(negative_users))
    #             batch_negative_users = negative_users[start_idx:end_idx]
    #             batch_negative_items = negative_items[start_idx:end_idx]
    #             negative_scores.append(self.model(batch_negative_users, batch_negative_items))
    #         test_scores = torch.concatenate(test_scores, dim=0)
    #         negative_scores = torch.concatenate(negative_scores, dim=0)


    #         if self.config['use_cuda'] is True:
    #             test_users = test_users.cpu()
    #             test_items = test_items.cpu()
    #             test_scores = test_scores.cpu()
    #             negative_users = negative_users.cpu()
    #             negative_items = negative_items.cpu()
    #             negative_scores = negative_scores.cpu()
    #         self._metron.subjects = [test_users.data.view(-1).tolist(),
    #                              test_items.data.view(-1).tolist(),
    #                              test_scores.data.view(-1).tolist(),
    #                              negative_users.data.view(-1).tolist(),
    #                              negative_items.data.view(-1).tolist(),
    #                              negative_scores.data.view(-1).tolist()]
    #     hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
    #     self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
    #     self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
    #     print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
    #     return hit_ratio, ndcg

    def save(self, alias, epoch_id, accuracy):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, accuracy)
        save_checkpoint(self.model, model_dir)