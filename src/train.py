import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import matplotlib.pyplot as plt

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 200,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 943,
              'num_items': 1682,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': False,
              'use_bachify_eval': False,
              'device_id': 6,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 50,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': False,
              'use_bachify_eval': False,
              'device_id': 6,
              'pretrain': False,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': 50,
                'batch_size': 256, # 64 change here
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 226570, #943 , 226570 ,6040 change here
                'num_items': 231637, #1682 , 231637 , 3706 change here
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,
                'device_id': 2,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_Accuracy{:.4f}.model'
                }


def plot_losses(train_losses, val_losses, mode):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{mode}_Classification_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/NCF_loss.png')
    plt.close()
    print(f"Plots saved at plots/NCF_loss.png")
    
def plot_metrics(train_accs, val_accs, mode):
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{mode} Accuracy')
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots/NCF_acc.png')
    plt.close()
    print(f"Plots saved at plots/NCF_acc.png")


# Load Data
ml1m_dir = '/raid/home/shyamsg/Final_Project/NCF/ml-1m (1)/ml-1m/food.dat'
# ml1m_dir = '/raid/home/shyamsg/Final_Project/NCF/ml-100k/ml-100k/u.data'
ml1m_rating = pd.read_csv(ml1m_dir, sep='\t', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')


print("Ratings shape:", ml1m_rating.shape)
# drop nulls in the dataset
ml1m_rating = ml1m_rating.dropna()

# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

# in the dataset i want to set all 0 ratings to 1
ml1m_rating['rating'] = ml1m_rating['rating'].apply(lambda x: 1 if x == 0 else x)

print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
# n_test specifies number of most recent interactions to be used for testing for each user
sample_generator = SampleGenerator(ml1m_rating,n_test=2)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
config = neumf_config
engine = NeuMFEngine(config)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
        
for epoch in range(config['num_epoch']):
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['batch_size'])
    test_loader = sample_generator.instance_a_test_loader(config['batch_size'])
    train_loss, train_acc = engine.train_an_epoch(train_loader, epoch_id=epoch)
    val_loss,accuracy = engine.evaluate(test_loader, epoch_id=epoch)
    # accuracy = engine.evaluate(evaluate_data, epoch_id=epoch)
    # engine.save(config['alias'], epoch, accuracy)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(accuracy)

 # save the train_losses and val_losses lists in some file so that we can plot them later
# with open(f'model_metrics/NCF_metrics.txt', 'w') as f:
#     f.write('Train Losses:\n')
#     f.write(str(train_losses))
#     f.write('\nVal Losses:\n')
#     f.write(str(val_losses))
#     f.write('\nTrain Accuracies:\n')
#     f.write(str(train_accs))
#     f.write('\nVal Accuracies:\n')
#     f.write(str(val_accs))
# print(f"Metrics saved at model_metrics/NCF_metrics.txt")

# plot_losses(train_losses, val_losses, mode="NCF")
# plot_metrics(train_accs, val_accs,mode="NCF")
      

