import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model import NN
from data_loader import DataLoader
import os

class Trainer:
    def __init__(self, problem_idx):
        self.model = NN()
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False,
                                # device_count={'GPU': 0},
                                )
        config.gpu_options.allow_growth = True
        # config.log_device_placement=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.data_loader = DataLoader(problem_idx=problem_idx, shuffle=True)
        #print(self.data_loader.name)
        self.features = self.data_loader.features
        self.train_set_size = self.features.shape[0]
        self.labels = self.data_loader.labels
        self.saver = tf.train.Saver()

    def next_batch(self, batch_size, batch_idx):
        batch_x = self.features[batch_idx * batch_size:(batch_idx+1) * batch_size]
        batch_y = self.labels[batch_idx * batch_size:(batch_idx+1) * batch_size]
        #batch_x = np.reshape(batch_x, (batch_size, -1))
        return batch_x, batch_y

    def normalize(self,):
        self.min_features = np.min(self.features, axis=0)
        self.max_featuers = np.max(self.features, axis=0)
        self.features = (self.features - self.min_features) * 1.0 / (self.max_featuers - self.min_features)

    def train(self, ):
        learning_rate = 1e-3
        batch_size = 16
        print(">>>>>>>>>>>>>>> begin training <<<<<<<<<<<<<<<<<")
        total_loss_lst = []
        for epoch in range(300):
            n_batch = int(self.train_set_size / batch_size)
            loss_lst = []
            clasloss_lst = []
            l2regloss_lst = []
            for i in range(n_batch):
                batch_x, batch_y = self.next_batch(batch_size, i)
                #print(batch_x.shape, batch_y.shape)

                _, loss, clas_loss, l2reg_loss, output = self.sess.run(
                    [self.model.optimizer, self.model.loss, self.model.clas_loss, self.model.l2reg_loss, self.model.output_y],
                    {self.model.input_x: batch_x, self.model.input_y: batch_y, self.model.lr: learning_rate})
                
                e_x = np.exp(output - np.max(output, axis=1).reshape(-1, 1))
                prob = e_x / np.sum(e_x, axis=1).reshape(-1, 1)

                loss_lst.append(loss)
                clasloss_lst.append(clas_loss)
                l2regloss_lst.append(l2reg_loss)
                total_loss_lst.append(loss)
            print("epoch:", epoch, "training loss:", sum(loss_lst) / len(loss_lst), "classification loss:", sum(clasloss_lst) / len(clasloss_lst),
                  "l2 regression loss:", sum(l2regloss_lst) / len(l2regloss_lst), "now_lr:", learning_rate)
            learning_rate *= 0.95
        print(">>>>>>>>>>>>>>> training end <<<<<<<<<<<<<<<<<")
        self.saver.save(self.sess, f"./saved_model/{self.data_loader.name}/model")


#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if __name__ == "__main__":
    trainer = Trainer(problem_idx=0)
    #print(trainer.features[0], trainer.features[1])
    trainer.train()
