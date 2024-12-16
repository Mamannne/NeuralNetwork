import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import logsumexp
import json

class MLP():
    def __init__(self,din,dout,w=None,b=None):
        if w is not None and b is not None:
            self.W = w
            self.b = b
        else:
            self.W = np.random.rand(dout,din)-0.5
            self.b = np.random.rand(dout) - 0.5

    def forward(self,x):
        self.x = x
        #print(self.x.shape,self.W.shape)
        return self.x @ self.W.T + self.b

    def backward(self,y):
        self.deltaW = y.T @ self.x
        self.deltab = y.sum(0)
        return y @ self.W
    
    def save(self):
        with open('save.json','wb') as f:
            data_to_parse = {'W':self.W,'b':self.b}
            json.dump(data_to_parse,f)


class SequentialNN():
    def __init__(self,layers: list):
        self.layers = layers

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,y):
        for layer in self.layers[::-1]:
            y = layer.backward(y)
        return y
    
    def save(self):
        with open('save.json','w') as f:
            f.write('{}')
        for layer in self.layers:
            if layer.__class__ == MLP:
                data_to_add = {
                                    'W':layer.W.tolist(),
                                    'b':layer.b.tolist()
                               }
                with open('save.json','r') as f:
                    data_read = json.load(f)
                with open('save.json','w') as f:
                    if data_read == {}:
                        data_to_parse = {
                                        '1': data_to_add
                                     }
                    else:
                        data_to_parse = {
                                        '1': data_read["1"],
                                        '2': data_to_add
                                     }
                    json.dump(data_to_parse, f)

class ReLU():
    def forward(self,x):
        self.x = x
        return np.maximum(0,x)

    def backward(self,y):
        u = y.copy()
        u[self.x<0]=0.
        return u
    
class logSoftmax():
    def forward(self,x):
        self.x = x
        return x - logsumexp(x, axis=1)[..., None]

    def backward(self, gradout):
        gradients = np.eye(self.x.shape[1])[None, ...]
        gradients = gradients - (np.exp(self.x) / np.sum(np.exp(self.x), axis=1)[..., None])[..., None]
        return (np.matmul(gradients, gradout[..., None]))[:, :, 0]
    
class NNLoss():
    def forward(self,pred,truth):
        self.pred = pred
        self.truth = truth

        loss = 0
        for b in range(pred.shape[0]):
            loss -= self.pred[b,self.truth[b]]
        return loss

    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((self.pred.shape[0],din))
        for b in range(self.pred.shape[0]):
            jacobian[b,self.truth[b]] = -1

        return jacobian

    def __call__(self,pred,truth):
        return self.forward(pred,truth)


class Optimizer():
    def __init__(self,alpha,sequential_nn: SequentialNN):
        self.alpha = alpha
        self.sequential_nn = sequential_nn

    def run(self):
        for layer in self.sequential_nn.layers:
            if layer.__class__ == MLP:
                layer.W = layer.W - self.alpha*layer.deltaW
                layer.b = layer.b - self.alpha*layer.deltab

def train(model,optimizer,trainX,trainY,loss_fct = NNLoss(),nb_epochs=14000, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        batch_idx = [np.random.randint(0,trainX.shape[0]) for _ in range(batch_size)]
        x = trainX[batch_idx]
        target = trainY[batch_idx]
        prediction = model.forward(x)
        loss_value = loss_fct(prediction,target)
        training_loss.append(loss_value)
        gradout = loss_fct.backward()
        model.backward(gradout)
        optimizer.run()
    return training_loss

if __name__ == '__main__':
    data = pd.read_csv('data_set/train.csv')

    data = np.array(data)
    m,n = data.shape 
    np.random.shuffle(data)

    data_dev = data[:1000]
    Y_dev = data_dev[:,0]
    X_dev = data_dev[:,1:] / 255

    data_train = data[1000:]
    Y_train = data_train[:,0]
    X_train = data_train[:,1:] / 255

    mlp = SequentialNN([MLP(784,10),ReLU(),MLP(10,10),logSoftmax()])

    opti = Optimizer(1e-3,mlp)

    loss = train(mlp,opti,X_train,Y_train)

    mlp.save()

    accuracy = 0

    for i in range(X_dev.shape[0]):
        prediction = mlp.forward(X_dev[i].reshape(1,784)).argmax()
        if prediction == Y_dev[i]: accuracy += 1
    print('Test accuracy', accuracy / X_dev.shape[0]*100, '%')

