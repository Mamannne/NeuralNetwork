from train import *
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv('data_set/test.csv')
    data = np.array(data)

    n = np.random.randint(0,data.shape[0])

    X = data[n] / 255

    image = X.reshape((28,28))

    with open('save.json','rb') as f:
        model_params = json.load(f)

    mlp_1 = MLP(784,10,np.array(model_params["1"]["W"]),np.array(model_params["1"]["b"]))
    mlp_2 = MLP(10,10,np.array(model_params["2"]["W"]),np.array(model_params["2"]["b"]))

    model = SequentialNN([mlp_1,ReLU(),mlp_2,logSoftmax()])

    prediction = model.forward(X.reshape(1,784)).argmax()

    plt.imshow(image,cmap='gray')
    plt.title('Prediction: '+str(prediction) + ' for image number '+str(n))
    plt.show()

if __name__ == '__main__':
    main()