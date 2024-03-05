import numpy as np
import os
import requests
import zipfile
import pandas as pd

class Data:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), "data")

    def fetch_iris(self):
        url = "https://archive.ics.uci.edu/static/public/53/iris.zip"
        req = requests.get(url)
        
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        with open(os.path.join(self.data_path, "iris.zip"), "wb") as f:
            f.write(req.content)
            
        return os.path.join(self.data_path, "iris.zip")

    def get_data_from_zip(self, zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=self.data_path)

        return pd.read_csv(os.path.join(self.data_path, "iris.data"))

    def train_test_set_split(self, X, y, test_set_size):
        train_set_len = len(X) - int(len(X)*test_set_size) 
        train_X, test_X = X[:train_set_len], X[train_set_len:]
        train_y, test_y = y[:train_set_len], y[train_set_len:]
        return train_X, train_y, test_X, test_y


class Transformer:
    def __init__(self):
        self._label_names = []
        self.sparse_vectors = []
        self._represent_dict = {}
    
    def transform_text_labels(self, labels_list: np.array):
        self._label_names = np.unique(labels_list)
        
        for i, name in enumerate(self._label_names):
            self._represent_dict[name] = i
        
        for label in labels_list:
            sparse_vec = np.zeros((len(self._label_names)))
            sparse_vec[self._represent_dict[label]] = 1
            sparse_vec = sparse_vec.reshape((len(self._label_names), 1))
            self.sparse_vectors.append(sparse_vec)

        self.sparse_vectors = np.array(self.sparse_vectors)


class LinearRegression:
    def __init__(self, loss_func, input_features, output_featurues, learning_rate):
        self.loss_func = loss_func

        self.input_features = input_features
        self.output_featurues = output_featurues

        self.learning_rate = learning_rate

        self.weights = np.random.random((3, 4))
        self.biases = np.random.random((3, 1))
    

    def compute_y_hat(self, X):
        X = np.expand_dims(X, -1)

        result = np.matmul(self.weights, X) + self.biases
        for i in range(0, len(X)):
            result[i] = SoftMax(result[i])

        return result

    def compute_gradients(self, loss, x):
        z = self.compute_y_hat(np.expand_dims(x, 0))
        a = SoftMax(z)
        x = np.expand_dims(x, -1)

        w_grads = (x * np.matmul(SoftMax_gradient(a), 2*loss).T).T
        b_grads = np.matmul(SoftMax_gradient(a), 2*loss)

        return w_grads, b_grads

    def compute_av_grads(self, loss, X):
        av_w_grad, av_b_grad = self.compute_gradients(loss[0], np.array(X[0]))

        for i in range(1, len(X)):
            w_grad, b_grad = self.compute_gradients(loss[i], np.array(X[i]))
            av_w_grad += w_grad
            av_b_grad += b_grad

        av_w_grad /= len(X)
        av_b_grad /= len(X)

        return av_w_grad, av_b_grad

    def pass_gradients(self, w_grads, b_grads):
        self.weights -= self.learning_rate*w_grads
        self.biases -= self.learning_rate*b_grads 

    def fit(self, X_train, y_train, X_test, y_test, epochs):
        self.train_epoch_losses = []
        self.test_epoch_losses = []

        epsilon = 1e-5
        stopping_criteria = 0.01

        for epoch in range(0, epochs):
            
            y_hat = self.compute_y_hat(X_train)

            loss = y_hat - y_train

            cost = -(y_train * np.log(y_hat + epsilon))
            cost = cost.sum(axis=1).mean()

            w_grads, b_grads = self.compute_av_grads(loss, X_train)

            self.pass_gradients(w_grads, b_grads)

            if epoch > 0 and cost - self.train_epoch_losses[-1] > stopping_criteria:
                print("training stopped early")
                print(f"prev loss: {self.train_epoch_losses[-1]}")
                print(f"current loss: {cost}")
                self.train_epoch_losses.append(cost)
                break

            self.test_epoch_losses.append(self.test_score(X_test, y_test))
            self.train_epoch_losses.append(cost)

        print(self.train_epoch_losses[::100])
        print(self.test_epoch_losses[::100])

    def test_score(self, X_test, y_test):
        epsilon = 1e-5

        y_hat = self.compute_y_hat(X_test)
        cost = -(y_test * np.log(y_hat + epsilon))
        cost = cost.sum(axis=1).mean()

        return cost


def MSE(y_hat, y):
    return np.power(y_hat - np.array(y).reshape(y_hat.shape), 2)

def SoftMax(x):
    return np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)))

def SoftMax_gradient(x):
    x = np.ravel(x)

    result = np.zeros((x.shape[0], x.shape[0]))

    for i in range(0, len(x)):
        for j in range(0, len(x)):
            if j==i:
                result[i,j] = x[i]*(1-x[j])
            else:
                result[i,j] += -x[j]*x[i]

    return result

#data thing
data_manager = Data()

zip_file_path = data_manager.fetch_iris()
raw_data = data_manager.get_data_from_zip(zip_file_path)

first_row = pd.DataFrame({
    "sepal_length": [5.1],
    "sepal_width": [3.5], 
    "petal_length": [1.4], 
    "petal_width": [0.2], 
    "label": ["Iris-setosa"]
})

rest_of_rows = pd.DataFrame({
    "sepal_length": raw_data["5.1"].values,
    "sepal_width": raw_data["3.5"].values, 
    "petal_length": raw_data["1.4"].values, 
    "petal_width": raw_data["0.2"].values, 
    "label": raw_data["Iris-setosa"].values
})

data = pd.concat([first_row, rest_of_rows])

data_transformer = Transformer()
data_transformer.transform_text_labels(np.array(data["label"].values))

X = np.array(data.copy().drop("label", axis=1).to_numpy())
y = data_transformer.sparse_vectors

X_train, y_train, X_test, y_test = data_manager.train_test_set_split(X, y, 0.2)

#model thing
model = LinearRegression(MSE, 4, 3, 0.1)
model.fit(X_train, y_train, X_test, y_test, 3000)