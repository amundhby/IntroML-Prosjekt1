import numpy as np

class LogisticRegression():
    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = 0.05
        self.epochs = 1000
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []

    def sigmoid_function(self, lin_model):
        y_pred = 1/(1 + np.exp(-lin_model))
        return y_pred
    
    def _compute_loss(self, y, y_pred):
        y_pred_fixed = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = -y*np.log(y_pred_fixed) - (1-y)*np.log(1-y_pred_fixed)
        return np.mean(loss)
    
    def compute_gradients(self, x, y, y_pred):
        grad_w = np.matmul((y_pred - y), x)/x.shape[0]
        grad_b = np.mean(y_pred - y)
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate*grad_w
        self.bias -= self.learning_rate*grad_b

    
    def accuracy(self, y, predictions):
        return np.mean(y == predictions)
    
    def formatting_data(self, train):
        X_formatted = train[['x0', 'x1']].to_numpy()
        X0 = train[['x0']].to_numpy()
        X1 = train[['x1']].to_numpy()

        X_formatted = np.hstack((X_formatted, X0*X1))

        y = train['y'].to_numpy()

        # X_formatted = []

        # for i in range(0, len(X)):
        #     X_formatted.append(X[i][0]*X[i][1])

        return X_formatted, y

    def fit(self, train, formatting):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement

        if formatting:
            X, y = self.formatting_data(train)
        else:
            X = train[['x0', 'x1']].to_numpy()
            y = train['y'].to_numpy()

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = np.matmul(X, self.weights) + self.bias
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)
    
    def predict(self, train, formatting):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement

        if formatting:
            X, y = self.formatting_data(train)
        else:
            X = train[['x0', 'x1']].to_numpy()
            y = train['y'].to_numpy()

        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]