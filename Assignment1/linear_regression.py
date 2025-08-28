import numpy as np

class LinearRegression():

    def __init__(self):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.L = 0.001
        self.epochs = 100
        self.w = 0
        self.b = 0

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        m = X.shape[0]

        #Gradient Descent
        for _ in range(self.epochs):
            y_pred = self.w*X + self.b #Prediction
            grad_w = (-2/m)*sum(X * (y - y_pred)) #Derivative of loss with respect to w
            grad_b = (-2/m)*sum(y - y_pred) #Derivative of loss with respect to b
            self.w -= self.L*grad_w #Updating w
            self.b -= self.L*grad_b #Updating b
    
    def predict(self, X):
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
        y_pred = self.w*X + self.b
        return y_pred
    
    # def __init__(self):
    #     # NOTE: Feel free to add any hyperparameters 
    #     # (with defaults) as you see fit
    #     self.learning_rate = 0.001
    #     self.epochs = 100
    #     self.weights, self.bias = None, None
    #     self.losses, self.train_accuracies = [], []

    # def MSE(self, y, y_pred):
    #     return 1/2
    
    # def compute_gradientAs(self, x, y, y_pred):
    #     return 

    # def update_parameters(self, grad_w, grad_b):
    #     self.weights -= self.learning_rate*grad_w
    #     self.weights -= self.learning_rate*grad_b

    # def _compute_loss(self, y, y_pred):
    #     return self.MSE(y, y_pred)
    
    # def accuracy(y, predictions):
    #     return np.mean(y == predictions)

    # def fit(self, X, y):
    #     """
    #     Estimates parameters for the classifier
        
    #     Args:
    #         X (array<m,n>): a matrix of floats with
    #             m rows (#samples) and n columns (#features)
    #         y (array<m>): a vector of floats
    #     """
    #     # TODO: Implement

    #     self.weights = np.zeros(X.shape[1])
    #     self.bias = 0
        
    #     # Gradient Descent
    #     for _ in range(self.epochs):
    #         y_pred = np.matmul(self.weights, X) + self.bias
    #         grad_w, grad_b = self.compute_gradients(X, y, y_pred)
    #         self.update_parameters(grad_w, grad_b)
    #         loss = self._compute_loss(y, y_pred)
    #         pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
    #         self.train_accuracies.append(self.accuracy(y, pred_to_class))
    #         self.losses.append(loss)
    
    # def predict(self, X):
    #     """
    #     Generates predictions
        
    #     Note: should be called after .fit()
        
    #     Args:
    #         X (array<m,n>): a matrix of floats with 
    #             m rows (#samples) and n columns (#features)
            
    #     Returns:
    #         A length m array of floats
    #     """
    #     # TODO: Implement

    #     y_pred = np.matmul(X, self.weights) + self.bias
    #     return [1 if _y > 0.5 else 0 for _y in y_pred]
    