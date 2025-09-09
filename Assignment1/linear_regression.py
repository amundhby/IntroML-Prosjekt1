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
    