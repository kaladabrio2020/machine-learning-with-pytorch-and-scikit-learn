import torch

class Perceptron:
    '''Perceptron Classifier

    Parameters
    ----------
    eta : float
        taxa de aprendizado (default: 0.01)
    
    max_iter : int
        número máximo de iterações (default: 1000)
    
    random_state : int
        semente para geração de números aleatórios (default: None)
    
    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        pesos do modelo após o treinamento
    bias_ : float
        bias do modelo após o treinamento
    '''
    def __init__(self, eta=0.01, max_iter=1000, random_state=1):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        '''Treina o modelo Perceptron

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            dados de entrada
        
        y : array-like, shape (n_samples,)
            rótulos de classe
        '''
        

        # definindo a semente para reprodutibilidade
        torch.manual_seed(self.random_state)

        # inicializando pesos e bias
        self.weights_ = torch.normal(0, 1, (X.shape[1], 1))
        
        self.bias_    = torch.zeros(1)

        self.errors_  = []


        for _ in range(self.max_iter):
            errors = 0 

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))

                # atualizando pesos e bias
                self.weights_ += update * xi.view(-1, 1)
                self.bias_    += update

                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        '''Calcula a entrada líquida do modelo

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            dados de entrada
        
        Returns
        -------
        net_input : array-like, shape (n_samples,)
            entrada líquida do modelo
        '''
        return torch.matmul(X, self.weights_) + self.bias_
    
    def predict(self, X):
        '''Faz previsões com o modelo Perceptron

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            dados de entrada
        
        Returns
        -------
        predictions : array-like, shape (n_samples,)
            previsões do modelo
        '''
        return torch.where(self.net_input(X) >= 0.0, 1, -1).squeeze()