import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import yaml
import time
import os
from utils.helper import to_python_native


class TrainModel():
    def __init__(self, 
                 kernel,
                 training_data,
                 training_labels,
                 testing_data,
                 testing_labels,
                 optimizer,
                 lr,
                 epochs,
                 evaluation_method,
                 ):
        super().__init__()
        self._kernel = kernel
        self._optimizer = optimizer
        self._epochs = epochs
        self._training_data = training_data
        self._training_labels = training_labels.to(torch.float32)
        self._testing_data = testing_data
        self._testing_labels = testing_labels.to(torch.float32)
        self._n_classes = torch.unique(training_labels)
        self._kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, self._kernel)
        self._executions = 0
        self._lr = lr
        self._loss_arr = []
        self.alignment_arr = []
        self.initial_training_accuracy = None
        self.final_training_accuracy = None
        self.initial_testing_accuracy = None
        self.final_testing_accuracy = None
        self._per_epoch_executions = None
        self._method = evaluation_method

        if optimizer == 'adam':
            self._kernel_optimizer = optim.Adam(self._kernel.parameters(), lr=self._lr)
        elif optimizer == 'gd':
            self._kernel_optimizer = optim.SGD(self._kernel.parameters(), lr=self._lr)

        if self._method == 'alignment':
            self._loss_function = self._loss_ta  # Single assignment instead of multiple conditions
        elif self._method == 'seperability':
            self._loss_function = self._loss_sk
        elif self._method == 'polarisation':
            self._loss_function = self._loss_polarisaion

    def _loss_ta(self, K, y):
        """Compute kernel alignment loss."""
        N = y.shape[0]
        assert K.shape == (N, N), "Shape of K must be (N,N)"
        
        # Ensure y is a tensor
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        yT = y.view(1, -1)  # Transpose of y, shape (1,N)
        Ky = torch.matmul(K, y)  # K*y, shape (N,)
        yTKy = torch.matmul(yT, Ky)  # yT * Ky, shape (1,1)
        K2 = torch.matmul(K, K)  # K^2, shape (N,N)
        trace_K2 = torch.trace(K2)

        result = yTKy / (torch.sqrt(trace_K2) * N)
        return result.squeeze()
    
    
    def _loss_sk(self, K, y):
        """Compute kernel seperability"""
        m = len(self._n_classes)
        sb, sw = torch.zeros((self._training_data.shape[1], self._training_data.shape[1])), torch.zeros((self._training_data.shape[1], self._training_data.shape[1]))
        a = self._training_data.mean(dim = 0)
        for nclass in self._n_classes:
            indices = (y == nclass).nonzero(as_tuple=True)[0]
            K_class = K[indices][:, indices]
            ai = self._training_data[indices].mean(dim = 0)
            li = K_class.shape[0]
            diff_b = (ai - a).unsqueeze(1)
            sb += li * diff_b @ diff_b.T 

            ai = K_class.mean(dim = 0)
            diff_w = (K_class - ai.unsqueeze(0))
            sw += diff_w @ diff_w.T

        trace_sb, trace_sw = torch.trace(sb), torch.trace(sw)
        J = trace_sb / trace_sw if trace_sw > 0 else torch.tensor(float('inf'))

        return J

    def _loss_polarisaion(self, K, y):
        """Compute kernel Polarisation"""
        N = y.shape[0]
        assert K.shape == (N, N), "Shape of K must be (N,N)"

        sum_same_class = 0
        sum_diff_class = 0

        for i in range(N):
            for j in range(N):
                if y[i] == y[j]:
                    sum_same_class += K[i, j]
                else:
                    sum_diff_class += K[i, j]

        return sum_same_class - sum_diff_class

    def _loss_feature_space(self):
        pass


    def fit_kernel(self):
        """Train kernel by maximizing alignment."""
        training_data = self._training_data
        training_labels = self._training_labels

        for epoch in range(self._epochs):
            x_0 = training_data.repeat(training_data.shape[0], 1)
            x_1 = training_data.repeat_interleave(training_data.shape[0], dim=0)

            self._kernel_optimizer.zero_grad()
            K = self._kernel(x_0, x_1).to(torch.float32)

            # Ensure labels are float32
            training_labels = training_labels.to(torch.float32)

            current_alignment = self._loss_function(K.reshape(training_data.shape[0], training_data.shape[0]), training_labels)
            loss = -current_alignment
            loss.backward()
            self._kernel_optimizer.step()
            self._loss_arr.append(loss.item())
            self.alignment_arr.append(current_alignment)

            print("------------------------------------------------------------------")
            print(f"Epoch: {epoch}, Alignment: {current_alignment}")
            print("------------------------------------------------------------------")

    def evaluate(self):
        """Evaluate kernel alignment and SVM classification."""
        test_data = self._testing_data
        test_labels = self._testing_labels

        # Compute kernel matrix for training
        x_0 = self._training_data.repeat(self._training_data.shape[0], 1)
        x_1 = self._training_data.repeat_interleave(self._training_data.shape[0], dim=0)
        _matrix = self._kernel(x_0, x_1).to(torch.float32).reshape(self._training_data.shape[0], self._training_data.shape[0])

        # Ensure labels are tensor
        if not isinstance(self._training_labels, torch.Tensor):
            self._training_labels = torch.tensor(self._training_labels, dtype=torch.float32)

        current_alignment = self._loss_function(_matrix, self._training_labels)

        # Convert to numpy for SVM
        _matrix = _matrix.detach().numpy()
        training_labels_np = self._training_labels.detach().numpy()

        # Train SVM on precomputed kernel
        self._model = SVC(kernel='precomputed', max_iter=10000).fit(_matrix, training_labels_np)
        predictions = self._model.predict(_matrix)
        training_accuracy = accuracy_score(training_labels_np, predictions)

        # Compute kernel matrix for testing
        x_0 = self._testing_data.repeat_interleave(self._training_data.shape[0], dim=0)
        x_1 = self._training_data.repeat(test_data.shape[0], 1)
        _matrix = self._kernel(x_0, x_1).to(torch.float32).reshape(test_data.shape[0], self._training_data.shape[0])

        # Convert to numpy for SVM
        _matrix = _matrix.detach().numpy()
        test_labels_np = test_labels.detach().numpy()

        predictions = self._model.predict(_matrix)
        accuracy = accuracy_score(test_labels_np, predictions)
        f1 = f1_score(test_labels_np, predictions, average='weighted')

        metrics = {
            'alignment': current_alignment.item(),
            'training_accuracy': training_accuracy,
            'testing_accuracy': accuracy,
            'f1_score': f1,
            'alignment_arr': self.alignment_arr,
            'loss_arr': self._loss_arr
        }
        return metrics
