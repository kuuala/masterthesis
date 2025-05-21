import random

import learn2learn as l2l
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchviz import make_dot


class MetaLearner:
    def __init__(self,
                 ways=5,
                 shots=1,
                 meta_lr=0.003,
                 fast_lr=0.5,
                 meta_batch_size=32,
                 adaptation_steps=1,
                 num_iterations=60000,
                 cuda=True,
                 seed=42,
                 data_root='~/data'):
        self.ways = ways
        self.shots = shots
        self.meta_lr = meta_lr
        self.fast_lr = fast_lr
        self.meta_batch_size = meta_batch_size
        self.adaptation_steps = adaptation_steps
        self.num_iterations = num_iterations
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.data_root = data_root

        self._set_seed()
        self._prepare_tasksets()
        self._build_model()

        self.train_errors = []
        self.train_accuracies = []
        self.valid_errors = []
        self.valid_accuracies = []

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def _prepare_tasksets(self):
        self.tasksets = l2l.vision.benchmarks.get_tasksets(
            'mini-imagenet',
            train_samples=2 * self.shots,
            train_ways=self.ways,
            test_samples=2 * self.shots,
            test_ways=self.ways,
            root=self.data_root,
        )

    def _build_model(self):
        self.model = l2l.vision.models.MiniImagenetCNN(self.ways).to(self.device)
        self.maml = l2l.algorithms.MAML(self.model, lr=self.fast_lr, first_order=False)
        self.opt = optim.Adam(self.maml.parameters(), self.meta_lr)
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def fast_adapt(self, batch, learner):
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)

        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(self.shots * self.ways) * 2] = True
        adaptation_indices = torch.from_numpy(adaptation_indices)
        evaluation_indices = ~adaptation_indices

        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

        for _ in range(self.adaptation_steps):
            adaptation_error = self.loss(learner(adaptation_data), adaptation_labels)
            learner.adapt(adaptation_error)

        predictions = learner(evaluation_data)
        evaluation_error = self.loss(predictions, evaluation_labels)
        evaluation_accuracy = self.accuracy(predictions, evaluation_labels)
        return evaluation_error, evaluation_accuracy

    def train(self):
        for iteration in range(self.num_iterations):
            self.opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            for _ in range(self.meta_batch_size):
                learner = self.maml.clone()
                batch = self.tasksets.train.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                learner = self.maml.clone()
                batch = self.tasksets.validation.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            meta_train_error /= self.meta_batch_size
            meta_train_accuracy /= self.meta_batch_size
            meta_valid_error /= self.meta_batch_size
            meta_valid_accuracy /= self.meta_batch_size

            self.train_errors.append(meta_train_error)
            self.train_accuracies.append(meta_train_accuracy)
            self.valid_errors.append(meta_valid_error)
            self.valid_accuracies.append(meta_valid_accuracy)

            print(f"Iteration {iteration}: "
                  f"Train Error={meta_train_error:.4f}, Train Acc={meta_train_accuracy:.4f}, "
                  f"Valid Error={meta_valid_error:.4f}, Valid Acc={meta_valid_accuracy:.4f}")

            for p in self.maml.parameters():
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            self.opt.step()

    def test(self):
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for _ in range(self.meta_batch_size):
            learner = self.maml.clone()
            batch = self.tasksets.test.sample()
            evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        meta_test_error /= self.meta_batch_size
        meta_test_accuracy /= self.meta_batch_size

        print(f"Meta Test Error: {meta_test_error:.4f}")
        print(f"Meta Test Accuracy: {meta_test_accuracy:.4f}")

    def visualize_training(self):
        epochs = range(len(self.train_errors))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_errors, label='Train Error')
        plt.plot(epochs, self.valid_errors, label='Validation Error')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Meta Train and Validation Error')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.valid_accuracies, label='Validation Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Meta Train and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_model(self):
        dummy_input = torch.randn(1, 3, 84, 84).to(self.device)
        output = self.model(dummy_input)
        dot = make_dot(output, params=dict(self.model.named_parameters()))
        dot.format = 'png'
        dot.render('model_architecture')
        print("Модель сохранена в файл model_architecture")


if __name__ == '__main__':
    meta_learner = MetaLearner(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=100,
        cuda=False,
        seed=42,
        data_root='~/data'
    )
    meta_learner.visualize_model()
    meta_learner.train()
    meta_learner.test()
    meta_learner.visualize_training()
