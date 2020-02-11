from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import pwd
import cocoex
import numpy as np
from config import args
import pathlib
import socket
from tqdm import tqdm
from config import consts

class VAE(nn.Module):
    def __init__(self, vae_mode):
        super(VAE, self).__init__()
        self.vae_mode = vae_mode

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
        )

        self.mu = nn.Linear(400, args.latent)
        self.std = nn.Sequential(
                    nn.Linear(400, args.latent),
                    nn.Softplus())

        self.decoder = nn.Sequential(nn.Linear(args.latent, 400),
                                     nn.ReLU(),
                                     nn.Linear(400, 784))

    def kl(self, mu, std):

        return (-torch.log(std) + (std ** 2 + mu ** 2) / 2 - 0.5).sum(dim=1)

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return mu + eps * std


        else:
            return mu

    def forward(self, x, part='all'):

        if part in ['all', 'enc']:

            x = self.encoder(x)
            mu = self.mu(x)
            std = self.std(x)
            x = self.reparameterize(mu, std)

            kl = self.kl(mu, std)

        if part in ['all', 'dec']:
            x = self.decoder(x)

        if part in ['all', 'enc']:
            return x, mu, std, kl
        elif part == 'dec':
            return x
        else:
            return NotImplementedError


class VaeModel(object):
    def __init__(self, vae_mode):
        root_dir = consts.vaedir
        self.vae_mode = vae_mode
        is_cuda = torch.cuda.is_available()
        torch.manual_seed(128)
        self.device = torch.device("cuda" if is_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
        self.batch_size = 128
        self.epochs = 50
        self.log_interval = 10

        self.model = VAE(vae_mode).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model_path = os.path.join(root_dir, 'vae_' + vae_mode+'_model')
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        data_path = os.path.join(root_dir, 'data')
        self.results = os.path.join(root_dir, 'results_' + vae_mode)

        pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.results).mkdir(parents=True, exist_ok=True)

        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, **kwargs)

    def save_model(self):
        state = {'model': self.model,
                 'optimizer': self.optimizer.state_dict()}

        torch.save(state, self.model_path)

    def load_model(self):
        if not os.path.exists(self.model_path):
            assert False, "load_model"

        state = torch.load(self.model_path)

        self.model = state['model'].to(self.device)
        self.optimizer.load_state_dict(state['optimizer'])

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            x = data.to(self.device).view(len(data), -1)
            self.optimizer.zero_grad()
            x_hat, _, _, kl = self.model(x)
            loss = self.loss(x_hat, x).sum(dim=1).mean() + kl.mean()
            loss.backward()
            self.optimizer.step()

            train_loss += float(loss)
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                x = data.to(self.device).view(len(data), -1)
                x_hat, _, _, kl = self.model(x)
                test_loss += float(self.loss(x_hat, x).sum(dim=1).mean() + kl.mean())
                if i == 0:
                    n = min(x.size(0), 8)
                    x_hat = torch.sigmoid(x_hat)
                    comparison = torch.cat([x[:n].view(n, 1, 28, 28), x_hat[:n].view(n, 1, 28, 28)])
                    save_image(comparison.cpu(), os.path.join(self.results, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def run_vae(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.train(epoch)
            self.test(epoch)
            with torch.no_grad():
                sample = torch.randn(self.batch_size, args.latent).to(self.device)
                sample = self.model(sample, part='dec').cpu()
                sample = torch.sigmoid(sample)
                save_image(sample.view(self.batch_size, 1, 28, 28), os.path.join(self.results, 'sample_' + str(epoch) + '.png'))

        self.save_model()


class VaeProblem(object):
    def __init__(self, problem_index):
        self.latent = args.latent
        self.vae = VaeModel(args.vae)
        self.vae.load_model()
        self.vae.model.eval()
        self.problem = None

        suite_name = "bbob"
        suite_filter_options = ("dimensions: " + str(self.latent))
        self.suite = cocoex.Suite(suite_name, "", suite_filter_options)
        self.reset(problem_index)

    def reset(self, problem_index):
        self.suite.reset()
        self.problem = self.suite.get_problem(problem_index)

        self.z_upper_bounds = self.problem.upper_bounds
        self.z_lower_bounds = self.problem.lower_bounds

        self.device = self.vae.device
        self.best_observed_fvalue1 = self.problem.best_observed_fvalue1
        self.index = self.problem.index
        self.id = 'vae_'+str(self.problem.id)
        self.dimension = 784
        self.lower_bounds = -5*torch.ones(self.dimension, dtype=torch.float)
        self.upper_bounds = 5*torch.ones(self.dimension, dtype=torch.float)
        #self.initial_solution = torch.min(torch.max(self.vae.model(torch.cuda.FloatTensor(self.problem.initial_solution), part='dec'), self.lower_bounds), self.upper_bounds).detach()/5
        self.initial_solution = torch.zeros(self.dimension, dtype=torch.float)
        self.evaluations = 0
        self.final_target_hit = self.problem.final_target_hit

    def constraint(self, x):
        return None

    def denormalize(self, policy):
        assert (np.max(policy) <= 1) or (np.min(policy) >= -1), "policy shape {} min {} max {}".format(policy.shape, policy.min(), policy.max())
        if len(policy.shape) == 2:
            assert (policy.shape[1] == self.latent), "action error"
            upper = np.repeat(self.z_upper_bounds, policy.shape[0], axis=0)
            lower = np.repeat(self.z_lower_bounds, policy.shape[0], axis=0)
        else:
            upper = self.z_upper_bounds.flatten()
            lower = self.z_lower_bounds.flatten()

        policy = 0.5 * (policy + 1) * (upper - lower) + lower
        return policy

    def func(self, x):
        z, _, _, _ = self.vae.model(x.unsqueeze(0), 'enc')
        z = z.detach().cpu().numpy()
        #z = self.denormalize(z).flatten()
        z = np.clip(z.flatten(), a_min=self.z_lower_bounds, a_max=self.z_upper_bounds)
        f_val = self.problem(z)

        self.best_observed_fvalue1 = self.problem.best_observed_fvalue1
        self.evaluations += 1
        self.final_target_hit = self.problem.final_target_hit

        return f_val


if __name__ == "__main__":
    vae = VaeModel(args.vae)
    vae.run_vae()

