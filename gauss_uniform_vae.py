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


username = pwd.getpwuid(os.geteuid()).pw_name
project_name = 'vae_bbo'

if "gpu" in socket.gethostname():
    root_dir = os.path.join('/home/dsi/', username, 'data', project_name)
elif "root" == username:
    root_dir = os.path.join('/workspace/data', project_name)
else:
    root_dir = os.path.join('/data/', username, project_name)


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

        if self.vae_mode == 'gaussian':
            return (-torch.log(std) + (std ** 2 + mu ** 2) / 2 - 0.5).sum(dim=1)
        elif self.vae_mode == 'uniform':
            ub = torch.tanh(mu + std)
            lb = torch.tanh(mu - std)
            return - torch.log((ub - lb) / 2).sum(dim=1)
        else:
            raise NotImplementedError

    def reparameterize(self, mu, std):
        if self.training:
            if self.vae_mode == 'gaussian':
                eps = torch.randn_like(std)
                return mu + eps * std
            elif self.vae_mode == 'uniform':
                eps = torch.ones_like(std).uniform_(-1, 1)
                ub = torch.tanh(mu + std)
                lb = torch.tanh(mu - std)
                return (lb + ub) / 2 + (ub - lb) / 2 * eps
            else:
                raise NotImplementedError

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
        self.vae_mode = vae_mode
        vae_base_dir = os.path.join(root_dir, 'vae_bbo', 'vae_' + vae_mode)
        is_cuda = torch.cuda.is_available()
        torch.manual_seed(128)
        self.device = torch.device("cuda" if is_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
        self.batch_size = 128
        self.epochs = 50
        self.log_interval = 10

        self.model = VAE(vae_mode).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model_path = os.path.join(vae_base_dir, 'vae_model')
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        data_path = os.path.join(vae_base_dir, 'data')
        self.results = os.path.join(vae_base_dir, 'results')

        pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.results).mkdir(parents=True, exist_ok=True)

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transforms.ToTensor()),
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
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            with torch.no_grad():
                if self.vae_mode == 'gaussian':
                    sample = torch.randn(self.batch_size, args.latent).to(self.device)
                elif self.vae_mode == 'uniform':
                    sample = torch.FloatTensor(self.batch_size, args.latent).uniform_(-1, 1).to(self.device)
                else:
                    raise NotImplementedError
                sample = self.model(sample, part='dec').cpu()
                sample = torch.sigmoid(sample)
                save_image(sample.view(self.batch_size, 1, 28, 28), os.path.join(self.results, 'sample_' + str(epoch) + '.png'))

        self.save_model()


class VaeProblem(object):
    def __init__(self, problem_index):
        dim = args.latent
        self.vae = VaeModel(args.vae)
        self.vae.load_model()
        self.vae.model.eval()
        self.problem = None

        suite_name = "bbob"
        suite_filter_options = ("dimensions: " + str(dim))
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
        self.initial_solution = self.vae.model.decode(torch.FloatTensor(self.problem.initial_solution).to(self.device)).detach().cpu().numpy()
        self.lower_bounds = -np.ones(self.dimension)
        self.upper_bounds = np.ones(self.dimension)
        self.evaluations = 0
        self.final_target_hit = self.problem.final_target_hit

    def constraint(self, x):
        return None

    def denormalize(self, policy):
        assert (np.max(policy) <= 1) or (np.min(policy) >= -1), "denormalized"
        if len(policy.shape) == 2:
            assert (policy.shape[1] == self.output_size), "action error"
            upper = np.repeat(self.z_upper_bounds, policy.shape[0], axis=0)
            lower = np.repeat(self.z_lower_bounds, policy.shape[0], axis=0)
        else:
            upper = self.z_upper_bounds.flatten()
            lower = self.z_lower_bounds.flatten()

        policy = 0.5 * (policy + 1) * (upper - lower) + lower
        return policy

    def func(self, x):
        x = torch.FloatTensor(x).to(self.device)
        mu, logvar = self.vae.model.encode(x)
        z = self.vae.model.reparameterize(mu, logvar).detach().cpu().numpy()
        z = self.denormalize(z)
        z = np.clip(z, self.z_lower_bounds, self.z_upper_bounds)
        f_val = self.problem(z)

        self.best_observed_fvalue1 = self.problem.best_observed_fvalue1
        self.evaluations += 1
        self.final_target_hit = self.problem.final_target_hit

        return f_val


if __name__ == "__main__":
    vae = VaeModel(args.vae)
    vae.run_vae()

