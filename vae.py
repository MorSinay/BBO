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

username = pwd.getpwuid(os.geteuid()).pw_name
vae_base_dir = os.path.join('/data/', username, 'gan_rl', 'vae')


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #return mu + eps*std
        return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VaeModel(object):
    def __init__(self):

        is_cuda = torch.cuda.is_available()
        torch.manual_seed(128)
        self.device = torch.device("cuda" if is_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
        self.batch_size = 128
        self.epochs = 10
        self.log_interval = 10

        self.model = VAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model_path = os.path.join(vae_base_dir, 'vae_model')

        data_path = os.path.join(vae_base_dir, 'data')
        if not os.path.exists(data_path):
            try:
                os.makedirs(data_path)
            except:
                pass

        self.results = os.path.join(vae_base_dir, 'results')
        if not os.path.exists(self.results):
            try:
                os.makedirs(self.results)
            except:
                pass

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

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
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
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(), os.path.join(self.results, 'reconstruction_' + str(epoch) + '.png'), nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def run_vae(self):
        for epoch in range(1, self.epochs + 1):
            self.train(epoch)
            self.test(epoch)
            with torch.no_grad():
                sample = torch.randn(64, 20).to(self.device)
                sample = self.model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28), os.path.join(self.results, 'sample_' + str(epoch) + '.png'))

        self.save_model()


class VaeProblem(object):
    def __init__(self, problem_index):
        dim = 20
        self.vae = VaeModel()
        self.vae.load_model()
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
        self.initial_solution = self.vae.model.decode(torch.tensor(self.problem.initial_solution, dtype=torch.float).to(self.device)).detach().cpu().numpy()
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
        x = torch.tensor(x, dtype=torch.float).to(self.device)
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
    vae = VaeModel()
    vae.run_vae()

