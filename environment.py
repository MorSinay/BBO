import numpy as np
from config import args, exp
import torchvision
import torch
import cv2
import dlib
import os
from imutils import face_utils
from collections import namedtuple
from torchvision import transforms
from torch import nn
from model import ResClassifier, ResGenerator, ResDiscriminator
import pandas as pd


class Env(object):

    def __init__(self, ind):

        # lendmark predictor
        self.predictor_path = args.predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.height = args.height
        self.width = args.width
        self.action_space = args.action_space
        self.markers = 68
        self.transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # dataset for generating landmark problem
        self.dataset = torchvision.datasets.ImageFolder(root=exp.dataset_dir)

        # load generator/discriminator and attribute classifier

        generator_path = os.path.join(args.generator_dir, 'checkpoints', 'checkpoint')
        classifier_path = os.path.join(args.classifier_dir, 'checkpoints', 'checkpoint')

        checkpoint = torch.load(generator_path)
        self.generator = checkpoint['generator_model']
        self.generator.module.load_state_dict(checkpoint['generator'])
        self.discriminator = checkpoint['discriminator_model']
        self.discriminator.module.load_state_dict(checkpoint['discriminator'])
        checkpoint = torch.load(classifier_path)
        self.classifier = checkpoint['net_model']
        self.classifier.load_state_dict(checkpoint['net'])

        self.generator.device_ids = list(range(torch.cuda.device_count()))
        self.discriminator.device_ids = list(range(torch.cuda.device_count()))

        self.generator.eval()
        self.discriminator.eval()
        self.classifier.eval()

        self.disc_loss = nn.MSELoss(reduction='none')

        # get attribute file and calculate the pos_weight

        self.att = pd.read_csv(exp.attributes_file)
        self.att_name = sorted(list(self.att.keys()))[:-1]
        self.n_att = len(self.att_name)

        self.pos_weight = torch.ones(self.n_att)
        # calculate pos weights
        for i, name in enumerate(self.att_name):
            self.pos_weight[i] = float(len(self.att) / (self.att[name] > 0).sum())

        self.att_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(exp.device), reduction='none')

        # gan pretrained model
        # self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', model_name='celebAHQ-512',
        #                             pretrained=True, useGPU=True)
        # self.model.netG.eval()

        self.penalty = args.penalty
        self.scale = np.stack([self.width * np.ones(self.markers), self.height * np.ones(self.markers)], axis=1)

        ind = ind if ind > 0 else np.random.randint(len(self.dataset))
        while True:
            self.problem = ind
            image = self.transform(self.dataset[ind][0]).to(exp.device)
            # self.z_target = torch.cuda.FloatTensor(1, self.action_space).normal_()
            # image = self.gen_images(self.z_target)[0].detach()
            landmark_target = self.landmarks(image)
            landmark_target = landmark_target.view(1, *landmark_target.shape)
            if not torch.isnan(landmark_target.sum()).item():
                self.landmark_target = landmark_target
                self.image_target = torch.clamp(0.5 * image + 0.5, 0, 1)

                self.attributes_target = (self.attributes(image) > 0.).float()
                break
            ind += 1

        self.best_policy = None
        self.best_reward = None
        self.best_image = None
        self.k = None
        self.results = namedtuple('env_results', 'image reward')
        self.reset()

    def attributes(self, image):

        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        y_hat = self.classifier(image)

        # if y_hat.shape[0] == 1:
        #     y_hat = y_hat.squeeze(0)

        return y_hat

    def landmarks(self, image):

        if type(image) == torch.Tensor:
            image = image.data.cpu().permute(1, 2, 0).numpy()
            image = ((image + 1) / 2. * 255).astype(np.uint8)

        h, w, c = image.shape
        if h != self.height or w != self.width:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(gray, 0)

        shape = torch.cuda.FloatTensor(self.markers, 2).fill_(np.nan)
        if rects:
            shape = self.predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            shape = shape / self.scale
            shape = torch.cuda.FloatTensor(shape)

        return shape

    def loss(self, images):

        landmarks = torch.stack([self.landmarks(img) for img in images])
        r_landmark = self.landmark_loss(landmarks)

        attributes = self.attributes(images)
        disc = self.discriminator(images)

        r_disc = self.disc_loss(disc, torch.cuda.FloatTensor(len(disc)).fill_(1.)).detach()
        r_att = self.att_loss(attributes, self.attributes_target.repeat(len(attributes), 1)).detach().mean(1)

        return -(r_landmark * 100 + r_disc + r_att)

    def landmark_loss(self, landmarks):

        l = (self.landmark_target - landmarks).pow(2).sum(dim=2).sum(dim=1)
        l[torch.isnan(l)] = self.penalty

        return l

    def get_initial_policy(self):

        return torch.cuda.FloatTensor(self.action_space).zero_()

    def reset(self):
        self.best_policy = self.get_initial_policy()
        self.best_image, self.best_reward = self.evaluate(self.best_policy)
        self.k = 0

    def gen_images(self, policy):

        with torch.no_grad():

            # image = self.model.netG(policy).detach()
            image = self.generator(policy).detach()

        return image

    def evaluate(self, policy):

        if len(policy.shape) == 1:
            policy = policy.view(1, *policy.shape)

        images = self.gen_images(policy)
        r = self.loss(images)

        if policy.shape[0] == 1:
            r = r.item()
            images = images.squeeze(0)

        return self.results(image=torch.clamp(0.5 * images + 0.5, 0, 1), reward=r)

    def step(self, policy):

        self.k += len(policy)

        images = self.gen_images(policy)
        r = self.loss(images)

        mr = float(torch.max(r))
        if mr > self.best_reward:
            self.best_reward = mr
            i = torch.argmax(r)
            self.best_policy = policy[i].clone()
            self.best_image = torch.clamp(0.5 * images[i].clone() + 0.5, 0, 1)

        return self.results(image=torch.clamp(0.5 * images + 0.5, 0, 1), reward=r)

