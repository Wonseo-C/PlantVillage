from model import ShallowCNN

import numpy as np
import argparse

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import torchvision


# shallowCNN and PCA
def get_output(Image):
    cnn_output = shallowCNN(Image)
    cnn_output = cnn_output.detach().cpu()
    cnn_output = cnn_output.numpy()
    # PCA: 99% reserved variance
    pca = PCA(n_components=0.99)
    output_pca = pca.fit_transform(cnn_output.reshape(1, -1))
    return np.reshape(output_pca, [1])


# hyper parameter
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./Plant_leave_diseases_dataset_with_augmentation", help="data dir's path")
parser.add_argument("--classifier", type=str, default="svm", help="svm or randomforest")
parser.add_argument("--batch_szie", type=int, default=1, help="batch size")
args, _ = parser.parse_known_args()
print(args)

# vgg16
vgg16 = torchvision.models.vgg16(pretrained=True)
# shallow CNN
shallowCNN = ShallowCNN()

# Load pretrained VGG16's weight
vgg_dict = vgg16.state_dict()
shallow_dict = shallowCNN.state_dict()
pretrained_dict = {k: v for k, v in vgg_dict.items() if k in shallow_dict}
shallow_dict.update(pretrained_dict) 

# load PlantVillage dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([204, 204]),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.ImageFolder(root=args.data_path, transform=transform)

dataset_size = dataset.__len__()
train_size = int(dataset_size * 0.8)
validation_size = dataset_size - train_size

train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size, 
                                               shuffle = True)

validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size = 1, 
                                               shuffle = True)

# get feature
x_train = np.array([])
y_train = np.array([])
for index, (data, label) in train_dataloader:
    x = get_output(data)
    x_train = np.concatenate([x_train, x], axis=0)
    y_train = np.concatenate([y_train, label], axis=0)

x_test = np.array([])
y_test = np.array([])
for index, (data, label) in validation_dataloader:
    x = get_output(data)
    x_test = np.concatenate([x_test, x], axis=0)
    y_test = np.concatenate([y_test, label], axis=0)

# Random Foreset
if args.classifier == "svm":
    randomForest = RandomForestClassifier(random_state=123)
    randomForest.fit(x_train, y_train)
    y_pred_rf = randomForest.predict(x_test)

    print("-----Random Forest result-----")
    print("accuracy is ", accuracy_score(y_test, y_pred_rf))
    print("recall is ", recall_score(y_test, y_pred_rf))
    print("precision is ", precision_score(y_test, y_pred_rf))
    print("f1 is ", f1_score(y_test, y_pred_rf))

# SVM: Gaussian kernel
elif args.classifier == "randomforest":
    svm = svm.SVC(kernel="rbf")
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)

    print("-----SVM result-----")
    print("accuracy is ", accuracy_score(y_test, y_pred_svm))
    print("recall is ", recall_score(y_test, y_pred_svm))
    print("precision is ", precision_score(y_test, y_pred_svm))
    print("f1 is ", f1_score(y_test, y_pred_svm))
