import os
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import torchvision

from torch.optim import lr_scheduler
from data import *
from constants import *
from training import *

torch.manual_seed(42)
np.random.seed(42)
os.environ["OMP_NUM_THREADS"] = "1"



def visualize_model(model, num_images=6, fig_name="Predictions"):
    images_so_far = 0
    _fig = plt.figure(fig_name)
    model.eval()
    with torch.no_grad():
        for _i, (inputs, labels) in enumerate(dataloaders["validation"]):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title("[{}]".format(class_names[preds[j]]))
                print(class_names[preds[j]] == class_names[labels.cpu().data[j]])
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    return



dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(4), normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(4))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]




def main():
    # Initialize dataloader
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
        for x in ["train", "validation"]
    }
    inputs, classes = next(iter(dataloaders["train"]))
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
        for x in ["train", "validation"]
    }

    model_hybrid = torchvision.models.resnet18(pretrained=True)

    n_layers = 6
    weight_shapes = {"weights": (n_layers, N_QUBITS)}

    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # Notice that model_hybrid.fc is the last layer of ResNet18
    model_hybrid.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        qlayer,
        nn.ReLU(),
        nn.Linear(4,2)
    )

    # Use CUDA or CPU according to the "device" object.
    model_hybrid = model_hybrid.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer_hybrid = optim.Adam(model_hybrid.fc.parameters(), lr=ALPHA)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_hybrid, step_size=10, gamma=GAMMA_LR_SCHEDULER
    )

    tloss = []
    acc = []
    model_hybrid = train_model(
        model_hybrid, criterion, optimizer_hybrid, exp_lr_scheduler, num_epochs=NUM_EPOCHS,
        tloss=tloss, acc=acc
    )

    plt.plot(tloss, label="Training loss")
    plt.title("Loss vs. Iterations")
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")

    plt.plot(acc, label="Validation accuracy")
    plt.title("Accuracy vs. Iterations")
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")

    visualize_model(model_hybrid, num_images=BATCH_SIZE)
    plt.show()


if __name__ == "__main__":
    main()