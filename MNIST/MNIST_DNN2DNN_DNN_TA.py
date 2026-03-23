# Train digital TAs from digital teachers
# Last updated: Nov 19, 2025
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

mnist_train = datasets.MNIST(root='./results/data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./results/data', train=False, transform=transform, download=True)

X_train = mnist_train.data.numpy().reshape(-1, 784)# / 255.0
X_test = mnist_test.data.numpy().reshape(-1, 784)# / 255.0

y_train = mnist_train.targets.numpy()
y_test = mnist_test.targets.numpy()

# Apply PCA
pca_dim = 16
pca = PCA(n_components=pca_dim)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Teacher logits
teacher_output = np.load("MNIST/teachers/mnist_train_logits_lenet.npy")
assert teacher_output.shape[0] == X_train.shape[0]

train_ds = TensorDataset(torch.tensor(X_train_pca, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long),
                         torch.tensor(teacher_output, dtype=torch.float32))

test_ds = TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

# complex linear layer
# defined by separating the calculations of real and imaginary parts
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_real = nn.Linear(in_features, out_features, bias=False) # Real part, no bias
        self.W_imag = nn.Linear(in_features, out_features, bias=False) # Imaginary part, no bias

    def forward(self, x_real, x_imag):
        out_real = self.W_real(x_real) - self.W_imag(x_imag)
        out_imag = self.W_real(x_imag) + self.W_imag(x_real)
        return out_real, out_imag

# Non-linearity
class ComplexReLU(nn.Module):
    # Apply ReLU separately on real and imaginary parts
    def forward(self, xr, xi):
        return F.relu(xr), F.relu(xi)

# Define a small MLP
class ComplexMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10, output_selection=range(10)):
        super().__init__()
        self.fc1 = ComplexLinear(input_dim, hidden_dim)
        self.act = ComplexReLU()
        self.fc2 = ComplexLinear(hidden_dim, hidden_dim)
        self.output_selection = output_selection

    def forward(self, x):
        xr, xi = x, torch.zeros_like(x)   # input is real PCA output

        xr, xi = self.fc1(xr, xi)
        xr, xi = self.act(xr, xi)

        xr, xi = self.fc2(xr, xi)
        xr, xi = self.act(xr, xi)

        # Convert complex to magnitude for classifier
        out = torch.sqrt(xr**2 + xi**2 + 1e-8)[:, self.output_selection]
        return out

# Define the loss
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):

    # Soft loss (KL div)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    loss_soft = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

    # Hard loss
    loss_hard = F.cross_entropy(student_logits, labels)

    return alpha * loss_soft + (1 - alpha) * loss_hard


# ============ Training ============ 
model = ComplexMLP(input_dim=16, hidden_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

EPOCHS = 20
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # train
    for xb, yb, tb in train_loader:
        xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)

        optimizer.zero_grad()
        student_logits = model(xb)
        loss = distillation_loss(student_logits, tb, yb, T=1.2, alpha=0.7)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    # test accuracy
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    test_acc = 100 * correct / total

    # if best model, save the logits
    if test_acc > best_acc:
        best_acc = test_acc

        model.eval()
        all_logits = []

        with torch.no_grad():
            for xb, yb, tb in train_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        print("Student logits saved with shape:", all_logits.shape)

        # save to file
        os.makedirs("MNIST/TA", exist_ok=True)
        np.save("MNIST/TA/digital_TA_16.npy", all_logits)
    print(f"Epoch {epoch+1}/{EPOCHS} | Test Accuracy: {test_acc:.2f} [%]")

# Final test accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

print(f"[Done] Test Accuracy: {100 * correct/total:.2f}%")
