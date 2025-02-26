import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Classifier")
        self.root.geometry("1000x600")
        
        # Set up device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = CNN().to(self.device)
        
        # Initialize UI
        self.setup_ui()
        
        # Load data
        self.load_data()
        
    def setup_ui(self):
        # Create frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.display_frame = ttk.Frame(self.root, padding="10")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control panel
        ttk.Label(self.control_frame, text="MNIST Digit Classifier", font=("Arial", 16)).pack(pady=10)
        
        ttk.Button(self.control_frame, text="Train Model", command=self.train_model).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Test Model", command=self.test_model).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Show Sample", command=self.show_sample).pack(fill=tk.X, pady=5)
        
        # Progress bar and status
        self.progress = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.control_frame, textvariable=self.status_var).pack(pady=5)
        
        # Results display
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Metrics display
        self.metrics_frame = ttk.LabelFrame(self.display_frame, text="Metrics", padding="10")
        self.metrics_frame.pack(fill=tk.X, pady=10)
        
        self.accuracy_var = tk.StringVar(value="Accuracy: N/A")
        ttk.Label(self.metrics_frame, textvariable=self.accuracy_var).pack(side=tk.LEFT, padx=20)
        
        self.loss_var = tk.StringVar(value="Loss: N/A")
        ttk.Label(self.metrics_frame, textvariable=self.loss_var).pack(side=tk.LEFT, padx=20)
        
    def load_data(self):
        # Data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load training data
        self.status_var.set("Loading MNIST dataset...")
        self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000)
        
        self.status_var.set("Dataset loaded")
        
    def train_model(self):
        self.status_var.set("Training model...")
        self.progress['value'] = 0
        self.root.update()
        
        # Training parameters
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 3
        log_interval = 100
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % log_interval == 0:
                    train_losses.append(loss.item())
                    self.status_var.set(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                          f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                    
                    # Update progress bar
                    progress_value = 100 * (epoch * len(self.train_loader) + batch_idx) / (num_epochs * len(self.train_loader))
                    self.progress['value'] = progress_value
                    self.root.update()
        
        # Plot training loss
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(train_losses)
        ax.set_title('Training Loss')
        ax.set_xlabel('Iterations (x100)')
        ax.set_ylabel('Loss')
        self.canvas.draw()
        
        self.status_var.set("Training completed")
        self.test_model()
        
    def test_model(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        self.status_var.set("Testing model...")
        self.progress['value'] = 0
        self.root.update()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Update progress bar
                progress_value = 100 * batch_idx / len(self.test_loader)
                self.progress['value'] = progress_value
                self.root.update()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        # Update metrics
        self.accuracy_var.set(f"Accuracy: {accuracy:.2f}%")
        self.loss_var.set(f"Loss: {test_loss:.4f}")
        
        self.status_var.set(f"Testing completed. Accuracy: {accuracy:.2f}%")
        
        # Plot confusion matrix if available
        self.show_sample()
    
    def show_sample(self):
        # Display a few sample predictions
        self.model.eval()
        
        # Get some random test examples
        data, target = next(iter(self.test_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        # Get predictions
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=True)
        
        # Plot 5x2 grid of samples
        self.fig.clear()
        for i in range(10):
            ax = self.fig.add_subplot(2, 5, i+1)
            ax.imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
            ax.set_title(f'Pred: {pred[i].item()}, True: {target[i].item()}')
            ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set("Displaying sample predictions")

# Run the application
def main():
    root = tk.Tk()
    app = MNISTApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()