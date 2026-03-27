#set par(leading: 0.55em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show heading: set block(above: 1.4em, below: 1em)

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#codly(languages: codly-languages, display-icon: false, display-name: false, breakable: true)

#{
  show heading: none
  heading[Cover]
}

#align(center + horizon)[
  #text(size: 1.2em, weight: "bold")[
    Department of Electrical Engineering \ \ \
    Indian Institute of Technology, Kharagpur \ \ \
    Algorithms, AI and ML Laboratory (EE22202) \ \ \
    Spring, 2025-26 \ \
  ]

  #text(size: 1.4em, weight: "bold")[
    Report 09: Fashion MNIST Classification using Convolutional Neural Networks
  ]

  #text(size: 1.2em, weight: "bold")[
    Name: Arijit Dey \ \
    Roll No: 24IE10001
  ]
]

#pagebreak()

#align(center)[= Fashion MNIST Classification using Convolutional Neural Networks]

== Data Loading and Preprocessing

Here we load the FashionMNIST dataset, and prepare the data loaders for training, validation, and testing. A 20% split of the training data is used for validation. The device is set to 'cuda' if available, otherwise 'cpu'.

#codly(header: [*Data Transformation and Loading*], number-format: numbering.with("1"))
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST("F_MNIST_data", download = True, train = True, transform = transform )
test_dataset = datasets.FashionMNIST("F_MNIST_data", download = True, train = False, transform = transform )

indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
split = int(0.2 * len(train_dataset))
val_ids , train_ids = indices[:split], indices[split:]
```

#codly(header: [*DataLoader Setup*], number-format: numbering.with("1"))
```python
train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
val_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = BATCH_SIZE, sampler = val_sampler)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size = BATCH_SIZE, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#codly(header: [*Data Inspection*], number-format: numbering.with("1"))
```python
print("Length of train data is " + str(len(train_sampler)))
print("Length of test data is " + str(len(test_dataset)))
print("Length of validation data is " + str(len(val_sampler)))

image, label = next(iter(train_loader))
print(image[0].shape, label.shape)

desc = ["T-shirt/top ", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
print(desc[label[0].item()])
plt.imshow(image[0].numpy().squeeze(), cmap="gray")
```
#codly(header: [*Result*], number-format: none)
```
Length of train data is 48000
Length of test data is 10000
Length of validation data is 12000
torch.Size([1, 28, 28]) torch.Size([128])
Sandal
```
#figure(
  image("fig1.png", width: 80%),
  caption: [Sample Image from FashionMNIST Dataset]
)

== Model Definition: `BaseCNN`

The `BaseCNN` class models our entire classification model pipeline. We inherit it from PyTorch's `torch.nn.Module` class to get access to PyTorch's pre-built layers like `Conv2D`, `BatchNorm2D`, `ReLU` and `Sequential`. The class allows for customisation of kernel size, stride, number of convolutional blocks, number of filters, fully connected layer hidden size, dropout probability, and batch normalization.

#codly(header: [*BaseCNN Class Definition*], number-format: numbering.with("1"))
```python
class BaseCNN(nn.Module):
    def __init__(self,
                 k=2,
                 s=1,
                 num_blocks=3,
                 filters=64,
                 fc_hidden=64,
                 dropout_p=0.0,
                 batch_norm=False):
        super(BaseCNN, self).__init__()
        self.num_blocks = num_blocks
        self.conv_layers = nn.ModuleList()

        in_channels = 1
        current_size = 28

        for i in range(num_blocks):
            # Use padding to try and keep spatial dimensions if s=1
            padding = (k - 1) // 2 if s == 1 else (k // 2)
            conv = nn.Conv2d(in_channels, filters, kernel_size=k, stride=s, padding=padding)
            self.conv_layers.append(conv)

            if batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(filters))

            self.conv_layers.append(nn.ReLU())

            current_size = (current_size + 2 * padding - k) // s + 1

            if current_size > 1:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size = current_size // 2

            in_channels = filters

        self.flat_size = filters * current_size * current_size

        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(fc_hidden, 10)
        ) 
  ```

=== The `forward` function
The `forward` method passes the input `x` through the defined convolutional and fully connected layers. It can also optionally return intermediate activations from the convolutional layers, which is useful for visualisation and debugging.

```python
    def forward(self, x, return_activations=False):
        activations = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if return_activations:
                    activations.append(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if return_activations:
            return x, activations
        return x
```

== Training and Reporting Function

This section defines the `train_and_report` function, which handles the training loop, validation, and testing of a given CNN model. It calculates and prints training, validation, and test accuracies, and visualises loss history and convolutional layer activations at certain epochs.

#codly(header: [*train_and_report Function Definition*], number-format: numbering.with("1"))
```python
def train_and_report(model, num_epochs=50, lr=0.001):
    # ... SETUP CODE OMITTED

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.sampler)
        train_losses.append(train_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
        val_loss = v_loss / len(val_loader.sampler)
        val_losses.append(val_loss)

      # PROGRESS REPORTING AND PLOTTING CODE OMITTED...
```

== Experiment 1: Baseline

This experiment establishes a baseline performance for the CNN model using default parameters: kernel size of 2, stride of 1, 3 convolutional blocks, 64 filters, 64 fully connected hidden units, no dropout, and no batch normalisation.

#codly(header: [*Baseline Model Training*], number-format: numbering.with("1"))
```python
model_baseline = BaseCNN(k=2, s=1, num_blocks=3, filters=64, fc_hidden=64, dropout_p=0.0, batch_norm=False)
baseline_acc = train_and_report(model_baseline, num_epochs=50)
```
#codly(header: [*Result*], number-format: none)
```
Experiment 1: Baseline
Total Trainable Parameters: 50314
Epoch 1/50, Train Loss: 0.7333, Val Loss: 0.5294
Epoch 10/50, Train Loss: 0.2646, Val Loss: 0.3054
Epoch 20/50, Train Loss: 0.1827, Val Loss: 0.3110
Epoch 30/50, Train Loss: 0.1256, Val Loss: 0.3463
Epoch 40/50, Train Loss: 0.0884, Val Loss: 0.4516
Epoch 50/50, Train Loss: 0.0622, Val Loss: 0.5607
Final Accuracies: Train: 97.44%, Val: 88.62%, Test: 88.24%
```
#figure(
  image("fig2.png", width: 80%),
  caption: [Epoch 1 Activations]
)
#figure(
  image("fig3.png", width: 80%),
  caption: [Epoch 10 Activations]
)
#figure(
  image("fig4.png", width: 80%),
  caption: [Epoch 20 Activations]
)
#figure(
  image("fig5.png", width: 80%),
  caption: [Epoch 30 Activations]
)
#figure(
  image("fig6.png", width: 80%),
  caption: [Epoch 40 Activations]
)
#figure(
  image("fig7.png", width: 80%),
  caption: [Epoch 50 Activations]
)
#figure(
  image("fig8.png", width: 80%),
  caption: [Loss History]
)

== Observations
The baseline model achieves a test accuracy of 88.24%. The training loss consistently decreases, but the validation loss starts to increase after a certain point, indicating potential overfitting. The activation maps show features being learned by the convolutional layers over epochs.

== Experiment 2: Kernel Size and Stride Ablation

This experiment investigates the effect of using a larger kernel size and stride on the CNN model's performance. The model is configured with a kernel size of 5 and a stride of 2, while keeping other parameters similar to the baseline.

#codly(header: [*Kernel Size and Stride Ablation Model Training*], number-format: numbering.with("1"))
```python
model_ks = BaseCNN(k=5, s=2, num_blocks=3, filters=64, fc_hidden=64, dropout_p=0.0, batch_norm=False)
ks_acc = train_and_report(model_ks, num_epochs=50)
```
#codly(header: [*Result*], number-format: none)
```
Total Trainable Parameters: 211402
Epoch 1/50, Train Loss: 0.6746, Val Loss: 0.4608
Epoch 10/50, Train Loss: 0.1843, Val Loss: 0.2525
Epoch 20/50, Train Loss: 0.0857, Val Loss: 0.3128
Epoch 30/50, Train Loss: 0.0398, Val Loss: 0.4987
Epoch 40/50, Train Loss: 0.0313, Val Loss: 0.5306
Epoch 50/50, Train Loss: 0.0272, Val Loss: 0.6256
Final Accuracies: Train: 99.45%, Val: 90.63%, Test: 90.19%
```
#figure(
  image("fig9.png", width: 80%),
  caption: [Epoch 1 Activations (KS/Stride Model)]
)
#figure(
  image("fig10.png", width: 80%),
  caption: [Epoch 10 Activations (KS/Stride Model)]
)
#figure(
  image("fig11.png", width: 80%),
  caption: [Epoch 20 Activations (KS/Stride Model)]
)
#figure(
  image("fig12.png", width: 80%),
  caption: [Epoch 30 Activations (KS/Stride Model)]
)
#figure(
  image("fig13.png", width: 80%),
  caption: [Epoch 40 Activations (KS/Stride Model)]
)
#figure(
  image("fig14.png", width: 80%),
  caption: [Epoch 50 Activations (KS/Stride Model)]
)
#figure(
  image("fig15.png", width: 80%),
  caption: [Loss History (KS/Stride Model)]
)

== Observations
This model achieved a test accuracy of 90.19%, which is an improvement over the baseline. The training loss decreases significantly, but the validation loss shows a clear upward trend after around epoch 10, indicating severe overfitting. The higher number of trainable parameters (211402) compared to the baseline (50314) suggests that increasing the kernel size and stride without regularisation can lead to a more complex model that overfits the training data.

== Experiment 3: Depth Ablation (Shallower)

This experiment examines the impact of reducing the network's depth on its performance. The model is configured with only 2 convolutional blocks, making it shallower than the baseline (3 blocks).

#codly(header: [*Shallower Model Training*], number-format: numbering.with("1"))
```python
model_shallow = BaseCNN(k=2, s=1, num_blocks=2, filters=64, fc_hidden=64, dropout_p=0.0, batch_norm=False)
shallow_acc = train_and_report(model_shallow, num_epochs=50)
```
#codly(header: [*Result*], number-format: none)
```
Experiment 3: Depth Ablation (shallower)
Total Trainable Parameters: 164938
Epoch 1/50, Train Loss: 0.5815, Val Loss: 0.4054
Epoch 10/50, Train Loss: 0.1881, Val Loss: 0.2524
Epoch 20/50, Train Loss: 0.0924, Val Loss: 0.3127
Epoch 30/50, Train Loss: 0.0410, Val Loss: 0.4491
Epoch 40/50, Train Loss: 0.0256, Val Loss: 0.6016
Epoch 50/50, Train Loss: 0.0202, Val Loss: 0.6897
Final Accuracies: Train: 99.31%, Val: 90.11%, Test: 90.24%
```
#figure(
  image("fig16.png", width: 80%),
  caption: [Epoch 1 Activations (Shallower Model)]
)
#figure(
  image("fig17.png", width: 80%),
  caption: [Epoch 10 Activations (Shallower Model)]
)
#figure(
  image("fig18.png", width: 80%),
  caption: [Epoch 20 Activations (Shallower Model)]
)
#figure(
  image("fig19.png", width: 80%),
  caption: [Epoch 30 Activations (Shallower Model)]
)
#figure(
  image("fig20.png", width: 80%),
  caption: [Epoch 40 Activations (Shallower Model)]
)
#figure(
  image("fig21.png", width: 80%),
  caption: [Epoch 50 Activations (Shallower Model)]
)
#figure(
  image("fig22.png", width: 80%),
  caption: [Loss History (Shallower Model)]
)

== Observations
The shallower model achieves a test accuracy of 90.24%, which is slightly better than the baseline and similar to the KS/Stride model. However, it also shows signs of overfitting, with validation loss increasing significantly in later epochs. This suggests that even with fewer layers, the model can still be complex enough to overfit if not properly regularised.

== Experiment 4: Dropout (p = 0.25)

This experiment evaluates the effect of adding dropout regularization to the CNN model to mitigate overfitting. The model is configured with a dropout probability of 0.25, while maintaining the baseline's architecture.

#codly(header: [*Dropout Model Training*], number-format: numbering.with("1"))
```python
model_dropout = BaseCNN(k=2, s=1, num_blocks=3, filters=64, fc_hidden=64, dropout_p=0.25, batch_norm=False)
dropout_acc = train_and_report(model_dropout, num_epochs=50)
```
#codly(header: [*Result*], number-format: none)
```
Experiment 4: Dropout (p = 0.25)
Total Trainable Parameters: 50314
Epoch 1/50, Train Loss: 0.8317, Val Loss: 0.5271
Epoch 10/50, Train Loss: 0.3154, Val Loss: 0.3161
Epoch 20/50, Train Loss: 0.2469, Val Loss: 0.2894
Epoch 30/50, Train Loss: 0.2022, Val Loss: 0.3064
Epoch 40/50, Train Loss: 0.1711, Val Loss: 0.3313
Epoch 50/50, Train Loss: 0.1520, Val Loss: 0.3624
Final Accuracies: Train: 95.32%, Val: 89.01%, Test: 89.19%
```
#figure(
  image("fig23.png", width: 80%),
  caption: [Epoch 1 Activations (Dropout Model)]
)
#figure(
  image("fig24.png", width: 80%),
  caption: [Epoch 10 Activations (Dropout Model)]
)
#figure(
  image("fig25.png", width: 80%),
  caption: [Epoch 20 Activations (Dropout Model)]
)
#figure(
  image("fig26.png", width: 80%),
  caption: [Epoch 30 Activations (Dropout Model)]
)
#figure(
  image("fig27.png", width: 80%),
  caption: [Epoch 40 Activations (Dropout Model)]
)
#figure(
  image("fig28.png", width: 80%),
  caption: [Epoch 50 Activations (Dropout Model)]
)
#figure(
  image("fig29.png", width: 80%),
  caption: [Loss History (Dropout Model)]
)

== Observations
The dropout model achieves a test accuracy of 89.19%, showing a slight improvement over the baseline. Compared to the models without dropout (baseline, KS/Stride, shallower), this model exhibits a smaller gap between training and validation loss, indicating that dropout helps in regularizing the model and reducing overfitting. However, the overall accuracy is not as high as the KS/Stride or shallower models.

== Experiment 5: Batch Normalisation

This experiment applies batch normalisation to the best-performing model identified from the previous experiments. The purpose is to assess if batch normalisation can further improve performance and stability.

#codly(header: [*Batch Normalisation Model Training*], number-format: numbering.with("1"))
```python
results = {
    "Baseline": baseline_acc,
    "KS/Stride": ks_acc,
    "Shallow": shallow_acc,
    "Dropout": dropout_acc
}
best_name = max(results, key=results.get)
print(f"Best performing model so far: {best_name} ({results[best_name]:.2f}%)")

if best_name == "Baseline":
    model_best_bn = BaseCNN(k=2, s=1, num_blocks=3, filters=64, fc_hidden=64, dropout_p=0.0, batch_norm=True)
elif best_name == "KS/Stride":
    model_best_bn = BaseCNN(k=5, s=2, num_blocks=3, filters=64, fc_hidden=64, dropout_p=0.0, batch_norm=True)
elif best_name == "Shallow":
    model_best_bn = BaseCNN(k=2, s=1, num_blocks=2, filters=64, fc_hidden=64, dropout_p=0.0, batch_norm=True)
else:
    model_best_bn = BaseCNN(k=2, s=1, num_blocks=3, filters=64, fc_hidden=64, dropout_p=0.25, batch_norm=True)

print("Retraining with Batch Normalization...")
best_bn_acc = train_and_report(model_best_bn, num_epochs=50)
```
#codly(header: [*Result*], number-format: none)
```
Best performing model so far: Shallow (90.24%)
Retraining with Batch Normalization...
Total Trainable Parameters: 165194
Epoch 1/50, Train Loss: 0.4191, Val Loss: 0.3704
Epoch 10/50, Train Loss: 0.1354, Val Loss: 0.2513
Epoch 20/50, Train Loss: 0.0496, Val Loss: 0.3380
Epoch 30/50, Train Loss: 0.0197, Val Loss: 0.4337
Epoch 40/50, Train Loss: 0.0210, Val Loss: 0.5534
Epoch 50/50, Train Loss: 0.0204, Val Loss: 0.5564
Final Accuracies: Train: 99.87%, Val: 91.62%, Test: 91.13%
```
#figure(
  image("fig30.png", width: 80%),
  caption: [Epoch 1 Activations (Batch Normalisation Model)]
)
#figure(
  image("fig31.png", width: 80%),
  caption: [Epoch 10 Activations (Batch Normalisation Model)]
)
#figure(
  image("fig32.png", width: 80%),
  caption: [Epoch 20 Activations (Batch Normalisation Model)]
)
#figure(
  image("fig33.png", width: 80%),
  caption: [Epoch 30 Activations (Batch Normalisation Model)]
)
#figure(
  image("fig34.png", width: 80%),
  caption: [Epoch 40 Activations (Batch Normalisation Model)]
)
#figure(
  image("fig35.png", width: 80%),
  caption: [Epoch 50 Activations (Batch Normalisation Model)]
)
#figure(
  image("fig36.png", width: 80%),
  caption: [Loss History (Batch Normalisation Model)]
)

== Observations
The model with batch normalisation, based on the "Shallow" architecture, achieves the highest test accuracy of 91.13%. This indicates that batch normalisation significantly improves the model's performance and potentially its stability. While training loss is very low, there is still some gap with validation loss, but it's better managed compared to models without batch normalisation.

== Conclusion
+   The baseline model achieved a test accuracy of 88.24%, demonstrating a reasonable starting point for classification.
+   Modifying kernel size and stride (Experiment 2) improved accuracy to 90.19%, but also led to increased overfitting as evidenced by the divergence of training and validation loss.
+   Reducing the network's depth (Experiment 3) yielded a comparable test accuracy of 90.24% but still exhibited overfitting.
+   Implementing dropout regularisation (Experiment 4) helped mitigate overfitting, as indicated by a smaller gap between training and validation loss, and achieved a test accuracy of 89.19%.
+   The most significant improvement was observed with batch normalisation (Experiment 5), which, when applied to the best-performing "Shallow" architecture, resulted in the highest test accuracy of 91.13%. This indicates that batch normalisation not only enhances performance but also contributes to better model stability.

In conclusion, for this task, a shallower network combined with batch normalisation proved to be the most effective configuration, achieving superior generalisation performance on the Fashion MNIST dataset. This highlights the importance of regularisation techniques like batch normalisation in building robust CNN models.
