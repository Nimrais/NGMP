# MNIST RxInfer Results on 10,000 Training Images

Results for direct-backend RxInfer Softplus MLPs with projected Nesterov updates,
trained
for 50 epochs on MNIST digits 0--9. Both models use 196 input features, batch
size 32, and 1,000 examples each for validation and test.

## Dataset

| Split | Number of images |
|---|---:|
| Training | 10,000 |
| Validation | 1,000 |
| Test | 1,000 |

## Model Configuration

| Model | Hidden layers | Epochs | Additional setting |
|---|---|---:|---|
| RxInfer flattened Softplus MLP | 32 | 50 | vector transport or projected Nesterov; `alpha=0.2`, direct weight-site scale 0.5 |
| RxInfer two-hidden-layer Softplus MLP | 32 / 32 | 50 | vector transport or projected Nesterov; 3 inner site-update iterations; direct weight-site scale 1.0 |
| Neural Softplus MLP | 1--7 layers of 32 units | 50 | `lr=0.001`, no weight decay |

## Accuracy Results

| Model | Final train accuracy | Best validation accuracy | Best validation epoch | Test accuracy |
|---|---:|---:|---:|---:|
| RxInfer MLP, 1 hidden layer (32), vector transport | 95.3% | 92.3% | 50 | 91.3% |
| RxInfer MLP, 2 hidden layers (32/32), vector transport | 93.3% | 90.7% | 47 | 89.7% |
| RxInfer MLP, 1 hidden layer (32), projected Nesterov | 94.3% | 93.7% | 47 | 92.2% |
| RxInfer MLP, 2 hidden layers (32/32), projected Nesterov | 93.8% | 92.6% | 45 | 91.7% |
| Neural MLP, 1 hidden layer (32) | 97.3% | 93.3% | 46 | 93.0% |
| Neural MLP, 2 hidden layers (32/32) | 97.2% | **94.2%** | 50 | **93.3%** |
| Neural MLP, 3 hidden layers (32 each) | **97.4%** | 94.0% | 46 | 92.7% |
| Neural MLP, 4 hidden layers (32 each) | 97.2% | 93.5% | 48 | 92.1% |
| Neural MLP, 5 hidden layers (32 each) | 96.7% | 91.8% | 46 | 91.2% |
| Neural MLP, 6 hidden layers (32 each) | 93.5% | 89.8% | 47 | 87.7% |
| Neural MLP, 7 hidden layers (32 each) | 85.0% | 81.2% | 50 | 80.5% |

The test accuracy is measured using the checkpoint selected by best validation
accuracy. Among the RxInfer variants, the one-hidden-layer projected-Nesterov
model achieved the stronger result: 93.7% validation accuracy and 92.2% test
accuracy. The two-hidden-layer projected-Nesterov model reached 92.6%
validation accuracy at epoch 45 and 91.7% test accuracy. The one- and
two-hidden-layer vector-transport models achieved 91.3% and 89.7% test accuracy,
respectively.

Among the neural MLP depth sweep, two hidden layers achieved the highest test
accuracy at **93.3%**, with the one-hidden-layer model close behind at 93.0%.
The three- and four-hidden-layer neural models also exceeded 92% test accuracy,
while performance declined for five to seven hidden layers; the seven-hidden
layer model achieved 80.5% test accuracy.

## Training Accuracy

![RxInfer training accuracy on 10k MNIST](mnist_10k_rxinfer_train_accuracy.png)

## Validation Accuracy

![RxInfer validation accuracy on 10k MNIST](mnist_10k_rxinfer_val_accuracy.png)
