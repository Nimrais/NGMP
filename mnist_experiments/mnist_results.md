# MNIST Model Results on 1,000 Training Images

Comparison of projected-Nesterov RxInfer flattened softplus MLP variants, neural softplus MLPs
with one through seven hidden layers, one neural categorical CNN, and one
RxInfer CNN-like model trained on the same subset of **1,000 MNIST training
images**. All runs use a batch size of 32.

## Dataset

| Split | Number of images |
|---|---:|
| Training | 1,000 |
| Validation | 100 |
| Test | 100 |

## Accuracy Results on 1,000 Training Images

| Model | Epochs | Final train accuracy | Best validation accuracy | Best validation epoch | Test accuracy |
|---|---:|---:|---:|---:|---:|
| RxInfer MLP, 1 hidden layer (32), vector transport | 50 | 95.4% | 88.0% | 37 | 89.0% |
| RxInfer MLP, 2 hidden layers (32/32), vector transport | 50 | 92.5% | 83.0% | 22 | 83.0% |
| RxInfer MLP, 1 hidden layer (32), projected Nesterov | 50 | 93.6% | 90.0% | 12 | 89.0% |
| RxInfer MLP, 2 hidden layers (32/32), projected Nesterov | 50 | 93.3% | **92.0%** | 26 | **90.0%** |
| RxInfer MLP without vector transport | 10 | 91.3% | 86.0% | 5 | 80.0% |
| Neural MLP, 1 hidden layer (32) | 50 | 94.7% | 86.0% | 14 | 83.0% |
| Neural MLP, 2 hidden layers (32/32) | 50 | 93.4% | 83.0% | 31 | 84.0% |
| Neural MLP, 3 hidden layers (32 each) | 50 | 78.8% | 70.0% | 50 | 71.0% |
| Neural MLP, 4 hidden layers (32 each) | 50 | 54.6% | 54.0% | 48 | 52.0% |
| Neural MLP, 5 hidden layers (32 each) | 50 | 36.0% | 38.0% | 48 | 32.0% |
| Neural MLP, 6 hidden layers (32 each) | 50 | 25.7% | 25.0% | 34 | 22.0% |
| Neural MLP, 7 hidden layers (32 each) | 50 | 22.6% | 24.0% | 41 | 21.0% |
| Neural categorical CNN | 50 | 86.5% | 83.0% | 42 | 84.0% |
| RxInfer CNN-like with vector transport | 3 | 67.2% | 76.6% | 2 | 78.1% |

The test accuracy is reported for the checkpoint selected by the best validation
accuracy in each run. Because the test set contains 100 images, one correctly
classified image corresponds to one percentage point of test accuracy.

## Test-Set Comparison

The two-hidden-layer projected-Nesterov RxInfer MLP achieved the highest test
accuracy at **90.0%**. Both one-hidden-layer RxInfer variants achieved 89.0%
test accuracy: vector transport reached 88.0% validation accuracy at epoch 37,
while projected Nesterov reached 90.0% at epoch 12. The two-hidden-layer
projected-Nesterov model reached 92.0% validation accuracy at epoch 26; its
vector-transport counterpart reached 83.0% validation and test accuracy.

The one-hidden-layer neural MLP reached 86.0% validation accuracy after 14
epochs, while RxInfer without vector transport first reached 86.0% validation
accuracy after 5 epochs.
Among the neural MLP depth sweep, two hidden layers gave the highest test
accuracy at 84.0%. Accuracy then declined sharply with additional depth: the
three- through seven-hidden-layer models achieved 71.0%, 52.0%, 32.0%, 22.0%,
and 21.0% test accuracy, respectively. With the shared initialization scale and
training settings, the deeper models also had much lower final training
accuracy, indicating an optimization problem rather than improved generalization.
The neural categorical CNN reached its best validation accuracy of 83.0% at
epoch 42 and achieved 84.0% test accuracy.
The RxInfer CNN-like model with vector transport reached its best validation
accuracy of 76.6% at epoch 2 and achieved 78.1% holdout test accuracy after a
three-epoch run.

These results come from a single run on a small validation and test set. Multiple
random seeds and a larger test set would be needed to determine whether the
observed differences are statistically reliable.

## Training Accuracy

![Training accuracy by epoch](mnist_train_accuracy_comparison.png)

## Validation Accuracy

![Validation accuracy by epoch](mnist_val_accuracy_comparison.png)
