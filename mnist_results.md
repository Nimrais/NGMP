# MNIST Model Results on 1,000 Training Images

Comparison of five flattened softplus MLP runs, one neural categorical CNN,
and one RxInfer CNN-like model trained on the same subset of **1,000 MNIST
training images**. All runs use a batch size of 32.

## Dataset

| Split | Number of images |
|---|---:|
| Training | 1,000 |
| Validation | 100 |
| Test | 100 |

## Accuracy Results on 1,000 Training Images

| Model | Epochs | Final train accuracy | Best validation accuracy | Best validation epoch | Test accuracy |
|---|---:|---:|---:|---:|---:|
| RxInfer MLP with vector transport | 10 | 94.7% | **88.0%** | 10 | **88.0%** |
| RxInfer two-hidden-layer MLP (32/32) with vector transport | 10 | 92.0% | 82.0% | 5 | 86.0% |
| RxInfer MLP without vector transport | 10 | 91.3% | 86.0% | 5 | 80.0% |
| RxInfer MLP with projected Nesterov | 10 | 93.6% | 86.0% | 6 | 84.0% |
| Neural MLP | 50 | 94.7% | 86.0% | 14 | 83.0% |
| Neural categorical CNN | 50 | 86.5% | 83.0% | 42 | 84.0% |
| RxInfer CNN-like with vector transport | 3 | 67.2% | 76.6% | 2 | 78.1% |

The test accuracy is reported for the checkpoint selected by the best validation
accuracy in each run. Because the test set contains 100 images, one correctly
classified image corresponds to one percentage point of test accuracy.

## Test-Set Comparison

The one-hidden-layer RxInfer MLP with vector transport achieved the highest test
accuracy at **88.0%**. The two-hidden-layer RxInfer MLP was second at 86.0%, two
percentage points behind the best model and three points above the neural MLP
baseline. It also finished two points above both the projected-Nesterov MLP and
the neural categorical CNN, although its best validation accuracy was lower at
82.0%.

Vector transport also reached 88.0% validation accuracy after 10 epochs. The
two-hidden-layer RxInfer MLP, using 32 units in each hidden layer and three inner
site-update iterations, reached its best validation accuracy of 82.0% at epoch
5, finished with 92.0% training accuracy, and achieved 86.0% test accuracy. The
neural MLP reached 86.0% validation accuracy after 14 epochs,
while RxInfer without vector transport first reached 86.0% validation accuracy
after 5 epochs.
The projected-Nesterov RxInfer MLP reached 86.0% validation accuracy at epoch 6
and achieved 84.0% test accuracy.
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
