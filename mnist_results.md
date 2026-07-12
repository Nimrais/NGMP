# MNIST Model Results on 1,000 Training Images

Comparison of three flattened softplus MLP models, one neural categorical CNN,
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
| RxInfer MLP without vector transport | 10 | 91.3% | 86.0% | 5 | 80.0% |
| Neural MLP | 50 | 94.7% | 86.0% | 14 | 83.0% |
| Neural categorical CNN | 50 | 86.5% | 83.0% | 42 | 84.0% |
| RxInfer CNN-like with vector transport | 3 | 67.2% | 76.6% | 2 | 78.1% |

The test accuracy is reported for the checkpoint selected by the best validation
accuracy in each run. Because the test set contains 100 images, one correctly
classified image corresponds to one percentage point of test accuracy.

## Test-Set Comparison

The RxInfer MLP with vector transport achieved the highest test accuracy at
**88.0%**. This is 8 percentage points higher than RxInfer without vector
transport and 5 percentage points higher than the neural MLP baseline.
It is also 4 percentage points higher than the neural categorical CNN.

Vector transport also reached 88.0% validation accuracy after 10 epochs. The
neural MLP reached 86.0% validation accuracy after 14 epochs, while RxInfer
without vector transport first reached 86.0% validation accuracy after 5 epochs.
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
