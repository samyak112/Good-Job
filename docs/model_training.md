# What the classifier does

The classifier predicts the type of a social media post using four classes.

| Label | Meaning |
|---|---|
| `ML Job` | Machine learning and AI-related jobs, including engineering and research roles |
| `Job` | Software-related paid work, including engineering jobs, freelance work, and software services |
| `Other` | Paid work that isn't directly software-related, such as data annotation, RLHF, and evaluator work |
| `Not Job` | Everything else |

The model also derives a binary `is_job` prediction from the four-class output. Any post classified as something other than `Not Job` is considered a potential opportunity.

The classifier focuses on understanding what a post describes rather than identifying the account that published it.

## 1. Why I chose a single classification task

Initially, I experimented with multi-task learning to predict both the type of post and whether the publisher behaved like a job board.

However, these are different prediction problems. Post classification is about understanding the content of an individual post, while job-board detection is about identifying publishing patterns associated with an account.

It turns out that solely training on post text doesn't reliably predicts if a post is from a job board or not. So i have parked that classification for now

This gives the model one clear objective: learn to distinguish between different types of opportunities and unrelated content.

## 2. The BERT classifier

I use `bert-base-uncased` as the starting point for the classifier.

The architecture looks like this:

```text
                       Input post
                           |
                           v
                        Tokenizer
                           |
                           v
                     BERT encoder
                           |
                           v
                    [CLS] representation
                           |
                           v
                   Classification head
                           |
                           v
                       4 classes
                           |
             +------+------+------+
             |      |      |      |
             v      v      v      v
           ML Job   Job   Other  Not Job
```

The classification head is a linear layer:

- Type head: 768 → 4

### Starting from pretrained BERT

I load the pretrained weights from `bert-base-uncased` into the encoder.

The classification head starts with randomly initialized weights because it is specific to my dataset and task.

During training, I fine-tune the entire encoder along with the classification head. I don't freeze BERT.

I also use separate learning rates for the encoder and classification head.

The encoder gets a lower learning rate because it already contains useful pretrained representations. The newly initialized classification head can use a higher learning rate because it needs to learn the mapping between BERT representations and my labels.

This allows the model to adapt BERT's existing language understanding to the patterns found in my dataset.

## 3. Training objective

The classifier uses cross-entropy loss.

The loss measures how far the model's predicted class probabilities are from the actual post-type labels.

For a four-class classification problem, the objective is:

```python
loss = F.cross_entropy(
    logits,
    labels,
    weight=class_weights,  # Optional
)
```

Mathematically:

$L_{\text{type}} = -\sum_{c=1}^{4} y_c \log(p_c)$

Here, \(y_c\) represents the ground-truth class and \(p_c\) represents the predicted probability for that class.

When class weighting is disabled, all classes contribute equally per example. When enabled, mistakes on underrepresented classes can contribute more to the training objective.

## 4. Handling class imbalance during training

Even after preparing the dataset, the training set can have an uneven number of examples for each class.

If I train normally, common classes will appear more frequently in batches. The model can become biased towards those classes and perform poorly on classes with fewer examples.

I support three configurations for handling class imbalance.

### No imbalance handling

In this configuration, every training example has an equal sampling probability and the loss isn't adjusted for class frequency.

### Class-weighted loss

Instead of treating every classification mistake equally, I calculate weights using inverse class frequency:

```python
weight = class_count ** (-class_power)
```

A class with fewer examples receives a higher loss weight.

This means mistakes on underrepresented classes contribute more to the training loss.

I can control how aggressively this happens using `CLASS_POWER`.

For example:

- `1.0` applies full inverse-frequency weighting.
- `0.5` applies a softer correction.
- `0.0` disables frequency-based weighting.

### Weighted sampling

I also support PyTorch's `WeightedRandomSampler`.

Instead of changing the loss, I change the probability of selecting each training example.

```python
WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)
```

Examples with higher sample weights are more likely to appear in a training batch.

I use replacement, which means an example can appear multiple times in an epoch while some other examples might not appear at all.

The weights come from the `sample_weight` field generated by the training data preparation pipeline.

## 5. Inference

The trained model exposes a `predict()` method that accepts either a single post or a list of posts.

For every post, the model produces four logits corresponding to the four post categories.

I apply softmax to convert these into normalized class probabilities.

The predicted post type is the class with the highest probability.

I also calculate the probability that a post is an opportunity:

```python
p_job = 1 - p_not_job
```

Since the three opportunity categories are `ML Job`, `Job`, and `Other`, this gives me their combined probability.

The output includes:

- Predicted type.
- Probabilities for all four types.
- Whether the post is an opportunity.
- Combined opportunity probability.

This allows downstream systems to use both the predicted category and the model's probability estimates.

The probabilities are softmax outputs and should not automatically be interpreted as calibrated confidence scores.

## 6. Evaluation

The model is evaluated using author-grouped validation and test splits.

Grouping by author prevents the same account from appearing in training, validation, and test sets. This helps reduce the risk of misleading evaluation results caused by the model seeing posts from familiar accounts.

### Latest evaluation results

| Metric | Validation | Test |
|---|---:|---:|
| Samples | 824 | 823 |
| Accuracy | 89.44% | 89.91% |
| Incorrect predictions | 87 | 83 |

Test set class recall:

| Class | Recall |
|---|---:|
| ML Job | 89.36% |
| Job | 87.57% |
| Other | 84.71% |
| Not Job | 98.12% |

The model performs particularly well at identifying unrelated posts, while distinguishing between different types of paid opportunities remains more challenging.

The most common test-set errors involve `Job` being classified as `Other`, followed by `Job` being classified as `ML Job`.

### CPU inference performance

Measured on CPU with 6 threads, batch size 32, without quantization.

| Split | Latency per post | Throughput |
|---|---:|---:|
| Validation | 27.86 ms | 35.9 posts/s |
| Test | 25.62 ms | 39.0 posts/s |

These measurements cover model forward-pass time only and exclude tokenization and preprocessing.

## 7. The complete pipeline

The project follows this sequence:

```text
Raw social media posts
        |
        v
Data preparation
        |
        v
Deduplication
        |
        v
Author-grouped train/val/test split
        |
        v
Training sample weights
        |
        v
BERT classifier
        |
        v
BERT encoder
        |
        v
Four-class classification
        |
        v
Opportunity category
```

Each stage solves a different problem.

The data preparation pipeline reduces repeated content and prepares useful training examples.

The author-grouped split prevents the same author from leaking into training and evaluation.

Weighted sampling or class-weighted loss addresses class imbalance without permanently changing the dataset.

Finally, the BERT encoder learns to classify posts into the four opportunity categories.
