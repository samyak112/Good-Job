## What the classifier does

The classifier predicts two things for every post.

**Post type**

| Label     | Meaning                                                                                           |
| --------- | ------------------------------------------------------------------------------------------------- |
| `ML Job`  | Machine learning and AI-related jobs, including engineering and research roles                    |
| `Job`     | Software-related paid work, including engineering jobs, freelance work, and software services     |
| `Other`   | Paid work that isn't directly software-related, such as data annotation, RLHF, and evaluator work |
| `Not Job` | Everything else                                                                                   |

**Job board detection**

The second task is binary classification. For posts that are jobs, does the publisher behave like a job board (an account that regularly publishes job opportunities) or like a regular account?

The label comes from the `poster` field in my data.

I kept this as a separate task because identifying what a post is about and identifying who published it are two different problems.

## 1. Why I chose multi-task learning

Initially, I could have trained two separate classifiers: one for identifying the type of post and another for identifying job boards.

But both tasks work with the same input and have some overlapping requirements.

To classify a post correctly, the model needs to understand what kind of opportunity it describes. To identify a job board, it needs to understand patterns associated with accounts that regularly publish job opportunities.

Both tasks can benefit from a shared language representation.

Instead of maintaining two independent transformer models, I use one shared BERT encoder with two separate classification heads.

The encoder learns a shared representation of the input, while each classification head learns its own task-specific patterns.

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
                    Shared BERT encoder
                           |
                           v
                     [CLS] representation
                           |
                    +------+------+
                    |             |
                    v             v
                Type head     Board head
                    |             |
                    v             v
                4 classes      2 classes
```

The heads are simple linear layers:

- Type head: 768 → 4
- Board head: 768 → 2

The type head predicts `ML Job`, `Job`, `Other`, or `Not Job`.

The board head predicts `Regular account` or `Job board`.

### Starting from pretrained BERT

I load the pretrained weights from `bert-base-uncased` into the shared encoder.

Both classification heads start with randomly initialized weights because they are specific to my dataset and tasks.

During training, I fine-tune the entire encoder along with both heads. I don't freeze BERT.

I also use separate learning rates for the encoder and classification heads.

The encoder gets a lower learning rate because it already contains useful pretrained representations. The newly initialized classification heads can use a higher learning rate because they need to learn the mapping between BERT representations and my labels.

This allows the model to adapt BERT's existing language understanding to the patterns found in my dataset.

## 3. Training objective

Both classification heads use cross-entropy loss.

The type head calculates how far its predictions are from the actual post-type labels. The board head does the same for job-board labels.

I combine both losses into a single training objective:

```python
total_loss = (
    LAMBDA_TYPE * type_loss
    + LAMBDA_BOARD * board_loss
)
```

Mathematically:

[
L\_{\text{total}} =
\lambda_t L\_{\text{type}}
+
\lambda_b L\_{\text{board}}
]

The lambda values control how much each task contributes to training.

### Masking the secondary task

The job-board classification task is only meaningful when a post represents an opportunity.

I don't want unrelated posts to contribute to the job-board loss. For example, if an account publishes a meme or a general discussion, whether that account is a job board is irrelevant to this particular training example.

To handle this, I mask the job-board labels of all `Not Job` examples using PyTorch's ignored target index (`-100`).

This means:

- `ML Job`, `Job`, and `Other` contribute to both classification losses.
- `Not Job` contributes only to the post-type classification loss.

The encoder still learns from every example, but the board head only receives supervised learning signals from actual job posts.

This keeps the two tasks connected without forcing the secondary task to learn from irrelevant examples.

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

### Why I keep these approaches separate

Class-weighted loss and weighted sampling solve the same general problem in different ways.

With class-weighted loss, I change how much each example contributes to the optimization objective.

With weighted sampling, I change how frequently each example is presented to the model.

For the post-type task, I use either class-weighted loss or weighted sampling, depending on the configuration. I don't combine both.

The job-board task has its own optional class-weighted loss because it has a separate label distribution.

All imbalance handling is restricted to the training set. Validation and test retain their original distributions because I want to measure performance on data that hasn't been artificially balanced.

## 5. Inference

The trained model exposes a `predict()` method that accepts either a single post or a list of posts.

For every post, the model produces two sets of logits:

- Type logits from the post-type head.
- Board logits from the job-board head.

I apply softmax to convert these into normalized class probabilities.

The predicted post type is the class with the highest probability.

I also calculate the probability that a post is an opportunity:

```python
p_job = 1 - p_not_job
```

Since the three opportunity categories are `ML Job`, `Job`, and `Other`, this gives me their combined probability.

If the predicted type is `Not Job`, I don't return a job-board prediction.

Otherwise, I return the predicted publisher category.

The output includes:

- Predicted type.
- Probabilities for all four types.
- Whether the post is an opportunity.
- Combined opportunity probability.
- Whether the publisher is a job board.
- Job-board probability.

This allows downstream systems to use both the predicted category and the model's probability estimates.

The probabilities are softmax outputs and should not automatically be interpreted as calibrated confidence scores.

## 6. The complete pipeline

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
Multi-task BERT classifier
        |
        v
Shared BERT encoder
        |
        +----------------+
        |                |
        v                v
    Type head        Board head
        |                |
        v                v
   Post category     Publisher type
```

Each stage solves a different problem.

The data preparation pipeline reduces repeated content and prepares useful training examples.

The author-grouped split prevents the same author from leaking into training and evaluation.

Weighted sampling or class-weighted loss addresses class imbalance without permanently changing the dataset.

Finally, the shared BERT encoder learns to classify post types and identify job boards using two separate heads.

## Limitations

There are a few limitations worth keeping in mind.

The two classification tasks share an encoder, so their training signals can influence each other. This can be beneficial when both tasks learn useful shared representations, but it can also introduce conflicting gradients. A shared encoder doesn't automatically guarantee better performance than training separate models.

The job-board head is trained only on posts that are classified as opportunities in the ground-truth data. It doesn't learn to identify job boards from unrelated posts.

Finally, similarity-based deduplication doesn't prove that two posts are duplicates. Similarity can also come from shared templates or common job descriptions.

The goal of this project is to build a classifier that learns from useful and varied examples, rather than simply training on as many collected posts as possible.
