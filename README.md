# good_job

`good_job` is a BERT-based classifier that identifies job opportunities from social media posts.

The project is built around a simple problem: social media contains a lot of useful opportunities, but finding them programmatically is difficult. Posts are often repeated, written in different formats, published by job boards, or mixed with completely unrelated content.

I wanted to build a classifier that can understand what a post is actually about instead of relying on keywords or a collection of regular expressions.

But training a classifier on raw social media data comes with its own problems. Before working on the model, I needed to make sure the dataset was useful, sufficiently varied, and split in a way that wouldn't give misleading evaluation results.

This project is divided into two main parts: preparing the dataset and training the classifier.

Link to Hugging face - https://huggingface.co/SpiderHomie/goodjob

## What does it classify?

The model predicts the type of a post using four classes.

| Label | Meaning |
|---|---|
| `ML Job` | Machine learning and AI-related jobs, including engineering and research roles |
| `Job` | Software-related paid work, including engineering jobs, freelance work, and software services |
| `Other` | Paid work that isn't directly software-related, such as data annotation, RLHF, and evaluator work |
| `Not Job` | Everything else |

It also has a second task: identifying whether the publisher is a regular account or a job board.

The two tasks share the same BERT encoder but have separate classification heads.

## How the project works

```text
             Raw social media posts
                       |
                       v
              Dataset preparation
                       |
                       v
                Text cleaning
                       |
                       v
             Similarity-based cleanup
                       |
                       v
             Author-grouped splitting
                       |
                       v
              Training dataset
                       |
                       v
              Multi-task BERT
                       |
                +------+------+
                |             |
                v             v
             Post type     Job board
             prediction    prediction
```

The dataset preparation stage focuses on removing redundant content, preserving useful examples, preventing author leakage, and handling class imbalance.

The classifier stage focuses on learning the relationship between post content, opportunity categories, and publisher types.

## Project documentation

The implementation and reasoning are explained in two separate documents.

### 1. [Data Cleaning and Dataset Preparation](docs/data_preparation.md)

Explains how I turn raw social media posts into a usable training dataset.

It covers text normalization, why I chose MiniLM and TF-IDF for similarity detection, why I use a graph instead of deleting posts based on a simple threshold, how I decide which posts to keep, and why the dataset is split by author.

It also explains how I handle class imbalance without artificially changing the validation and test distributions.

### 2. [The BERT Classifier and Training Process](docs/model_training.md)

Explains the model architecture and how it is trained.

It covers the shared BERT encoder, the two classification heads, multi-task learning, the combined training objective, masked job-board loss, imbalance handling, optimization, evaluation, and inference.

## Why this approach?

I didn't want to build a model that performs well only because it has seen the same announcements or authors during training.

A large dataset doesn't necessarily mean a diverse dataset. Repeated posts can make the training set look bigger without providing much additional information.

That's why dataset preparation is an important part of this project, rather than just a preprocessing step.

The goal is to train a classifier that learns from useful and varied examples and can make predictions on posts it hasn't seen before.
