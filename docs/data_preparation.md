# Data Cleaning and Dataset Preparation

Before training `good_job`, I needed to make sure the dataset contained useful and reasonably independent examples.

The raw data comes from social media, so it has a few problems. The same vacancy can be published multiple times, different accounts can repost the same announcement, and job boards often use similar templates. There are also differences in how frequently each class appears.

If I train directly on this data, the model might see the same information repeatedly. That would make the dataset look larger than the actual variety of examples it contains.

I decided to handle these problems before training.

## 1. Text normalization

The first step is to normalize the post text.

Two posts can contain the same information but look different because of URLs, tracking parameters, email addresses, UUIDs, HTML entities, or unnecessary whitespace.

For example, these two posts might be identical in meaning even if they contain different tracking URLs.

```text
We are hiring an ML Engineer!
Apply here: https://example.com/apply?ref=twitter

We are hiring an ML Engineer!
Apply here: https://example.com/apply?ref=linkedin
```

I remove these kinds of differences because they don't contribute much to the meaning of the announcement.

However, normalization only handles superficial differences. It cannot identify two posts that use different wording but describe the same opportunity.

That's why I needed a separate deduplication process.

## 2. Why I chose similarity-based deduplication

Exact text matching is not enough for social media posts.

The wording is different, but both posts communicate almost the same announcement.

I needed a way to identify posts that are similar in meaning, not just posts whose text is identical.

I use two methods for this: MiniLM embeddings and TF-IDF.

They capture different kinds of similarity, so I run them sequentially rather than combining their scores.

## 3. MiniLM embeddings

The first pass uses `all-MiniLM-L6-v2`.

I convert every post into an embedding. Since the embeddings are normalized, their dot product gives cosine similarity.

Cosine similarity helps identify posts that describe similar things even when they use different words.

These posts might have different wording but still have a high semantic similarity.

I use this method first because I want to catch repeated announcements that aren't necessarily written in exactly the same way.

## 4. Why I used a similarity graph instead of independent comparisons

I didn't want to treat every similarity match as an isolated decision.

When comparing posts, the relationships between them can form chains and branches. I wanted to see how these relationships were connected, which nodes were acting as central points, and which nodes could be removed without losing useful information about the remaining relationships.

That's why I represented the dataset as a graph.

Every post becomes a node. When two posts cross the graph similarity threshold, I create an edge between them and store their similarity score.

This gives me a complete view of how similar posts are connected.

### Understanding chaining and branching

Consider this example:

```text
A -------- B -------- C
   0.93       0.92
```

Here, B connects A and C.

If I only look at individual comparisons, I might decide that B is redundant and delete it. But doing that also removes the connections between B and the other nodes.

The graph makes this relationship visible before I make that decision.

This becomes more important when the structure branches:

```text
          A
          |
          B
         / \
        C   D
        |   |
        E   F
```

B is a central node connecting two branches.

Removing B doesn't just remove one post. It disconnects the relationships between the nodes on either side.

I need to understand these structures because the edges are evidence of similarity. If I delete a node carelessly, I can lose the connections that would have helped me identify other redundant posts.

### Why the central node matters

I use the graph to identify nodes that are directly connected to many other nodes.

These central nodes are important because they can represent a common announcement or template shared across multiple posts.

For example, if B is directly connected to A, C, D, and E with similarity scores above the deletion threshold, B is a strong candidate to act as a representative of that group.

Keeping B allows me to remove its directly connected redundant alternatives while retaining one representative.

However, I cannot assume that every node in the same connected component is redundant. A connected component only tells me that nodes are reachable through a chain of edges. It doesn't establish that every pair of nodes is directly similar.

### How this affects deletion

I first construct the graph and inspect its relationships. Then I use the edges above the deletion threshold to decide which nodes can be removed.

The selection process is greedy:

1. Find the node with the highest number of currently connected neighbors.
2. Keep that node as the representative.
3. Remove its directly connected neighbors.
4. Recalculate the remaining connections.
5. Repeat until no connected nodes remain.

If two nodes have the same degree, I prefer the one with longer normalized text.

The important detail is that I recalculate the connections after each selection. I don't simply delete every node that was originally connected to a central node without considering the remaining graph.

This approach preserves a representative for each group of directly connected redundant posts while allowing other parts of the graph to be processed independently.

The graph is therefore not just a way to find similar posts. It helps me understand the structure of their relationships before deleting anything.

My goal is to remove redundant content without losing useful examples or making deletion decisions based on relationships that are only indirect.

## 5. Graph construction and edge thresholds

I use two thresholds for each similarity method.

| Method | Graph threshold | Deletion threshold |
|---|---:|---:|
| MiniLM | 0.85 | Greater than 0.90 |
| TF-IDF | 0.85 | Greater than 0.90 |

The graph threshold determines which relationships are stored.

The deletion threshold determines which relationships are strong enough to influence deduplication.

I deliberately keep these thresholds separate.

An edge with a score of `0.87` tells me that two posts are related. It doesn't automatically mean one should be deleted.

An edge with a score greater than `0.90` is treated as stronger evidence of redundancy and can participate in the deletion process.

I also require both posts to have the same label before creating a normal similarity edge.

This is important because two posts can look very similar while belonging to different classes.

For example, a generic hiring announcement and a post about a non-software paid opportunity might share a template. I don't want the cleaning process to automatically remove examples from different classes just because their text looks similar.

Keeping the label constraint makes deduplication more conservative.

## 6. Why I use TF-IDF after MiniLM

MiniLM and TF-IDF capture different properties.

MiniLM focuses on semantic similarity. It can identify posts that communicate similar information even when their wording differs.

TF-IDF focuses more on shared words and phrases.

For the second pass, I use word unigrams and bigrams with TF-IDF.

This is useful for detecting repeated job announcements and templates that share a lot of their original wording.

These posts have a very similar structure and vocabulary, even though one requirement differs.

TF-IDF helps identify this kind of textual overlap.

I run the methods sequentially:

```text
Normalized posts
       |
       v
MiniLM embeddings
       |
       v
MiniLM similarity graph
       |
       v
Graph-based deduplication
       |
       v
Remaining posts
       |
       v
TF-IDF vectors
       |
       v
TF-IDF similarity graph
       |
       v
Graph-based deduplication
       |
       v
Cleaned dataset
```

I don't combine the MiniLM and TF-IDF scores. Each method gets its own pass over the remaining dataset.

This lets each method identify redundancy using its own representation.

## 7. How I decide which posts to keep

After filtering the graph to retain only edges above the deletion threshold, I process the connected components individually.

I use a greedy representative-selection algorithm.

The process is:

1. Find the node with the highest number of currently connected neighbors in the cluster.
2. Keep that node as the representative.
3. Remove its directly connected neighbors.
4. Repeat the process with the remaining nodes until no connected nodes are left.

If two nodes have the same degree, I prefer the one with longer normalized text.

### Why choose the highest-degree node?

The degree of a node tells me how many other nodes it is directly connected to.

A node with a higher degree is directly similar to more posts in the current cluster.

Keeping that node allows me to remove several redundant alternatives at once.

For example:

```text
       A
       |
B ---- C ---- D
       |
       E
```

If C has the highest degree, I keep C and remove its directly connected neighbors.

This is preferable to deleting every node in the connected component because it preserves a representative instead of discarding the entire group.

After removing those neighbors, I repeat the process on whatever remains.

This is a greedy heuristic, not a mathematically optimal solution. The objective is to reduce repeated content while preserving as much useful variety as possible.

## 8. Why I split by author instead of randomly splitting posts

Deduplication reduces repeated content, but it doesn't completely solve the problem of similar posts appearing in different datasets.

Posts from the same author can still share writing styles, templates, and recurring patterns.

If I randomly split individual posts into training and validation sets, posts from the same author could appear in both.

The model might then perform well on validation because it has already seen similar content from that author during training.

This is a form of data leakage.

To avoid this, I split the dataset by author groups rather than individual posts.

I use `StratifiedGroupKFold` with 10 folds.

When a source field is available, I use `(source, author)` as the group. This means the same username on two different platforms is treated as two separate groups.

I assign the folds as follows:

| Folds | Purpose |
|---|---|
| Fold 0 | Validation |
| Fold 1 | Test |
| Folds 2–9 | Training |

This gives an approximate 80/10/10 split.

The important constraint is that an author group cannot appear in more than one split.

## 9. Handling class imbalance

After cleaning and splitting, the training data can still have an uneven number of examples per class.

For example, there might be many more `Not Job` examples than `ML Job` examples.

If I train normally, common classes will appear more frequently in batches. The model can become biased towards those classes and perform poorly on classes with fewer examples.

I decided to handle this during training instead of permanently modifying the dataset.

For each training example, I calculate a weight based on the frequency of its class:

```python
weight = class_count ** (-class_power)
```

With `class_power=1.0`, I use full inverse-frequency weighting.

With `class_power=0.5`, I apply a softer correction.

With `class_power=0`, class balancing is disabled.

I normalize the weights so that their mean is 1.

I then use PyTorch's `WeightedRandomSampler` with replacement.

This changes the probability of selecting each example during training. Examples from minority classes have a higher probability of being selected.

Because sampling uses replacement, an example can appear multiple times in an epoch, while some examples from majority classes might not appear at all.

The original dataset remains unchanged.

I only apply this balancing to the training set. Validation and test retain their natural distributions because I want to measure performance on data that hasn't been artificially balanced.

## 10. Why these decisions matter

Each stage addresses a different source of noise or bias.

Text normalization removes superficial differences that make identical content look different.

MiniLM identifies semantic similarity, while TF-IDF identifies textual overlap and repeated templates.

The graph preserves the relationships between posts. It prevents me from treating an entire connected component as if every post inside it were a duplicate of every other post.

The separate graph and deletion thresholds let me retain weaker relationships without automatically deleting posts because of them.

The greedy representative-selection algorithm removes directly connected redundant alternatives while keeping representative examples.

Author-grouped splitting prevents the same author from appearing across training, validation, and test, reducing the risk of author-related leakage.

Finally, weighted sampling addresses class imbalance during training without permanently changing the dataset or artificially balancing evaluation data.

The overall objective is not to make the dataset as small as possible. It is to remove redundant examples while preserving useful variation, prevent leakage between splits, and give underrepresented classes a reasonable chance to contribute during training.
