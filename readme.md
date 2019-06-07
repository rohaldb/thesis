# Thesis

1. I suggest you write down the model clearly before implementing it. A good practice is to write it formally, and also include a list of ‘todos’ or ‘notes’. Some of the ‘notes’ we discussed are:

  - include regularisation so that the centre points in the kernels (i.e., 𝜆s) are pushed away from each other.
  - loss function: current consensus is to include both
    - (1) a task-dependent part, which encourages the model to map kNN points of a query close to each other, while pushing non-kNN points away from them.
    - the negative examples should be several times larger than positives ones (only k per query), and a fair amount should be those that nearly make it into the top-k. I think a distance-based criteria here would be better than rank-based, so that even if some points are ranked as, say, 80th NN of the query, as long as its distance is not too large than the true k-NN of the query, it does not incur a huge loss. But we shall endow a huge loss to those negative points that are substantially far away than the k-NN of the query.
    - (2) a task-independent part, which generally preserves (up to some extent) the proximity of points in the original space.
    - Think about the margin-base loss function (the one that you missed the Relu). It has a fixed margin hyper-parameter 𝛼. This is not really fair as it does not consider the “local density” around a query point. If there are indeed many points around, it will be impossible to push the (k+1)-NN away from k-NN in the embedding space (think why?). So it probably makes sense to change this fixed 𝛼 to sth that reflects the intrinsic difficulty of this goal and set it in a realistic manner. Also, from the retrieval point of view, a fixed 𝛼 is a very strong sufficient condition to make sure that the true retrieval performance is good, but not necessarily a necessary condition for good-enough performance.
    - One possible useful measure to set 𝛼 adaptively may be the r_k, i.e., the distance of the k-th NN of the query.
    - A good practice would be to training the 2nd part first, and then fine-tune it using the a loss function that takes care of a linear combination of both parts. Think of pre-training on ImageNet or Word2Vec as an example.

2. One issue with binary encoding is that it loses too much information, despite being compact and fast. As I mentioned before, it may be alleviated by imposing a weighted Hamming distance on the binary code when doing the kNN search in the embedding space. To see why this may help, think about an embedding dimension where the corresponding weight is small. This means the kNN search in the embedding space is likely to consider both 0 and 1 on this dimension. A more fancier variant would be to determine the weight automatically and adaptively for each query.

4. Another idea to solve the same problem may be to allow mapping one data point to multiple binary codes to compensate this!! The naive idea is that if the real value before the binarization is close to the cut-off point, we map it to both 0 and 1. BTW, a similar idea is the Spill Tree in [1]. Of course, we need to control the expansion ratio explicitly (otherwise, we get 2^d ratio), and this induces non-trivial challenges to ML in terms of how to fairly compute its loss and also enable gradient to propagate back correctly.

[1] Ting Liu, Andrew W. Moore, Alexander G. Gray, Ke Yang: An Investigation of Practical Approximate Nearest Neighbor Algorithms. NIPS 2004: 825-832.
