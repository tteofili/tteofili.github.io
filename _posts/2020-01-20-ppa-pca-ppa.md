
# Reduce embeddings dimensionality using "Simple and Effective Dimensionality Reduction for Word Embeddings"
> Use algorithm proposed in "Simple and Effective Dimensionality Reduction for Word Embeddings" paper to reduce dimensionality of word embeddings

Post processing algorithm (ppa)

```python
from sklearn.decomposition import PCA
import copy
import numpy as np

def ppa(embedding_matrix_orig, n_components):
    pca = PCA(n_components=n_components)
    embedding_matrix = copy.deepcopy(embedding_matrix_orig)
    temp = embedding_matrix - np.average(embedding_matrix, axis=0)
    principalComponents = pca.fit_transform(temp)
    principalAxes = pca.components_
    toSubstract = np.matmul(np.matmul(embedding_matrix, principalAxes.T), principalAxes)
    processed = embedding_matrix - toSubstract
    return processed
```

Use Gensim to get the pretrained GoogleNews word2vec model

```python
import gensim.downloader as api

model = api.load('word2vec-google-news-300')
```

Backup model weights

```python
weights = np.copy(model.vectors)
```

Define the target embeddings dimensionality 

```python
dim = 8
```

Run ppa, PCA and ppa again with the target dimensionality

```python
reduced = ppa(weights, n_components = 300)
```

```python
pca = PCA(n_components = dim)
reduced = reduced - np.mean(reduced)
principalComponents = pca.fit_transform(reduced)
```

```python
reduced = ppa(principalComponents, n_components = dim)
```

Create a new model, identical to the original one, but with the reduced weights

```python
from gensim.models import KeyedVectors
outv = KeyedVectors(dim)
outv.vocab = model.vocab
outv.index2word = model.index2word
outv.vectors = reduced
```

```python
outv.most_similar(positive="man", topn=10)
```




    [('Miki_Otomo', 0.9952907562255859),
     ('possible.The', 0.9908273220062256),
     ('Goa_Kamat', 0.9899711608886719),
     ('Hensyel', 0.9896929264068604),
     ('Destination_PEACE', 0.9877277612686157),
     ('Peteraf', 0.9874942302703857),
     ('Gaoganediwe', 0.9873812198638916),
     ('Islamofascism_Awareness_Week', 0.9869686365127563),
     ('Balla_Keita', 0.985584557056427),
     ('Nasdaq_AUTH', 0.9855645895004272)]



Save the newly created model

```python
outv.save_word2vec_format('/path/to/ppa-pca-ppa-reduced-8.txt', binary=False)
```
