
# Reduce embeddings dimensionality using "Simple and Effective Dimensionality Reduction for Word Embeddings"
> Use algorithm proposed in "Simple and Effective Dimensionality Reduction for Word Embeddings" paper to reduce dimensionality of word embeddings

Post process embeddings using "Simple and Effective Dimensionality Reduction for Word Embeddings"

Post processing algorithm (ppa)

```python
from sklearn.decomposition import PCA
import copy
import numpy as np

def ppa(X_train, n_components):
    # PCA to get top components
    pca =  PCA(n_components = n_components)
    X_train = X_train - np.mean(X_train)
    X_fit = pca.fit_transform(X_train)
    U1 = pca.components_
    
    z = []

    # removing projections on top components
    for i, x in enumerate(X_train):
        for u in U1[0:7]:        
            x = x - np.dot(u.transpose(),x) * u 
        z.append(x)

    z = np.asarray(z)
    return z
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
reduced = ppa(weights, weights.shape[1])
```

```python
pca = PCA(n_components = dim)
reduced = reduced - np.mean(reduced)
principalComponents = pca.fit_transform(reduced)
```

```python
reduced = ppa(principalComponents, dim)
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




    [('Defenseman_Anssi_Salmela', 1.0),
     ('Castle_Rushen_High', 1.0),
     ('Orb_Audio', 1.0),
     ('Lija_Athletic', 1.0),
     ('Iosefa_Tekori', 1.0),
     ('multiton', 1.0),
     ('Livno', 1.0),
     ('Seagate_ST1_Series', 1.0),
     ('www.towerbancorp.com', 1.0),
     ('OSI_Geospatial_Signs', 1.0)]



Save the newly created model

```python
outv.save_word2vec_format('/path/to/ppa-pca-ppa-reduced-8.txt', binary=False)
```
