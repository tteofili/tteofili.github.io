# Finding related papers on CORS dataset with Lucene, Anserini and BioBERT

```python
COVIDEX_RELATED_PATH='/home/tteofili/dev/anserini/lucene-index-covid-biobert-centroids'
COVIDEX_PATH='/home/tteofili/dev/anserini/lucene-index-covid-2020-04-03'
```


```python
import json
import os
import numpy
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk"
```


```python
from pyserini.index import pyutils

index_utils = pyutils.IndexReaderUtils(COVIDEX_PATH)
```


```python
from pyserini.search import pysearch

searcher = pysearch.SimpleSearcher(COVIDEX_PATH)
query = 'asymptomatic transmission of COVID-19'
top_hits = searcher.search(query)

# Prints the first 5 hits
for i in range(0, 5):
    print(f'{i+1:2} {top_hits[i].lucene_document.get("doi")} {top_hits[i].score:.5f} {top_hits[i].lucene_document.get("title")} ')
    

```

     1 10.1101/2020.03.09.20033514 10.54510 The time scale of asymptomatic transmission affects estimates of epidemic potential in the COVID-19 outbreak 
     2 10.1007/s11427-020-1661-4 10.45820 Clinical characteristics of 24 asymptomatic infections with COVID-19 screened among close contacts in Nanjing, China 
     3 10.1101/2020.02.20.20025619 10.45820 Clinical Characteristics of 24 Asymptomatic Infections with COVID-19 Screened among Close Contacts in Nanjing, China 
     4  10.18290 Advances on presymptomatic or asymptomatic carrier transmission of COVID-19 
     5 10.3760/cma.j.cn112338-20200228-00207 10.18290 [Advances on presymptomatic or asymptomatic carrier transmission of COVID-19] 



```python
from pyserini.search import pysearch

in_doc_title = top_hits[0].lucene_document.get("title")
in_doc_id = top_hits[0].lucene_document.get("id")

nnsearcher = pysearch.SimpleNearestNeighborSearcher(COVIDEX_RELATED_PATH)
nn_hits = nnsearcher.search(in_doc_id, 3)
k = 1
for k_hits in nn_hits:
  print(f'top hits for centroid {k}')
  for i in range(0, len(k_hits)):
      if k_hits[i].id != in_doc_id:
        sdoc = searcher.doc(k_hits[i].id)
        title = sdoc.lucene_document().get('title')
        print(f' {title[:75]} {k_hits[i].score:.5f}')
  k += 1
```

    top hits for centroid 1
     Optimal timing for social distancing during an epidemic 32.03171
     Reducing the Impact of the Next Influenza Pandemic Using Household-Based Pu 31.84916
    top hits for centroid 2
     Asymptomatic Middle East Respiratory Syndrome Coronavirus (MERS-CoV) infect 77.95080
     On the Role of Asymptomatic Infection in Transmission Dynamics of Infectiou 76.57796
    top hits for centroid 3
     Detecting Differential Transmissibilities That Affect the Size of Self-Limi 75.30797
     Analyzing Vaccine Trials in Epidemics With Mild and Asymptomatic Infection 73.81608



```python
import operator
reranked = []
for hits in nn_hits:
    for i in range(0, len(hits)):
        term = hits[i].id
        if term != in_doc_id:
            sdoc = searcher.doc(term)
            title = sdoc.lucene_document().get('title')
            reranked.append([title, hits[i].score])
reranked.sort(key=operator.itemgetter(1), reverse=True)
```


```python
print('Related papers for:\n "' + in_doc_title + '"\n')
for i in range(len(reranked)):
     print(f'{i+1:2}  {reranked[i][0][:75]}... ({reranked[i][1]:.5f})')
```

    Related papers for:
     "The time scale of asymptomatic transmission affects estimates of epidemic potential in the COVID-19 outbreak"
    
     1  Asymptomatic Middle East Respiratory Syndrome Coronavirus (MERS-CoV) infect... (77.95080)
     2  On the Role of Asymptomatic Infection in Transmission Dynamics of Infectiou... (76.57796)
     3  Detecting Differential Transmissibilities That Affect the Size of Self-Limi... (75.30797)
     4  Analyzing Vaccine Trials in Epidemics With Mild and Asymptomatic Infection... (73.81608)
     5  Optimal timing for social distancing during an epidemic... (32.03171)
     6  Reducing the Impact of the Next Influenza Pandemic Using Household-Based Pu... (31.84916)
