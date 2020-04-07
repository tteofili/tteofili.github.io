# Finding related papers on CORS dataset with Lucene, Anserini and S-BERT

```
%cd
!apt-get install maven -qq >& /dev/null
!git clone https://github.com/castorini/anserini.git >& /dev/null
%cd anserini
!mvn clean package appassembler:assemble -q -Dmaven.javadoc.skip=true >& /dev/null

```

    /root
    /root/anserini
    Branch 'emb-file-read-bug' set up to track remote branch 'emb-file-read-bug' from 'origin'.
    Switched to a new branch 'emb-file-read-bug'



```
%%capture
!pip install pyserini==0.8.1.0

import json
import os
import numpy
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
```


```
%%capture
!wget https://www.dropbox.com/s/uvjwgy4re2myq5s/lucene-index-covid-2020-03-20.tar.gz
!tar xvfz lucene-index-covid-2020-03-20.tar.gz
```


```
!du -h lucene-index-covid-2020-03-20
```

    1.3G	lucene-index-covid-2020-03-20


Get pretrained s-BERT model (from *'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks'* by Nils Reimers, Iryna Gurevych, see https://arxiv.org/abs/1908.10084)




```
!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
```

    Collecting sentence-transformers
    [?25l  Downloading https://files.pythonhosted.org/packages/07/32/e3d405806ea525fd74c2c79164c3f7bc0b0b9811f27990484c6d6874c76f/sentence-transformers-0.2.5.1.tar.gz (52kB)
    [K     |████████████████████████████████| 61kB 2.0MB/s 
    [?25hCollecting transformers==2.3.0
    [?25l  Downloading https://files.pythonhosted.org/packages/50/10/aeefced99c8a59d828a92cc11d213e2743212d3641c87c82d61b035a7d5c/transformers-2.3.0-py3-none-any.whl (447kB)
    [K     |████████████████████████████████| 450kB 8.3MB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from sentence-transformers) (4.38.0)
    Requirement already satisfied: torch>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from sentence-transformers) (1.4.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from sentence-transformers) (1.18.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from sentence-transformers) (0.22.2.post1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from sentence-transformers) (1.4.1)
    Requirement already satisfied: nltk in /usr/local/lib/python3.6/dist-packages (from sentence-transformers) (3.2.5)
    Collecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/a6/b4/7a41d630547a4afd58143597d5a49e07bfd4c42914d8335b2a5657efc14b/sacremoses-0.0.38.tar.gz (860kB)
    [K     |████████████████████████████████| 870kB 52.1MB/s 
    [?25hRequirement already satisfied: boto3 in /usr/local/lib/python3.6/dist-packages (from transformers==2.3.0->sentence-transformers) (1.12.23)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers==2.3.0->sentence-transformers) (2.21.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers==2.3.0->sentence-transformers) (2019.12.20)
    Collecting sentencepiece
    [?25l  Downloading https://files.pythonhosted.org/packages/74/f4/2d5214cbf13d06e7cb2c20d84115ca25b53ea76fa1f0ade0e3c9749de214/sentencepiece-0.1.85-cp36-cp36m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 50.2MB/s 
    [?25hRequirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->sentence-transformers) (0.14.1)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from nltk->sentence-transformers) (1.12.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers==2.3.0->sentence-transformers) (7.1.1)
    Requirement already satisfied: botocore<1.16.0,>=1.15.23 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers==2.3.0->sentence-transformers) (1.15.23)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers==2.3.0->sentence-transformers) (0.9.5)
    Requirement already satisfied: s3transfer<0.4.0,>=0.3.0 in /usr/local/lib/python3.6/dist-packages (from boto3->transformers==2.3.0->sentence-transformers) (0.3.3)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers==2.3.0->sentence-transformers) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers==2.3.0->sentence-transformers) (2.8)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers==2.3.0->sentence-transformers) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers==2.3.0->sentence-transformers) (2019.11.28)
    Requirement already satisfied: docutils<0.16,>=0.10 in /usr/local/lib/python3.6/dist-packages (from botocore<1.16.0,>=1.15.23->boto3->transformers==2.3.0->sentence-transformers) (0.15.2)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /usr/local/lib/python3.6/dist-packages (from botocore<1.16.0,>=1.15.23->boto3->transformers==2.3.0->sentence-transformers) (2.8.1)
    Building wheels for collected packages: sentence-transformers, sacremoses
      Building wheel for sentence-transformers (setup.py) ... [?25l[?25hdone
      Created wheel for sentence-transformers: filename=sentence_transformers-0.2.5.1-cp36-none-any.whl size=67076 sha256=b2747f4a5428838f260dd7fac018df0def3006de52dfaa63507f65d84c25c71d
      Stored in directory: /root/.cache/pip/wheels/22/ca/b4/7ca542b411730a8840f8e090df2ddacffa1c4dd9f209684c19
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.38-cp36-none-any.whl size=884628 sha256=90ebb852349c222c16feb54f05fed491166351c085a162bdf650125476f5d10f
      Stored in directory: /root/.cache/pip/wheels/6d/ec/1a/21b8912e35e02741306f35f66c785f3afe94de754a0eaf1422
    Successfully built sentence-transformers sacremoses
    Installing collected packages: sacremoses, sentencepiece, transformers, sentence-transformers
    Successfully installed sacremoses-0.0.38 sentence-transformers-0.2.5.1 sentencepiece-0.1.85 transformers-2.3.0



<p style="color: red;">
The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>
We recommend you <a href="https://www.tensorflow.org/guide/migrate" target="_blank">upgrade</a> now 
or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:
<a href="https://colab.research.google.com/notebooks/tensorflow_version.ipynb" target="_blank">more info</a>.</p>



    100%|██████████| 405M/405M [00:17<00:00, 23.0MB/s]


Get paper tuples in the form `[doi, text, file]`





```
from pyserini.index import pyutils
sentences = []
index_utils = pyutils.IndexReaderUtils('lucene-index-covid-2020-03-20/')
for i in range(45000):
  try :
    json_obj = json.loads(index_utils.get_raw_document_contents(str(i)))
    text = json_obj['abstract']
    if text == '':
      text = json_obj['title']
    title = json_obj['title']
    doi = json_obj['doi']
    sentences.append([doi, text, title])
  except Exception:
    pass
```


```
sentences_array = numpy.array(sentences)
sentences_array.shape
```




    (15693, 3)



Extract sentence embeddings from papers' texts using s-BERT



```
sentence_embeddings = model.encode(sentences_array[:,1].tolist())
```

Create GloVe-style embedding file (each row is in the form: *doi f1 f2 ... fn* )


```
f = open('sembs.txt', 'w')
for doi, embedding in zip(sentences_array[:,0], sentence_embeddings):
  line = doi+' '+numpy.array2string(embedding, separator=' ', suppress_small=True, max_line_width=1000000).replace("[","").replace("]","\n")
  f.write(line)
f.close()
```

Index s-BERT embedding model in Lucene


```
!target/appassembler/bin/IndexVectors -input sembs.txt -path lucene-index-covid-sentemb -encoding fw -fw.q 80
```

    Loading model sembs.txt
    Creating index at lucene-index-covid-sentemb...
    12599 words indexed
    Index size: 13MB
    Total time: 00:00:24


Find papers similar to an input one (by DOI) via approximate nearest neighbour search (over s-BERT embeddings)


```
!target/appassembler/bin/ApproximateNearestNeighborSearch -input sembs.txt -path lucene-index-covid-sentemb -encoding fw -fw.q 80 -word 10.1128/JVI.01958-12
```

    Loading model sembs.txt
    Reading index at lucene-index-covid-sentemb
    10 nearest neighbors of '10.1128/JVI.01958-12':
    1. 10.1128/JVI.01958-12 (68.491)
    2. 10.1515/hsz-2014-0156 (61.349)
    3. 10.1128/JVI.00627-16 (60.989)
    4. 10.1128/JVI.00902-18 (60.734)
    5. 10.1128/JVI.00720-11 (60.570)
    6. 10.1128/JVI.02564-13 (60.457)
    7. 10.1073/pnas.1618310114 (60.207)
    8. 10.1128/JVI.02643-12 (60.061)
    9. 10.1128/JVI.03069-15 (60.016)
    10. 10.1038/cmi.2017.15 (59.502)
    Search time: 378ms



```
def get_title(doi):
  return list(filter(lambda x:doi in x, sentences))[0][2]

query = get_title('10.1128/JVI.01958-12') 
print('Papers similar to:\n  "'+query+'":\n')
results = ['10.1515/hsz-2014-0156', '10.1128/JVI.00627-16', '10.1128/JVI.00902-18', '10.1128/JVI.00720-11', '10.1128/JVI.02564-13']
i = 1
for r in results:
  print(str(i)+'. '+get_title(r))
  i += 1

```

    Papers similar to:
      "Severe Acute Respiratory Syndrome Coronavirus Protein nsp1 Is a Novel Eukaryotic Translation Inhibitor That Represses Multiple Steps of Translation Initiation":
    
    1. The leader proteinase of foot-and-mouth disease virus: structure-function relationships in a proteolytic virulence factor
    2. Infectious Bronchitis Coronavirus Limits Interferon Production by Inducing a Host Shutoff That Requires Accessory Protein 5b
    3. Inhibition of Stress Granule Formation by Middle East Respiratory Syndrome Coronavirus 4a Accessory Protein Facilitates Viral Translation, Leading to Efficient Virus Replication
    4. Mechanism of Glycyrrhizic Acid Inhibition of Kaposi's Sarcoma-Associated Herpesvirus: Disruption of CTCF-Cohesin-Mediated RNA Polymerase II Pausing and Sister Chromatid Cohesion
    5. Suppression of PACT-Induced Type I Interferon Production by Herpes Simplex Virus 1 Us11 Protein
