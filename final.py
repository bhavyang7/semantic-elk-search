!pip install -U sentence-transformers

! pip install transformers

import pandas as pd
import numpy as np
import json
import os
import uuid


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from tqdm.auto import tqdm
tqdm.pandas()
df  = pd.read_csv("data job posts.csv")
df.shape

class Tokenizer(object):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_token(self, documents):
        sentences  = [documents]
        sentence_embeddings = self.model.encode(sentences)
        encod_np_array = np.array(sentence_embeddings)
        encod_list = encod_np_array.tolist()
        return encod_list[0]



PUT posting 
{
   "settings":{
      "index":{
         "number_of_shards":1,
         "number_of_replicas":0,
         "knn":{
            "algo_param":{
               "ef_search":40,
               "ef_construction":40,
               "m":"4"
            }
         }
      },
      "knn":true
   },
   "mappings":{
      "properties":{
         "vector":{
            "type":"knn_vector",
            "dimension":384
         },
         "title":{
            "type":"text",
            "fields":{
               "keyword":{
                  "type":"keyword",
                   "index":false
               }
            }
         },
         "company":{
            "type":"keyword",
            "index":false
         },
         "location":{
            "type":"keyword",
            "index":false
         },
         "salary":{
            "type":"keyword",
            "index":false
         },
         "job_description":{
            "type":"keyword",
            "index":false
         }
      }
   }
}



PUT posting1 
{
   "settings":{
      "index":{
         "number_of_shards":20,
         "number_of_replicas":1,
         "knn":{
            "algo_param":{
               "ef_search":40,
               "ef_construction":40,
               "m":"4"
            }
         }
      },
      "knn":true
   },
   "mappings":{
      "properties":{
         "vector":{
            "type":"knn_vector",
            "dimension":384
         }
      }
   }
}



elk_data = df.to_dict("records")
es = Elasticsearch(hosts=[ENDPOINT],  http_auth=(USERNAME, PASSWORD))
es.ping()

for x in elk_data:
    try:
        _={
           "title":x.get("Title", ""),
            "company":x.get("Company", ""),
            "location":x.get("Location", ""),
            "salary":x.get("Salary", ""),
            "vector":x.get("vector", ""),
            "job_description":x.get("JobDescription", ""),

        }
        es.index(index='posting1',body=_)
    except Exception as e:pass


INPUT = input("Enter the Input Query ")
token_vector = token_instance.get_token(INPUT)

query  ={
  "_source": ["title", "job_description"], 
   "size":50,
   "query":{
      "bool":{
         "must":[
            {
               "knn":{
                  "vector":{
                     "vector":token_vector,
                     "k":20
                  }
               }
            }
         ]
      }
   }
}


res = es.search(index='posting1',
                size=50,
                body=query,
                request_timeout=55)

title = [x['_source']  for x in res['hits']['hits']] 
title
