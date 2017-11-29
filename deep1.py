

"""DNNRegressor with custom estimator for candidates and job positions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.preprocessing.text import VocabularyProcessor
from sklearn import metrics

import argparse
import sys
import os
import csv



import numpy as np
import tensorflow as tf
import pandas as pd



FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
LEARNING_RATE = 0.01

CORPUS = "f:/tmp/validation-raw.csv"

CSV_COLUMNS = ["skills", "experience", "education", "title", "keywords", 
                "label","job_id","candidate_id"]
COLUMN_TYPES = {"skills": np.str, "experience": np.str, "education": np.str, 
                "title": np.str, "keywords":np.str, "label": np.int,
                "job_id":np.int, "candidate_id":np.int}


MAX_SKILLS_LENGTH = 30
MAX_EXPERIENCE_LENGHT = 10 
MAX_EDUCATION_LENGHT = 5
MAX_TITLE_LENGHT = 2
MAX_KEYWORD_LENGHT = 15

MIN_SKILLS_FREQUENCY = 20
MIN_EXPERIENCE_FREQUENCY = 75
MIN_EDUCATION_FREQUENCY = 5
MIN_TITLE_FREQUENCY = 0
MIN_KEYWORD_FREQUENCY = 5


MAX_LABEL = 2
#
#EMBEDDING_SIZE_SKILLS = 10
#EMBEDDING_SIZE_EXP = 10
#EMBEDDING_SIZE_EDUC = 10
#EMBEDDING_SIZE_TITLE = 5
#EMBEDDING_SIZE_KWRDS = 7

EMBEDDING_SIZE_SKILLS = 16
EMBEDDING_SIZE_EXP = 16
EMBEDDING_SIZE_EDUC = 16
EMBEDDING_SIZE_TITLE = 12
EMBEDDING_SIZE_KWRDS = 13


def save_metadata(vocabulary, filename):
  """Stores the words and their mapping words ids to a tsv file."""
  f=open(filename, "w", encoding = "utf8")
  f.write("word_id" +"\t" + "word" + "\n")
  for key, item in vocabulary.vocabulary_._mapping.items():
    f.write(str(item) + "\t" + str(key) + "\n")
  f.close()
  

def mapwords(data, mode):
  """maps words stored in pandas dataframe to inique ids  
    Args
    data: dictionary with pandas dataframes for skills, experience, education.
          title, keywords {"skills: dataframe with words in each row, 
                           "experience": dataframe.....}
    mode: keyword which can have two values {"train", "test"}
    
    returns a dictionary with arrays -> {"skills : arrays with word ids,
                                         "education: arrays....}  
  """
  x = {}  
  # Transform the features from keywords to numbers
  if mode == "train":
    x_transform_skls = vocab_processor_skls.fit_transform(data["skills"])
    x_transform_exp = vocab_processor_exp.fit_transform(data["experience"])
    x_transform_edc = vocab_processor_edc.fit_transform(data["education"])
    x_transform_tl = vocab_processor_tl.fit_transform(data["title"])
    x_transform_kwrd = vocab_processor_kwrd.fit_transform(data["keywords"])
  elif mode =="test":
    x_transform_skls = vocab_processor_skls.transform(data["skills"])
    x_transform_exp = vocab_processor_exp.transform(data["experience"])
    x_transform_edc = vocab_processor_edc.transform(data["education"])
    x_transform_tl = vocab_processor_tl.transform(data["title"])
    x_transform_kwrd = vocab_processor_kwrd.transform(data["keywords"])
    
    
  # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
  # ids start from 1 and 0 means 'no word'. But
  # categorical_column_with_identity assumes 0-based count and uses -1 for
  # missing word.  
  x["skills"] = np.array(list(x_transform_skls)) - 1
  x["experience"] = np.array(list(x_transform_exp)) - 1
  x["education"] = np.array(list(x_transform_edc)) - 1
  x["title"] = np.array(list(x_transform_tl)) - 1
  x["keywords"] = np.array(list(x_transform_kwrd)) - 1  
  return x

def load_data(filepath, mode):
  """load the data from file.
  args:
    filepath: the file path of the datafile
    mode: "train", "test"
    
  returns dicts with pd.Dataframes
    if train data dict -> {"skills": , "experience":, "education", 
                           "title":,"keywords": }
    if test data dict -> {"skills": , "experience":, "education", 
                           "title":,"keywords": , "job_id":, "candidate_id": }
  """
  dataframes = {}
  df_data = pd.read_csv(
      filepath,
      names=CSV_COLUMNS,
      dtype=COLUMN_TYPES
      )
  x_skills = pd.Series(np.array(df_data["skills"]))
  x_experience = pd.Series(np.array(df_data["experience"]))
  x_education = pd.Series(np.array(df_data["education"]))
  x_title = pd.Series(np.array(df_data["title"]))
  x_keywords = pd.Series(np.array(df_data["keywords"]))
  y_label = pd.Series(np.array(df_data["label"]))
  
  # Replace the NA's values with an empty string keyword 
  x_skills.fillna("", inplace = True)
  x_experience.fillna("", inplace = True)
  x_education.fillna("", inplace = True)
  x_title.fillna("", inplace = True)
  x_keywords.fillna("", inplace = True)
  
  dataframes["skills"] = x_skills
  dataframes["experience"] = x_experience
  dataframes["education"] = x_education
  dataframes["title"] = x_title
  dataframes["keywords"] = x_keywords
  dataframes["label"] = y_label
  
  if mode == "train":
    return dataframes    
  elif mode == "test":
    y_job_id = pd.Series(np.array(df_data["job_id"]))
    y_candidate_id = pd.Series(np.array(df_data["candidate_id"]))
    dataframes["job_id"] = y_job_id
    dataframes["candidate_id"] = y_candidate_id
    return dataframes


def estimator_spec_for_softmax_classification(
    logits, labels, mode, params):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(
            learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  # Add some custom evaluation metrics  
  accuracy = tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  precision = tf.metrics.precision(
          labels=labels, predictions=predicted_classes)
  recall = tf.metrics.recall(
               labels = labels, predictions = predicted_classes)
  f1 = (tf.divide(
          tf.multiply(2 * precision[0],recall[0]), tf.add(precision[0],recall[0])
          ), 
         precision[1])
  auc_roc = tf.metrics.auc(
          labels = labels, predictions = predicted_classes
          )
  eval_metric_ops = {
      "accuracy" : accuracy, 
      "precision" : precision,
      "recall" : recall,
      "f1" : f1,
      "auc_roc" : auc_roc
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        
          
def deep_fn(features, labels, mode, params):
  """A bag-of-words model DNN with one hidden layer """
  
  #Bag of words colunns with intenger words ids
  bow_column_skls = tf.feature_column.categorical_column_with_identity(
        "skills", num_buckets=n_skills)
  bow_column_exp = tf.feature_column.categorical_column_with_identity(
            "experience", num_buckets = n_experience)
  bow_column_edc = tf.feature_column.categorical_column_with_identity(
            "education", num_buckets = n_education)
  bow_column_tl = tf.feature_column.categorical_column_with_identity(
            "title", num_buckets = n_title)
  bow_column_kwrd = tf.feature_column.categorical_column_with_identity(
            "keywords", num_buckets = n_keywords)
  

    #columns with embedding of keywords
  with tf.name_scope("Embeddings_input"):  
    bow_embedding_column_skls = tf.feature_column.embedding_column(
        bow_column_skls, dimension=EMBEDDING_SIZE_SKILLS, combiner = "sqrtn")
    bow_embedding_column_exp = tf.feature_column.embedding_column(
        bow_column_exp, dimension=EMBEDDING_SIZE_EXP, combiner = "sqrtn")
    bow_embedding_column_edc = tf.feature_column.embedding_column(
        bow_column_edc, dimension=EMBEDDING_SIZE_EDUC, combiner = "sqrtn")
    bow_embedding_column_tl = tf.feature_column.embedding_column(
        bow_column_tl, dimension=EMBEDDING_SIZE_TITLE, combiner = "sqrtn")
    bow_embedding_column_kwrd = tf.feature_column.embedding_column(
        bow_column_kwrd, dimension=EMBEDDING_SIZE_KWRDS, combiner = "sqrtn") 

  bow = tf.feature_column.input_layer(
        features,
        feature_columns=[bow_embedding_column_skls, bow_embedding_column_exp,
                         bow_embedding_column_edc, bow_embedding_column_tl,
                         bow_embedding_column_kwrd])
    
  first_hidden_layer = tf.layers.dense(bow, 1000, activation = tf.nn.relu,
                                       name="hiddel_layer1")

      
  logits = tf.layers.dense(first_hidden_layer, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode, params=params)


def main(unused_argv):
  global n_skills
  global n_experience
  global n_education
  global n_title
  global n_keywords
  global vocab_processor_skls
  global vocab_processor_exp
  global vocab_processor_edc
  global vocab_processor_tl
  global vocab_processor_kwrd
  
  
  ####################### File names preprocessing ######################### 
  #check if the model folder exist and create it otherwise
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    
  #Files for saving the results ******not working yet
  predictions_file = FLAGS.model_dir + "/predictions.csv"
#  emb_filenames = ["emb_skls.csv" , "emb_exp.csv", "emb_edu.csv", "emb_title.csv",
#                   "emb_kwrd.csv"]
#  emb_filepaths = [FLAGS.model_dir + "/" + f for f in emb_filenames]
  
  
  metrics_file = FLAGS.model_dir + "/sk_metrics.csv"
  
  #Files for storing the dictionaries of keywords in skills,experience,education
  #title, keywords
#  voc_filenames = ["skills_voc.txt", "education_voc.txt", "experience_voc.txt", 
#                   "title_voc.txt", "keywords_voc.txt"]
#  voc_skills_file = FLAGS.model_dir + "/" + "skills_voc.txt"
#  voc_education_file = FLAGS.model_dir + "/" + "education_voc.txt"
#  voc_experience_file = FLAGS.model_dir + "/" + "experience_voc.txt"
#  voc_title_file = FLAGS.model_dir + "/" + "title_voc.txt"
#  voc_keywords_file = FLAGS.model_dir + "/" + "keywords_voc.txt"
  
   #################  Vocabularies ################### 
  vocab_processor_skls = VocabularyProcessor(
      max_document_length = MAX_SKILLS_LENGTH,
      min_frequency = MIN_SKILLS_FREQUENCY)
  vocab_processor_exp = VocabularyProcessor(
      max_document_length = MAX_EXPERIENCE_LENGHT,
      min_frequency = MIN_EXPERIENCE_FREQUENCY
      )
  vocab_processor_edc = VocabularyProcessor(
      max_document_length = MAX_EDUCATION_LENGHT,
      min_frequency = MIN_EDUCATION_FREQUENCY
      )
  vocab_processor_tl = VocabularyProcessor(
      max_document_length = MAX_TITLE_LENGHT,
      min_frequency = MIN_TITLE_FREQUENCY
      )
  vocab_processor_kwrd = VocabularyProcessor(
      max_document_length = MAX_KEYWORD_LENGHT,
      min_frequency = MIN_KEYWORD_FREQUENCY
      )
  # check if there is already stored vacabulary file for each feature and
  # load it 
#  isVocabulary = [0, 0, 0, 0, 0]
#  if os.path.isfile(voc_skills_file): 
#    vocab_processor_skls.restore(filename = voc_skills_file)
#    isVocabulary[0]=1
#  if os.path.isfile(voc_education_file): 
#    vocab_processor_edc.restore(filename = voc_education_file)
#    isVocabulary[1]=1
#  if os.path.isfile(voc_experience_file): 
#    vocab_processor_exp.restore(filename = voc_experience_file)
#    isVocabulary[2]=1
#  if os.path.isfile(voc_title_file): 
#    vocab_processor_tl.restore(filename = voc_title_file)
#    isVocabulary[3]=1
#  if os.path.isfile(voc_keywords_file): 
#    vocab_processor_kwrd.restore(filename = voc_keywords_file)
#    isVocabulary[4]=1
#  
#  #check if all the vacubalary files exist in not feed all vocabularies
#  if sum(isVocabulary) < 5 :  
  print("Feeding vovabularies...")
  corpus_data = load_data(CORPUS, mode="train")
  feed_vocabularies = mapwords(corpus_data, mode="train")
  corpus_data = None
  feed_vocabularies = None

   ################ Load Train and Test Data  ###################### 
  train_data = load_data(FLAGS.train_data, mode="test")
  test_data = load_data(FLAGS.test_data, mode="test")
  
  
  
  y_train = train_data["label"]
  y_test = test_data["label"]
  y_test_job_id = test_data["job_id"]
  y_test_cand_id = test_data["candidate_id"]
  
  ##transform the words into words_ids using the vocabularies
  x_train_ids = mapwords(train_data, mode="train")
  x_test_ids = mapwords(test_data, mode="test")
  
  #Free memmory
  train_data = None
  test_data = None
  
  # print the each vobavulary sizes
  n_skills = len(vocab_processor_skls.vocabulary_)
  n_experience = len(vocab_processor_exp.vocabulary_)
  n_education = len(vocab_processor_edc.vocabulary_)
  n_title = len(vocab_processor_tl.vocabulary_)
  n_keywords = len(vocab_processor_kwrd.vocabulary_)
  print('Total skills: %d' % n_skills)
  print('Total experience: %d' % n_experience)
  print('Total education: %d' % n_education)
  print('Total job titles: %d' % n_title)
  print('Total keywords: %d' % n_keywords)  
  
  #save the vocaubaries to 
#  vocab_processor_skls.save(voc_skills_file)
#  vocab_processor_exp.save(voc_experience_file)
#  vocab_processor_edc.save(voc_education_file)
#  vocab_processor_tl.save(voc_title_file)
#  vocab_processor_kwrd.save(voc_keywords_file)
  
  #these files are useful in Tensorboard's embeddings plotting
  save_metadata(vocab_processor_skls, FLAGS.model_dir + "/metadata_skills.tsv")
  save_metadata(vocab_processor_exp, FLAGS.model_dir + "/metadata_experience.tsv")
  save_metadata(vocab_processor_edc, FLAGS.model_dir + "/metadata_education.tsv")
  save_metadata(vocab_processor_tl, FLAGS.model_dir + "/metadata_title.tsv")
  save_metadata(vocab_processor_kwrd, FLAGS.model_dir + "/metadata_keywords.tsv")
  
    
  
  ################## Training and Evaluation #############################
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=x_train_ids,
      y=y_train,
      batch_size= FLAGS.train_batch,
      num_epochs=None,
      shuffle=True)
  
  # Set model params
  model_params = {"learning_rate": LEARNING_RATE}
  model_fn = deep_fn
  
  classifier = tf.estimator.Estimator(model_fn, model_dir= FLAGS.model_dir,
                                      params= model_params)
  classifier.train(input_fn=train_input_fn,  steps=FLAGS.train_steps)
  
  
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=x_test_ids,
      y=y_test,
      batch_size= len(y_test),
      num_epochs=1,
      shuffle=False)
  
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted.reshape((y_predicted.shape[0], 1))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)
  myresults = pd.DataFrame(data = y_predicted, columns = ["prediction"])
  myresults["target"] = pd.Series(y_test)
  myresults["job_id"] = pd.Series(y_test_job_id)
  myresults["candidate_id"] = pd.Series(y_test_cand_id)
  myresults.to_csv(predictions_file, index=False)
#  
  sk_accuracy = metrics.accuracy_score(y_test, myresults['prediction'])
  sk_precision = metrics.precision_score(y_test, myresults['prediction'])
  sk_recall = metrics.recall_score(y_test, myresults['prediction'])
  sk_f1 = metrics.f1_score(y_test, myresults['prediction'])
  sk_ROC = metrics.roc_auc_score(y_test, myresults['prediction'])
  sk_metrics = [sk_accuracy,sk_precision,sk_recall,sk_f1,sk_ROC]
  with open(metrics_file, 'a', newline='',encoding="utf8") as f1:
    writer = csv.writer(f1)
    writer.writerow(sk_metrics)
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))
  
  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
  
  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )  
#  parser.add_argument(
#      "--model_type",
#      type=str,
#      default="deep",
#      help="Valid model types: {'deep', 'deep_0'}."
#      )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=None,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  parser.add_argument(
      "--train_batch",
      type=int,
      default=50000,
      help="Batch size for the train data."
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)