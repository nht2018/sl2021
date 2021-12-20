# sl2021
statistic learning 2021
many of the files are useless,the useful ones include:
  preprocess.py
  this file preprocess the two original dataset into three set:train/validate/test.json, in each file format:{'1':{},'2':{},'3':{},````}
  which zxy wants
 
  CounterVectorizer_svm.py
  this is the method we found in https://www.kaggle.com/rickyhp/predicting-protein-sequences-with-svm
  the method:
  sequence of different length -->a sparse matrix,where exponents are the set of all '4(or 5,6 if you like) nearest charactors',and each row of matrix,represents
  a certain protein-sequence, shows how many times does each exponents exist in the sequence.
  thus, different length-->the samme length, CounterVectorizer does this job
  then apply MultinomialNB, svm(I wait really long time), and linear svm to fit the traindata,that's all I did.
  
 by the way, the trainset in preprocess and CounterVectorize is differerent and I need to change a bit
