# sl2021
statistic learning 2021
<<<<<<< HEAD

branch nht

=======
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
>>>>>>> origin/wzy

===============
to run the entire pipeline: 
open kernel_svm version_newest.ipynb, and run each block consecutively.


** a number of several hyperparameters are adjustable, plz refer to them in block 7.

HID_DIM = 16   # this refers to the hidden size of Rnn in both encoder/ decoder 
N_LAYERS = 2   # this refers to the numbers of layers of Rnn in both encoder/ decoder
TARG_LENGTH = 16 # this refers to the output sequence length of seq2seq2 module.
reducing these number will lead to a simpler model.


** block 8 randomly generates data required for training,
the variable "datanum" in this block refers to the number of data to be generated.
this can be used to debug the pipeline,
in actual training process, there is no need to run block 8. (run block 7, and directly jump to block 9)


** notice that in block 6 -> kernel_np(self, x1, x2), 
and in block 10: there are several lines before  the command "model.update_alpha(torch.tensor(alpha_list))"
these are crucial datatype transformation that must be undertaken.

**



