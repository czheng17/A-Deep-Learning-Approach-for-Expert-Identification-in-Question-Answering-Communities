# -*- coding: utf-8 -*-
import numpy as np
import random


#################################################################################
#################################################################################
#
# These part refer to the raw data,test raw data, cnn word vocabulary,
# userid vocabulary, and userid list
#
#################################################################################
#################################################################################

# this function is use for build up the cnn_word_vocabulary
# {'UNKNOWN':0,'nihao':2...}
def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open('../../8_21_stack_overflow_data/qu/vector_only_question.txt'):
        items = line.strip().split(' ')
        word = items[0]
        if not word in vocab:
            vocab[word] = code
            code += 1
    return vocab

# this function is use for build up the userid_vocabulary
# {'UNKNOWN':0,'aslan':1...}
def build_userid_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open('../../8_21_stack_overflow_data/qu/vector_only_userid.txt'):
    #for line in open('vectors_userid.txt'):
        items = line.strip().split(' ')
        word = items[0]
        if not word in vocab:
            vocab[word] = code
            code += 1
    return vocab
# print (build_userid_vocab())

# this function is use for put the raw data of train_use.txt into a list
#[   [  [0 , qid, question, userid1, userid2  ],[0 , qid, question, userid1, userid2  ]...   ]
# ['0', 'qid:0', 'Google__已经__关闭__或__停止__维护__的__产品__有__哪些__？___<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>', 'zhao-wei', 'xljroy']
def read_raw():
    raw = []
    for line in open('../../8_21_stack_overflow_data/qu/train_use.txt'):
        items = line.strip().split(' ')
        if items[0] == '0':
            raw.append(items)
    # print raw
    return raw
# print(read_raw()[0])

# this function is use for put the raw data of test_use.txt into a list
#[   [  [0 , qid, question, userid1, userid2  ],[0 , qid, question, userid1, userid2  ]...   ]
def test_read_raw():
    raw = []
    for line in open('../../8_21_stack_overflow_data/qu/test1_use.txt'):
        items = line.strip().split(' ')
        raw.append(items)
    # print raw
    return raw
# print(test_read_raw()[0])

def read_neg_alist():
    alist = []
    for line in open('../../8_21_stack_overflow_data/qu/train_use.txt'):
        items = line.strip().split(' ')
        alist.append(items[len(items)-1])
    print('read random neg answerlist done ......')
    return alist

##############################################################
# the function of this:
# at first, eg. string is "my_name_is_aslan"
# in vocabluary,{"my":120, "name":210, "is":310, "aslan":610}
# the result is [120,210,310,610]
# in this list,  the length is 50, so [120,210,310,610,...,xx]
##############################################################
'''
for example:
string:
    Google__已经__关闭__或__停止__维护__的__产品__有__哪些__？___<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>_<a>
return:
    [994, 0, 251, 0, 1474, 0, 93, 0, 2066, 0, 2327, 0, 4, 0, 161, 0, 7, 0, 17, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'''
def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x
# print(encode_sent(build_vocab(),read_raw()[0][2],50))

#############################################################################################
# the function of this:
# Generate the word embedding list (also a matrix)
# for the  word,"UNKNOWN", use [0,0,0,0,0,0,...,0] totally 100 dimensions
# for the other words, just extract values in the txt file and generate the 100 dimensions.
#############################################################################################
def load_vectors():
    vectors = []
    temp=[]
    for i in range(0,100):
        temp.append(0.0)
    vectors.append(temp)
    aslan_temp_i = 0
    for line in open('../../8_21_stack_overflow_data/qu/vector_only_question.txt'):
        if aslan_temp_i > 0 :
            items = line.strip().split(' ')
            if (len(items) < 101):
                continue
            vec = []
            for i in range(1, 101):
                vec.append(float(items[i]))
            vectors.append(vec)
        if aslan_temp_i == 0:
            aslan_temp_i = aslan_temp_i+1
    # print vectors
    return vectors
# print(load_vectors()[10])

def load_userid_vectors():
    vectors = []
    temp=[]
    for i in range(0,200):
        temp.append(0.0)
    vectors.append(temp)
    aslan_temp_i = 0
    # for line in open('vectors_userid.txt'):
    for line in open('../../8_21_stack_overflow_data/qu/vector_only_userid.txt'):
        if aslan_temp_i > 0 :
            items = line.strip().split(' ')
            if (len(items) < 201):
                continue
            vec = []
            for i in range(1, 201):
                vec.append(float(items[i]))
            vectors.append(vec)
        if aslan_temp_i == 0:
            aslan_temp_i = aslan_temp_i+1
    # print vectors
    return vectors
# print(len(load_userid_vectors()[10]))

#################################################################################
#################################################################################
#
# These part refer to the question, userid1, userid2 pairs.
#
#################################################################################
#################################################################################

# training part

'''
:return
    [[ 994    0  251    0 1474    0   93    0 2066    0 2327    0    4    0
   161    0    7    0   17    0    3    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0]]
'''

def load_data_question(vocab ,raw, size):
    x_train_ques = []
    for i in range(0, size):
        items = raw[i]
        x_train_ques.append(encode_sent(vocab, items[2], 50)) #150 a_b_<a> total 150_
    # return x_train_ques
    return np.array(x_train_ques)
# print(load_data_question(build_vocab(),read_raw(),1))


'''
input:
    load_data_userid1(build_userid_vocab(),read_raw(),10)
output:
    [[30], [20], [20], [0], [0], [0], [0], [0], [1803], [0]]
'''
def load_data_userid1(userid_vocab, raw, size):
    x_train_userid = []
    for i in range(0, size):
        items = raw[i]
        if items[3] in userid_vocab:
            temp_list=[]
            temp_list.append(int(userid_vocab[items[3]]))
            x_train_userid.append(temp_list)
        else:
            temp_list=[]
            temp_list.append(int(userid_vocab['UNKNOWN']))
            x_train_userid.append(temp_list)
    # return x_train_userid
    return np.array(x_train_userid)
# print(load_data_userid1(build_userid_vocab(),read_raw(),10))


'''
same
'''
def load_data_userid2(userid_vocab, raw, size):
    x_train_userid = []
    for i in range(0, size):
        items = raw[i]
        if items[4] in userid_vocab:
            temp_list=[]
            temp_list.append(int(userid_vocab[items[4]]))
            x_train_userid.append(temp_list)
        else:
            temp_list=[]
            temp_list.append(int(userid_vocab['UNKNOWN']))
            x_train_userid.append(temp_list)
    # return x_train_userid
    return np.array(x_train_userid)

# print(load_data_userid2(build_userid_vocab(),read_raw(),100))

def load_data_userid_test(userid_vocab,alist, size):
    x_neg_userid = []
    all_userid = np.array(alist)
    choice_random_user = np.random.choice(all_userid,size)
    for i in range(0,size):
        if choice_random_user[i] in userid_vocab:
            temp_list=[]
            temp_list.append(int(userid_vocab[choice_random_user[i]]))
            x_neg_userid.append(temp_list)
        else:
            temp_list=[]
            temp_list.append(int(userid_vocab['UNKNOWN']))
            x_neg_userid.append(temp_list)

    return np.array(x_neg_userid)