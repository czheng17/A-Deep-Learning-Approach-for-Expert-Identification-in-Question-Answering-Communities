# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import torch
import model
import Dataset_handle
import torch.nn.functional
import torch.utils.data as Data
from torch.autograd import Variable
import os
import operator
import numpy as np



parser = argparse.ArgumentParser(description='Q-CNN')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
parser.add_argument('-optim', type=str, default='Adam', help='Adam or SGD')
parser.add_argument('-epochs', type=int, default=5000, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=100, help='batch size for training [default: 128]')
parser.add_argument('-log_interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test_interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save_interval', type=int, default=90000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max_norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-word_embed_dim', type=int, default=100, help='number of word embedding dimension [default: 100]')
parser.add_argument('-user_embed_dim', type=int, default=200, help='number of user embedding dimension [default: 200]')
parser.add_argument('-kernel_num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel_sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

# Load data
print("Loading data...")
cnn_word_vocab = Dataset_handle.build_vocab()
userid_vocab = Dataset_handle.build_userid_vocab()

# this is train data
raw_data = Dataset_handle.read_raw()
# print(len(raw_data))
# this is test data
test_raw_data = Dataset_handle.test_read_raw()
# print(len(test_raw_data))

#number_questions= 128000
#number_questions_val =1280
number_questions= 97000
number_questions_val =10000

# load the training data: quesion, userid1, userid2
x_train_ques=torch.from_numpy(Dataset_handle.load_data_question(cnn_word_vocab, raw_data, number_questions))
x_train_userid1= torch.from_numpy(Dataset_handle.load_data_userid1(userid_vocab,raw_data,number_questions))
x_train_userid2= torch.from_numpy(Dataset_handle.load_data_userid2(userid_vocab,raw_data,number_questions))

# load the testing data: question, userid1, userid2
test_ques = torch.from_numpy(Dataset_handle.load_data_question(cnn_word_vocab,test_raw_data, number_questions_val))
test_userid1 = torch.from_numpy(Dataset_handle.load_data_userid1(userid_vocab,test_raw_data,number_questions_val))
alist = Dataset_handle.read_neg_alist()
test_userid2 = torch.from_numpy(Dataset_handle.load_data_userid_test(userid_vocab,alist,number_questions_val))

# load the cnn word embedding
vectors_ques = Dataset_handle.load_vectors()
# load the uerid word embedding
vectors_userid = Dataset_handle.load_userid_vectors()
print("Load done...")

val_file = '../../8_21_stack_overflow_data/qu/test1_use.txt'
precision = '../../8_21_stack_overflow_data/qu/test1_use_change_margin.acc'


'''
get batch from the training data set
'''
# data_loader = Data.DataLoader(
#     dataset=x_train_ques,
#     batch_size=args.batch_size,
#     shuffle=True,
# )
def get_batch(sentence, uid1, uid2, i, evaluation=False):
    # seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(sentence[i:i+args.batch_size], volatile=evaluation)
    target1 = Variable(uid1[i:i+args.batch_size].view(-1))
    target2 = Variable(uid2[i:i+args.batch_size].view(-1))
    return data, target1, target2

args.word_embed_num = len(cnn_word_vocab)
args.user_embed_num = len(userid_vocab)
args.pretrained_word_embedding = torch.FloatTensor(vectors_ques)
args.pretrained_user_embedding = torch.FloatTensor(vectors_userid)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# print("\nParameters:")
# for attr, value in sorted(args.__dict__.items()):
#     print("\t{}={}".format(attr, value))

'''
instance the CNN model and prepare user embedding lookup function
'''
cnn = model.CNN_Text(args)

user_embed = torch.nn.Embedding(args.user_embed_num, args.user_embed_dim)
user_embed.weight = torch.nn.Parameter(args.pretrained_user_embedding)
user_embed.weight.requires_grad = False

'''
cuda and optimizer
'''
if args.cuda:
    cnn = cnn.cuda()

# parameters = filter(lambda p: p.requires_grad, cnn.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=args.lr)

cnn.train()

'''
begin to training
'''
def batch_dot_product(input1,input2,batch_size_num):
    tmp = torch.dot(input1[0],input2[0])
    for i in range(1,batch_size_num):
        res = torch.dot(input1[i],input2[i])
        tmp=torch.cat((tmp,res),0)
    return tmp

for epoch in range(args.epochs):
    for step in range(0, len(x_train_ques),args.batch_size):
        t_x, t_y1, t_y2 = get_batch(x_train_ques, x_train_userid1, x_train_userid2, step)

        if args.cuda:
                t_x, t_y1, t_y2 = t_x.cuda(), t_y1, t_y2

        sentence_output = cnn(t_x)

        user1_embedding = user_embed(t_y1)
        user2_embedding = user_embed(t_y2)
        if args.cuda:
            user1_embedding = user1_embedding.cuda()
            user2_embedding = user2_embedding.cuda()

        # this is my method for the matmal
        # matmul_o1 = batch_dot_product(sentence_output , user1_embedding,args.batch_size)
        # matmul_o2 = batch_dot_product(sentence_output , user2_embedding,args.batch_size)

        # cos simlilaity
        matmul_o1 = torch.nn.functional.cosine_similarity(sentence_output,user1_embedding,dim=1, eps=1e-6)
        matmul_o2 = torch.nn.functional.cosine_similarity(sentence_output, user2_embedding, dim=1, eps=1e-6)

        first_part = Variable(torch.zeros(args.batch_size))
        second_part = Variable(torch.linspace(0.1,0.1,args.batch_size))
        if args.cuda:
            first_part = first_part.cuda()
            second_part = second_part.cuda()
        losses = torch.max(first_part, (second_part - matmul_o1 + matmul_o2) )
        # losses = torch.max(Variable(torch.zeros(args.batch_size)), ( Variable(torch.ones(args.batch_size)) - matmul_o1 + matmul_o2) )
        loss = torch.mean(losses)
        # if i%100 ==0:
        #     print(torch.max(Variable(torch.zeros(args.batch_size)), ( Variable(torch.ones(args.batch_size)) - matmul_o1 + matmul_o2) ))

        # in the cos similarity, we need this 2 line, but in the matmul, wu do not need
        matmul_o1_1 = matmul_o1.cpu().data.numpy()
        matmul_o2_1 = matmul_o2.cpu().data.numpy()

        count = 0
        if step % 30000 ==0 and step != 0:
            for correct_num in range(len(matmul_o1_1)):
                if(matmul_o1_1[correct_num] >= matmul_o2_1[correct_num]):
                    count+=1
            print('step: '+str(step)+' loss:'+str(loss.cpu().data.numpy())+' correct number: '+str((count+0.0))+' total number :'+str(len(matmul_o1_1))+' training accuracy:'+str((count+0.0)/len(matmul_o1_1)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if step % args.save_interval == 0:
        #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        #     save_prefix = os.path.join(args.save_dir, 'snapshot')
        #     save_path = '{}_steps{}_{}.pt'.format(save_prefix, step, epoch)
        #     torch.save(cnn.state_dict(), save_path)

    '''
    if not use epoch per test, tab all the code below
    '''

    if epoch % 10 == 0 and epoch != 0 :
        # if step % 10000 == 0:
        cnn.eval()
        corrects, avg_loss = 0, 0
        scoreList = []
        for val_step in range(0, number_questions_val,args.batch_size ):
            t_x, t_y1, t_y2 = get_batch(test_ques, test_userid1, test_userid2, val_step)
            if args.cuda:
                    t_x, t_y1, t_y2 = t_x.cuda(), t_y1, t_y2

            sentence_output = cnn(t_x)

            user1_embedding = user_embed(t_y1)
            user2_embedding = user_embed(t_y2)
            if args.cuda:
                user1_embedding = user1_embedding.cuda()
                user2_embedding = user2_embedding.cuda()

            # matmul_o1 = batch_dot_product(sentence_output , user1_embedding,args.batch_size)
            # matmul_o2 = batch_dot_product(sentence_output , user2_embedding,args.batch_size)
            matmul_o1 = torch.nn.functional.cosine_similarity(sentence_output,user1_embedding,dim=1, eps=1e-6)
            matmul_o2 = torch.nn.functional.cosine_similarity(sentence_output, user2_embedding, dim=1, eps=1e-6)


            for score in matmul_o1:
                scoreList.append(score.cpu().data.numpy()[0])
                # print(scoreList)



            first_part = Variable(torch.zeros(args.batch_size))
            second_part = Variable(torch.linspace(0.1,0.1,args.batch_size))
            if args.cuda:
                first_part = first_part.cuda()
                second_part = second_part.cuda()
            losses = torch.max(first_part, (second_part - matmul_o1 + matmul_o2) )
            # losses = torch.max(Variable(torch.zeros(args.batch_size)), ( Variable(torch.ones(args.batch_size)) - matmul_o1 + matmul_o2) )
            loss = torch.mean(losses)
            avg_loss += loss

            # count = 0
            # for correct_num in range(len(matmul_o1)):
            #         if(matmul_o1[correct_num] >= matmul_o2[correct_num]):
            #             count+=1
            # corrects += count

        size = number_questions_val
        avg_loss = loss.cpu().data[0]/size
        cnn.train()
        print('\nEvaluation - loss: {:.6f}  ) \n'.format(avg_loss))
        # print(len(scoreList))

        '''
            compute the top1
        '''
        sessdict = {}
        index = int(0)
        # print("cccccccccccccccccccccccccccccc")
        for line in open(val_file):
            items = line.strip().split(' ')
            qid = items[1].split(':')[1]
            if not qid in sessdict:
                sessdict[qid] = []
            # print(sessdict)
            sessdict[qid].append((scoreList[index], items[0]))
            index += 1
            # print(index)
            if index >= number_questions_val:
                break
        # print("ddddddddddddddddddddddddddddddd")
        lev1 = float(0)
        lev0 = float(0)
        of = open(precision, 'aw')
        of.write('lr:' + str(args.lr) + 'kernel_sizes:' + str(args.kernel_sizes) + 'optim:' + str(args.optim)+ '\n')
        for k, v in sessdict.items():
            # print k,v
            v.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = v[0]
            if flag == '1':
                lev1 += 1
            if flag == '0':
                lev0 += 1
        #print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        of.write('lev1:' + str(lev1) + '\n')
        of.write('lev0:' + str(lev0) + '\n')
        of.write('top1:' + str(lev1/(lev1+lev0)) + '\n')
        print('lev1 ' + str(lev1))
        print('lev0 ' + str(lev0))
        print('top1 ' + str(lev1/(lev1+lev0)))
        of.close()