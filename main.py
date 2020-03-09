import numpy as np
from scipy import sparse
import time
from matplotlib import pyplot as plt


# Uploading Data
i, j, k = np.loadtxt('data/transition.txt', delimiter = ' ', unpack = True, dtype = 'int')
r,c =np.loadtxt('data/doc_topics.txt', delimiter = ' ', unpack = True, dtype = 'int')



# transition matrix transfer
def Matrix_Trans(k, i, j):
    
    mat_size=max(max(i),max(j))

    # make sparse matrix
    trans_csr_mtx = sparse.csr_matrix((k, (i-1, j-1)), shape=(mat_size, mat_size), dtype=np.float)

    # normalize by row
    rsums = np.array(trans_csr_mtx.sum(1))[:,0]
    ri, ci = trans_csr_mtx.nonzero()
    trans_csr_mtx.data /= rsums[ri]
    
    # transpose
    trans_mtx = trans_csr_mtx.T

    print ("::Transition matrix transfer finished.")
    
    return trans_mtx


def GPR(trans_mtx, alpha, maxerr):
    
    start = time.time() 
    (n,n)=trans_mtx.shape

    # get the p0 matrix
    p0 = np.asmatrix(np.divide(np.ones(n), n)).T

    # initialize the pagerank vector r
    ro, r = np.asmatrix(np.zeros(n)).T, np.asmatrix(np.ones(n)/n).T

    # Compute pagerank r until we converge
    while np.sum(np.abs(r-ro)) > maxerr :
        ro = r.copy()
        
        # If theres no any link, put 1/n 
        rsums = np.array(trans_mtx.sum(0))[0]

        link_0_sum=np.sum(r[np.where(rsums==0)[0]])
        
        r=(1 - alpha)*trans_mtx * r + (1 - alpha)*(1/n)*link_0_sum + alpha*p0
  
        # check ranksum
        #print(np.sum(r))
        
    print("time :", time.time() - start) 
    print(":: PageRank calculation finished.")
    print(":: Final converged GPR values \n", r)

    
    # result save
    f = open('GPR.txt', 'w')
    doc_id = 0
    for ele in r:
        doc_id += 1
        f.write(str(doc_id) + " " + str(ele[0,0]) + '\n')
        
    return r


trans_mtx=Matrix_Trans(k,i,j)
gpr=GPR(trans_mtx, 0.2, 1e-8)


plt.plot(gpr)
plt.xlabel('query')
plt.ylabel('probability')


# topic-specific teleportation vector
def TSTV(r, c):
    
    # make topic sparse matrix
    p=np.ones(r.shape)
    topic_csr_mtx = sparse.csr_matrix((p, (r-1, c-1)), shape=(max(r), max(c)), dtype=np.float)
    
    # nomalization
    rsums = np.array(topic_csr_mtx.sum(0)).reshape(-1)
    ri, ci = topic_csr_mtx.nonzero()
    topic_csr_mtx.data /= rsums[ci]
    
    assert np.sum(topic_csr_mtx)==max(c)

    print ("::Topic Sensitive vector transfer finished.")
    
    return topic_csr_mtx



def TSPR(trans_mtx,topic_mtx, alpha, beta, gamma, maxerr):
    
    (n,n)=trans_mtx.shape
    
    # get the p0 matrix
    p0 = np.asmatrix(np.divide(np.ones(n), n)).T
    
    tspr_vec=[]
    for i in range(0, 12):
        # initialize the pagerank vector r
        ro, r = np.asmatrix(np.zeros(n)).T, np.asmatrix(np.ones(n)/n).T
        
        # Compute pagerank r until we converge
        while np.sum(np.abs(r-ro)) > maxerr :
            ro = r.copy()
            
            rsums = np.array(trans_mtx.sum(0))[0]
            link_0_sum=np.sum(r[np.where(rsums==0.0)[0]])
       
            r= alpha * (trans_mtx * r + (1/n) * link_0_sum) + beta * topic_mtx[:,i] + gamma * p0
            
            # check ranksum
            #print(np.sum(r))

        tspr_vec.append(r)
        
    print(":: Topic Sensitive PageRank Matrix Generated ")
    
    return tspr_vec


topic_mtx = TSTV(r, c)



start = time.time() 
tspr=TSPR(trans_mtx,topic_mtx, 0.8, 0.1,0,1e-8)
print("time :", time.time() - start) 


# Query based Matrix 만들기
def QTD():
    
    QTD = open('data/query-topic-distro.txt', 'r')
    qtd_p = np.zeros((38, 12))
    
    for k, line in enumerate(QTD):
        qtd_line = line.split(' ')
        for i in range(2,14):
            qtd_p[k][i-2]=qtd_line[i].split(':')[1]
    
    print(":: Query-Topic Distro Imported")
    return qtd_p

def UTD():
    
    UTD = open('data/user-topic-distro.txt', 'r')
    utd_p = np.zeros((38, 12))
    
    for k, line in enumerate(UTD):
        utd_line = line.split(' ')
        for i in range(2,14):
            utd_p[k][i-2]=utd_line[i].split(':')[1]
            
    print(":: User-Topic Distro Imported")

    return utd_p


# QTSPR

def QTSPR(tspr, qtd):
    
    start = time.time() 
    tspr=np.asarray(tspr)
    r = len(tspr[0])
    c = len(tspr)
    cur_prob = np.zeros((r, c))

    # i2 81433개 중에 하나
    for i in range(0, r):
        for i2 in range(0,c):
            cur_prob[i][i2] = qtd[0][i2] * tspr[i2][i][0]
            
    qtspr=cur_prob.sum(axis=1)  
    print("Final converged QTSPR values of user2 on query1 \n",qtspr)
    
    print("time :", time.time() - start) 
    
    
    f = open('QTSPR-U2Q1.txt', 'w')
    
    doc_id = 0
    for ele in qtspr:
        doc_id += 1
        f.write(str(doc_id) + " " + str(ele) + '\n')
    
    print(":: Save Data")
    return qtspr

#UTSPR

def PTSPR(tspr, utd):
    
    start = time.time() 
    tspr=np.asarray(tspr)
    r = len(tspr[0])
    c = len(tspr)
    cur_prob = np.zeros((r, c))

    # i2 81433개 중에 하나
    for i in range(0, r):
        for i2 in range(0,c):
            cur_prob[i][i2] = utd[12][i2] * tspr[i2][i][0]
            
    utspr=cur_prob.sum(axis=1)  
    
    print(" \n Final converged UTSPR values of user2 on query1\n",utspr)
    print("time :", time.time() - start) 
    
    
    f = open('PTSPR-U2Q1.txt', 'w')
    
    doc_id = 0
    for ele in utspr:
        doc_id += 1
        f.write(str(doc_id) + " " + str(ele) + '\n')
    
    print(":: Save Data")
    return utspr


qtd=QTD()
utd=UTD()



qtspr=QTSPR(tspr, qtd)
ptspr=PTSPR(tspr, utd)



plt.plot(qtspr)
plt.xlabel('query')
plt.ylabel('probability')



plt.plot(ptspr)
plt.xlabel('query')
plt.ylabel('probability')






