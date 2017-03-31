# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:59:24 2016

@author: othmane
"""
import numpy as np
import matplotlib.pyplot as plt

import utilities
"""
# truc pour un affichage plus convivial des matrices numpy

np.set_printoptions(precision=2, linewidth=320)
plt.close('all')
"""



X,Y = utilities.load_data("../resources/lettres.pkl")


nCl = 26

def discretise_bis(X,d):
	intervalle = 360./d
	return np.floor(X/intervalle)

def discretise(X,d):
	l=[discretise_bis(X[i],d) for i in range(len(X))]
	return np.array(l)

def initGD(X,N):
	"""
	X sont les observations
	"""
	p=list()
	for x in X:
		p.append(np.floor(np.linspace(0,N-.00000001,len(x))))

	return np.array(p)

X0 = [ 1.,  9., 8.,  8.,  8.,  8.,  8.,  9.,  3.,  4.,  5.,  6.,  6.,  6.,  7.,  7.,  8.,  9.,  0.,  0.,  0.,  1.,  1.]

p=initGD(X,4)


def learnHMM(allx, allq, N, K,initTo0=False):
	if initTo0:
		A = np.zeros((N,N))
		B = np.zeros((N,K))
		Pi = np.zeros(N)
	else:
		eps = 1e-8
		A = np.ones((N,N))*eps
		B = np.ones((N,K))*eps
		Pi = np.ones(N)*eps
	for s in allq:#Pour tout les etats
		for i in range(len(s)-1):
			A[int(s[i])][int(s[i+1])]+=1

		Pi[int(s[0])]+=1

	for q,x in zip(allq,allx):
		#q est l etat x l observation
		for t in range(len(q)):
			B[int(q[t])][int(x[t])]+=1

		#print([int(e) for e in B[0,:]])

	A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
	B = B/np.maximum(B.sum(1).reshape(N,1),1) # normalisation

	Pi = Pi/Pi.sum()

	return(Pi,A,B)

K = 10 # discrétisation (=10 observations possibles)
N = 5  # 5 états possibles (de 0 à 4 en python)
# Xd = angles observés discrétisés
Xd = discretise(X,K)
q=initGD(Xd,N)
Pi, A, B = learnHMM(Xd[Y=='a'],q[Y=='a'],N,K)
#A(i,j) : la probabilite d'aller a l'etat j en etant dans i
#B(i,j)=B_i(j) : la probabilite d'observer j en etant dans l'etat i
def viterbi(x,Pi,A,B):
	N=len(Pi)#nombre d etats
	T=len(x)#nombre d observation
	phi=np.zeros([N,T])
	phi[:,0]=np.log(Pi)+np.log(B[:,int(x[0])])
	psi=np.zeros([N,T])
	psi[:,0]=[-1 for i in range(N)]
	for t in range(1,T):#i de 1 a T-1
		for j in range(N):
			m=np.array(phi[:,t-1]+np.log(A[:,j])).max()
			phi[j,t]= m + np.log(B[j,int(x[t])])
			psi[j,t]=np.array(phi[:,t-1]+np.log(A[:,j])).argmax()

	p_est=phi[:,T-1].max()

	S_optimale=phi[:,T-1].argmax()
	s_est=[S_optimale]
	for t in range(0,T-1):
		s_est.append(psi[int(s_est[-1]),T-1-t])

	s_est.reverse()
	return s_est,p_est


s_est, p_est = viterbi(Xd[0], Pi, A, B)

"""
eps = 1e-8

A_2 = np.ones((N,N))*eps
B_2 = np.ones((N,K))*eps
Pi_2= np.ones(N)*eps

for i in range(len(s_est)-1):
	A_2[s_est[i]][s_est[i+1]]+=1

	Pi_2[s_est[0]]+=1


for t in range(len(s_est)):
	B_2[s_est[t]][Xd[0][t]]+=1


A_2 = A_2/np.maximum(A_2.sum(1).reshape(N,1),1) # normalisation
B_2 = B_2/np.maximum(B_2.sum(1).reshape(N,1),1) # normalisation

Pi_2 = Pi_2/Pi_2.sum()
"""

def evaluation_log(x,Pi,A,B):
	"""
	Etant donne une observation et un model on retourne log(P(X|alpha))
	"""
	N=len(Pi)
	T=len(x)
	omega = np.zeros(T)
	alpha=np.zeros([N,T])
	alpha[:,0]=Pi*B[:,int(x[0])]
	omega[0] = alpha[:,0].sum()
	alpha[:,0]/=omega[0]
	for t in range(1,T):
		alpha[:,t]=np.dot(alpha[:,t-1].reshape(1,N),A) * B[:,int(x[t])]
		omega[t]=alpha[:,t].sum()
		alpha[:,t]/=omega[t]


	return np.log(omega).sum()

e=evaluation_log(Xd[0],Pi,A,B)


def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:np.floor(pc*n)])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest
# exemple d'utilisation
itrain,itest = separeTrainTest(Y,0.8)

def apprentissage_complet(X,Y,itrain,N,K,max_iter,eps):
	"""
	K parametre de discretisation
	N le nombre d etats possible
	itrain l'indice pour les elements de X qu'on utilisera pour l'apprentissage

	"""
	Pi=dict()
	A=dict()
	B=dict()
	Xd = discretise(X,K)
	q=initGD(Xd,N)
	vrai_prec=1
	for i in range(len(np.unique(Y))):
		cl=np.unique(Y)[i]
		Pi[cl], A[cl], B[cl] = learnHMM(Xd[itrain[i]],q[itrain[i]],N,K)
		"""
		for obs in Xd[itrain[i]]:
			vrai_prec+=evaluation_log(obs,Pi[cl],A[cl],B[cl])
		"""


	vraisemblance=1
	iteration=0
	print("( L^k ,\t L^k+1 ,\t vraisemblance )")
	while vraisemblance > eps and iteration < max_iter:
		vrai_current=0

		for i in range(len(np.unique(Y))):
			cl=np.unique(Y)[i]
			q_estime=list()
			for obs in Xd[itrain[i]]:
				s_est,p_est=viterbi(obs, Pi[cl], A[cl], B[cl])
				vrai_current+=evaluation_log(obs,Pi[cl],A[cl],B[cl])
				#vrai_current+=p_est
				q_estime.append(s_est)

			Pi[cl], A[cl], B[cl] = learnHMM(Xd[itrain[i]],q_estime,N,K)


		print(vrai_prec,vrai_current,vraisemblance)

		vraisemblance = (vrai_prec - vrai_current)/float(vrai_prec)
		vrai_prec=vrai_current

		iteration=+1

	print(vrai_prec,vrai_current,vraisemblance)

	return Pi,A,B

N=5
K=10
models=apprentissage_complet(X,Y,itrain,N,K,10,1e-6)

"""
def test_performance(Pi,A,B,X,cl):

	print("Testing"+cl+"\n")
	for i in range(len(X)):
		e=evaluation_log(X[i],Pi_bis[cl],A_bis[cl],B_bis[cl])
		print("Probabilite que la sequence "+str(i)+" correspondante a la lettre ' "+Y[i]+" ' avec ' " +cl+" ' est de "+str(e))

test_performance(Pi_bis,A_bis,B_bis,Xd,'a')
"""

def generate_HMM(Pic,Ac,Bc,N):


	s=[list(np.where( Pic > np.random.random(), 1., 0.)).index(1)]
	print(s)
	o=[list(np.where( Bc[s[-1],:] > np.random.random(), 1., 0.)).index(1)]

	for i in range(1,N):
		etat=s[-1]
		s.append(list(np.where( Ac[etat,:] > np.random.random(), 1., 0.)).index(1))
		o.append(list(np.where( Bc[etat,:] > np.random.random(), 1., 0.)).index(1))

	return o,s

#cl='a'
#obs,seq=generate_HMM(models[0][cl],models[1][cl],models[2][cl],23)

# affichage d'une lettre (= vérification bon chargement)
def tracerLettre(let):
    a = -let*np.pi/180;
    coord = np.array([[0, 0]]);
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.plot(coord[:,0],coord[:,1])
    return

#Trois lettres générées pour 5 classes (A -> E)
n = 3          # nb d'échantillon par classe
nClred = 5   # nb de classes à considérer
fig = plt.figure()
d=K
for cl in np.unique(Y)[:nClred]:
	Pic = models[0][cl].cumsum() # calcul des sommes cumulées pour gagner du temps
	Ac = models[1][cl].cumsum(1)
	Bc = models[2][cl].cumsum(1)
	index=list(np.unique(Y)).index(cl)
	long = np.floor(np.array([len(x) for x in Xd[itrain[index]]]).mean()) # longueur de seq. à générer = moyenne des observations
	for im in range(n):
		s,x = generate_HMM(Pic, Ac, Bc, int(long))
		intervalle = 360./d  # pour passer des états => angles
		newa_continu = np.array([i*intervalle for i in x]) # conv int => double
		sfig = plt.subplot(nClred,n,im+n*index+1)
		sfig.axes.get_xaxis().set_visible(False)
		sfig.axes.get_yaxis().set_visible(False)
		tracerLettre(newa_continu)
plt.savefig("lettres_hmm.png")
