#coding:utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import random
from time import time
import os

def f(v,F):
	for u in F:
		G_bar.remove_edge(v,u)
	dists = nx.shortest_path_length(G_bar, source=v)
	val = sum(1/d for d in dists.values() if d != 0)
	for u in F:
		G_bar.add_edge(v,u)
	return val

def reverse_f(v,F):
	for u in F:
		G_bar.add_edge(v,u)
	dists = nx.shortest_path_length(G_bar, source=v)
	val = sum(1/d for d in dists.values() if d != 0)
	for u in F:
		G_bar.remove_edge(v,u)
	return val

def f_hat(v,x):
	h = []
	for u in x:
		heapq.heappush(h,(-1*x[u],u))
	s_next = heapq.heappop(h)
	val = (1 - (-1*s_next[0])) * f(v,[])
	X = [s_next[1]]
	while len(h) > 0:
		s_prev_val = -1*s_next[0]
		s_next = heapq.heappop(h)
		if s_prev_val == 0: break
		val += (s_prev_val - (-1*s_next[0])) * f(v,X)
		X.append(s_next[1])
	return val

def partial_f_hat(v,x):
	y = {}
	h = []
	for u in x:
		heapq.heappush(h,(-1*x[u],u))
	s_next = heapq.heappop(h)
	X = [s_next[1]]
	y[s_next[1]] = f(v,X) - f(v,[])
	for i in range(len(x) - 1):
		prev_f = f(v,X)
		s_next = heapq.heappop(h)
		X.append(s_next[1])	
		y[s_next[1]] = f(v,X) - prev_f
	return y

def partial_stochastic_f_hat(v,x):
	h = []
	X = []
	for u in x:
		heapq.heappush(h,(-1*x[u],u))
	index = random.randint(1,len(x))
	for i in range(index):
		s_next = heapq.heappop(h)
		X.append(s_next[1])
	return (s_next[1], len(x)*(f(v,X) - f(v,X[:-1])))

def proj(x,k,eps=1e-05): 
	proj_x = {}
	sum_ = 0
	for u in x: 
		sum_ += min(max(x[u],0),1)
	if sum_ <= k: 
		for u in x:
			proj_x[u] = min(max(x[u],0),1)
	else: 
		lambda_L = 0
		lambda_R = max(x.values())
		while lambda_R - lambda_L > eps:
			lambda_mid = (lambda_L + lambda_R) / 2
			x_modified = {}
			for u in x:
				x_modified[u] = x[u] - lambda_mid
			sum_ = 0
			for u in x:
				sum_ += min(max(x_modified[u],0),1)
			if sum_ <= k: lambda_R = lambda_mid
			else: lambda_L = lambda_mid
		for u in x:
			proj_x[u] = min(max(x[u]-lambda_R,0),1)
	return proj_x

def Algorithm_1(v,k):
	t1 = time()
	in_neighbors = []
	for u in G_bar.successors(v):
		in_neighbors.append(u)
	for u in in_neighbors:
		G_bar.remove_edge(v,u)
	h = []
	for u in in_neighbors: 
		dists = nx.shortest_path_length(G_bar, source=u)
		val = sum(1/d for d in dists.values() if d != 0)
		heapq.heappush(h,(-1*val,u))
	X = [heapq.heappop(h)[1] for i in range(k)]
	t2 = time()
	for u in in_neighbors:
		G_bar.add_edge(v,u)
	return X, t2 - t1
	
def Algorithm_2(v,k,alpha=1/3):
	t1 = time()
	x = relaxation(v,k)
	cent_vals_small = []
	sizes_small = []
	cent_vals_large = []
	sizes_large = []
	p = random.uniform(alpha,1)
	X = []
	for u in x:
		if x[u] >= p: X.append(u)
	t2 = time()
	return X, t2 - t1

def relaxation(v,k): 
	x = {}
	for u in G_bar.successors(v): 
		x[u] = 0
	x_inter = {}
	t = 1
	best_val = f_hat(v,x)
	x_best = {}
	Theta = G_bar.out_degree(v)/2
	L = f(v,[])
	while t < 1000:
		partial = partial_f_hat(v,x)
		for i in x.keys():
			x_inter[i] = x[i] - (math.sqrt(2*Theta)/(L*math.sqrt(t+1))) * partial[i]
		x_next = proj(x_inter,k)
		if f_hat(v,x_next) < best_val:
			best_val = f_hat(v,x_next)
			x_best = x_next
		t += 1
		for i in x.keys():
			x[i] = x_next[i]
	return x_best

def Random(v,k): 
	t1 = time()
	in_neighbors = []
	for u in G_bar.successors(v): 
		in_neighbors.append(u)
	X = random.sample(in_neighbors,k)
	t2 = time()
	return X, t2 - t1

def Degree(v,k):
	t1 = time()
	h = []
	for u in G_bar.successors(v): 
		heapq.heappush(h,(-1*G.in_degree(u),u))
	X = [heapq.heappop(h)[1] for i in range(k)]
	t2 = time()
	return X, t2 - t1
	
def Greedy(v,k):
	t1 = time()
	X = []
	in_neighbors = []
	for u in G_bar.successors(v):
		in_neighbors.append(u)
	while len(X) < k: 
		min_ = G_bar.number_of_nodes()
		min_u = -1
		for u in in_neighbors:
			if f(v,[u]) < min_: 
				min_u = u
				min_ = f(v,[u])
		X.append(min_u)
		in_neighbors.remove(min_u)
		G_bar.remove_edge(v,min_u)
	t2 = time()
	for u in X:
		G_bar.add_edge(v,u)
	return X, t2 - t1


random.seed(0)

G = nx.DiGraph()
G = nx.read_edgelist("./graph/out.moreno_blogs", create_using=nx.DiGraph)
G = nx.convert_node_labels_to_integers(G,ordering="sorted")

G_bar = G.reverse()

file = open("./target_id/out.moreno_blogs", 'r') 
lines = file.readlines()
line = lines[0].strip().split()
file.flush()
file.close()
targets = [int(element) for element in line] 
target = targets[0] # max-degree target
b = math.floor(G.in_degree(target) * 0.5)

sol_Algorithm_1 = Algorithm_1(target,b)
sol_Algorithm_2 = Algorithm_2(target,b)
sol_Random = Random(target,b)
sol_Degree = Degree(target,b)
sol_Greedy = Greedy(target,b)
print("Algorithm 1: f(F) =",f(target, sol_Algorithm_1[0]),"time(s):",sol_Algorithm_1[1])
print("Algorithm 2: f(F) =",f(target, sol_Algorithm_2[0]),"time(s):",sol_Algorithm_2[1])
print("Random: f(F) =",f(target, sol_Random[0]),"time(s):",sol_Random[1])
print("Degree: f(F) =",f(target, sol_Degree[0]),"time(s):",sol_Degree[1])
print("Greedy: f(F) =",f(target, sol_Greedy[0]),"time(s):",sol_Greedy[1])
