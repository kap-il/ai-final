
##Team and ID:
Kapil Rajpopat (ktr43) & [INSERT NAME] (NETID)

*Faces*
## 2: Algorithm Desc:

- Part 1(a): Binary Perceptron

Model and architecture:
A single binary perceptron with on weight input per pixel - Weights
are stored as a (70, 60) matrix 'w' and one scalar bias 'b'. For an input
image 'x' of shape (70, 60):

score 's' = sum(w\*x)
predict 1 (face) if s >= 0
predict 0 (not face) otherwise 

Weights are initialized to zero and the bias to zero, so the first update
is purely data-driven.

Loss / update rule: No explicit loss; the perceptron uses the mistaken-
driven update rule. For each training example (x, y): 
y\_hat = predict(x)
if y\_hat != y: 
	w <- w + (y - y\_hat) \* x
	b <- b + (y - y\_hat)
	
This is the equivalent ot: if the true label is 1 and we predicted 0, 
push 'w' toward 'x'; if the true label is 0 and we predicted 1, push 'w' 
away from 'x'. The classifier sweeps the training set 
'max\_iterations' = 100 times

Features: Pixel-level binary features. Each char in the ASCII face image
to 1 if it is + or #, else 0 (util\_faces.\_parse\_image\_file). The 
perceptron consumes the raw (70, 60) 2D grid via element-wise multiplication

2.2 Part 1(b): 3-layer NN (numpy from scratch)
Model and architecture: Fully connected feed-forward netowrk: 

input (4200) -> hidden1 (128, ReLU) -> hidden2 (64, ReLU) -> output (2, softmax)
Weights are init with He scaling

(W - N(0, sqrt(2/ fan\_in))); biases at zero. The 70x60 image is flattened to a 
4200-dim vector before entering the network.

Loss. Cross-entropy on a softmax over two output classes (face / not face).
For a batch of N examples:

L = -(1/N) sum\_i sum\_k y\_ik log(yhat\_ik)

where y\_ik is the one-hot true label and yhat\_ik is the softmax output.

Update rule:
Vanilla mini-batch gradient descent (no momentum, no weight decay) at learning rate
'lr' = 0.01

W <- W - lr dL/dW, b <- b - lr dL/db

Backpropagation derivation sketch:
With ReLU on the two hidden layers and softmax+cross-entropy at the output, the 
gradients are:

Forward:
z1 = X W1 + b1, a1 = ReLU(z1)
z2 = a1 W2 + b2, a2 = ReLU(z2)
z3 = a2 W3 + b3, yhat = softmax(z3)

Output layer (softmax + cross-entropy collapse cleanly)
dz3 = (yhat - y) / N
dW3 = a2^T @ dz3
db3 = sum(dz3, axis=0)

Hidden layer 2 (ReLU')
da2 = dz3 @ W3^T
dz2 = da2 * 1[z2 > 0]
dW2 = a1^T @ dz2
db2 = sum(dz2, axis=0)

Hidden layer 1 (ReLU')
da1 = dz2 @ W2^T
dz1 = da1 * 1[z1 > 0]
dW1 = X^T @ dz1
db1 = sum(dz1, axis=0)

The clean 'dz3' = (yhat - y)/N form is the standard simplification when composing
softmax with cross-entropy. ReLU's derivative is the indicator 1[z > 0], applied 
element-wise. Implemented in q1b\_neural\_net\_scratch\_faces.py:81-103.

Features. Same pixel-level binary features as (1a), but flattened to a 4200-dim 
vector via util\_faces.flatten\_images.

2.3 Part 1c: 3-layer NN (pytorch)
Model and architecture. Identical topology to 1b:

nn.Linear(4200, 128) -> ReLU -> nn.Linear(128, 64) -> ReLU -> nn.Linear(64, 2)

Default pytorch nn.Linear init (Kaiming uniform). The forward method returns raw logits, 
softmax is implicit inside the loss

Loss. nn.CrossEntropyLoss, which composes log-softmax and NLL internally over the 
two-class output 

Update rule:
Adam (torch.optim.Adam, lr = 1e-3) with default betas. Mini-batches of size 32, 100 
epochs

Features: 
Same pixel-level binary features, flattened to 4200 dims, packaged as torch.float32
tensors (flatten\_images then torch.tensor)

## 3 Experimental protocol (faces)
The driver q2q3\_run\_all\_stats.py runs the same protocol for every classifier and 
every training fraction:

Training fractions:
p in {10, 20, 30... 100} (every 10%)

Sampling: 
For each p, draw num\_iterations independent random subsamples of size p% of the
training set (np.random.choice(..., replace=False)).

Fresh classifier per draw: 
A new instance of the classifier is constructed and trained from scratch on each 
subsample (no warm-starting)

Eval:
Each trained classifier is evaluated on the *full test set*
(facedatatest / facedatatestlabels).

Reported statistics:
Per (classifier, p) cell we report mean\_train\_time, mean\_error, and std\_error
across the num\_iterations draws.

Default num\_iterations:
5

Hyperparameters used (faces):
Perceptron: 
n/a hidden sizes
step activation
no LR
mistake driven
100 passes
1 Batch
no explicit loss

Scratch NN:
hidden sizes: 128, 64
ReLU
lr: 0.01
optimiser: sgd
100 epochs
Batch: 32
loss: softmax +CE

pytorch NN:
hidden sizes: 128, 64
ReLU
lr: 1e-3
optimiser: adam
100 epochs
Batch: 32
loss: crossEntropyLoss

Random Seeding: No fixed seed is set in any of the face mains: 
each iterations genuinely draws a new subsample, which is what the std\_error bars
are meant to capture.

## 4 Results:
to be filled out:

## 5 Discussion:

Hypothesis space gap:
perceptron is a single linear classifier on raw pixes: two MLPS add two non-linear
hidden layers. on binary task as visually structured as faces, the perceptron is 
already surprisingly strong baseline because the classes are roughly linearly 
separable in pixel space - MLPS should match or beat with enough data

optimiser gap bw NNs:
two MLPS share an architecture but differ in optimiser and init.
Any systematic gap between 1b and 1c at the same training fraction is attributable
to these two factors, not to capacity.

variance source:
std bars conflate two independent sources of randomness: training subsample draw and
the weight init / mini-batch shuffle. At small p the subsample term dominates; at 
p = 100 only the init / shuffle term remains

Time scaling:
per-cost for both NNs is O(N) per pass but with a much smaller constant. Expect
the curves to be apprx. linear with different time slopes















