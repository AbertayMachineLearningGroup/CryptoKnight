'''
	CryptoKnight
	@gregorydhill
'''

from __future__ import print_function
import argparse, csv, torch, sys, signal, os, random, itertools, threading, time, subprocess
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from plot import Plot

BASE = os.path.dirname(os.path.realpath(__file__))
HEAD = os.path.join(BASE, "../../data")
CONF = os.path.join(HEAD, "config")
FLAGS = os.path.join(CONF, "flags")
LABELS = os.path.join(CONF, "labels")
POOL = os.path.join(CONF, "pool")

primitives = {}
with open(POOL, 'rb') as samples:
	cores = csv.reader(samples)
	for core in cores:
		primitives[core[0]] = core[2]

# dimension / embeddings, total number of conv layers, filter widths, channels, topmost conv layer, fully-connected
hyperparams = [16, 20, [99, 9, 358], [1, 2, 5], 800, 200]

parser = argparse.ArgumentParser(description='PyTorch Dynamic Convolutional Neural Network')
parser.add_argument('--predict', type=str, metavar='sample',
                    help='variant to classify')
parser.add_argument('--evaluate', type=str, metavar='set',
                    help='distribution for confusion matrix')
parser.add_argument('--train', type=str, metavar='set',
                    help='location of training set')
parser.add_argument('--test', type=str, metavar='set',
                    help='location of testing set')
parser.add_argument('--tune', dest='tune', type=int, nargs='?', const=10, default=None, \
						metavar='N', help='number of epochs for tuning (default: 10)', required=False)
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

classes = []
with open(LABELS, 'rb') as pool:
	reader = csv.reader(pool)
	for row in reader:
		classes.append(row)

curve = Plot()
curve.create('Loss Curve', '#Epoch', 'Loss', 'loss')
curve.create('Accuracy Curve', '#Epoch', 'Accuracy', 'accuracy')

# create feature set from distribution
class Features():
	def __init__(self, data_set):
		self.features = []
		self.labels = []
		self.m = 0
		with open(data_set) as f:
			for line in f:
				matrix = []
				sentence = line.split(";")
				self.labels.append(int(sentence[0]))
				words = sentence[1].split(":")
				for word in words:
					embeddings = word.split(",")
					matrix.append([ int(i) for i in embeddings])
				self.features.append(matrix)
		z = zip(self.features, self.labels)
		random.shuffle(z)
		self.features, self.labels = zip(*z)
		self.length = len(self.features)

	# can't batch because of arbitrary sizes
	def next(self):
		x_plane = torch.Tensor([[self.features[self.m]]])
		y_onehot = torch.LongTensor([self.labels[self.m]])
		if self.m == (self.size()-1):
			self.m = 0
		else:
			self.m += 1
		return (x_plane, y_onehot)

	def size(self):
		return self.length

# construct Tensor from file
def buildCustomSample(sample):
	features = []
	with open(sample) as f:
		for line in f:
			vector = []
			sentence = line.split(";")
			words = sentence[0].split(":")
			for word in words:
				embeddings = word.split(",")
				vector.append([ int(i) for i in embeddings])
			features.append(vector)
	return torch.Tensor([[features[0]]])

# return cuda variable if on
def createVar(x):
	if args.cuda:
		return Variable(x).cuda()
	else:
		return Variable(x)

# select specified topk activations from each row
def k_max_pool(x, k):
	vectors = len(x)
	if args.cuda: a = torch.cuda.FloatTensor()
	else: a = torch.Tensor()
	for m in range(0, vectors):
		if args.cuda: b = torch.cuda.FloatTensor()
		else: b = torch.Tensor()
		channels = len(x[m])
		for n in range(0, channels):
			if args.cuda: c = torch.cuda.FloatTensor()
			else: c = torch.Tensor()
			rows = len(x[m][n][0])
			cols = len(x[m][n])
			for o in range(0, rows):
				if args.cuda: row = torch.cuda.LongTensor([o])
				else: row = torch.LongTensor([o])
				r = torch.index_select(x.data, 3, row)
				y = torch.topk(r, k, 2, sorted=False)
				c = torch.cat((c, y[0]), 3)
			b = torch.cat((b, c), 2)
		a = torch.cat((a, b), 1)
	return createVar(c)

# fold matrix by d/2
def fold(x):
	vectors = len(x)
	if args.cuda: a = torch.cuda.FloatTensor()
	else: a = torch.Tensor()
	for m in range(0, vectors):
		if args.cuda: b = torch.cuda.FloatTensor()
		else: b = torch.Tensor()
		channels = len(x[m])
		for n in range(0, channels):
			if args.cuda: c = torch.cuda.FloatTensor()
			else: c = torch.Tensor()
			rows = len(x[m][n][0])
			cols = len(x[m][n])
			for o in range(0, rows, 2):
				if args.cuda:
					row1 = torch.cuda.LongTensor([o])
					row2 = torch.cuda.LongTensor([o+1])
				else:
					row1 = torch.LongTensor([o])
					row2 = torch.LongTensor([o+1])
				r1 = torch.index_select(x.data, 3, row1)
				r2 = torch.index_select(x.data, 3, row2)	# can't fold uneven
				d = torch.add(r1, r2)
				c = torch.cat((c, d), 3)
			b = torch.cat((b, c), 2)
		a = torch.cat((a, b), 1)
	return createVar(a)

class Net(nn.Module):
	""" 
	Dynamic Convolutional Neural Network
	Based on research by Kalchbrenner et al.
	http://phd.nal.co/papers/Kalchbrenner_DCNN_ACL14
	"""

	def __init__(self):
		self.d = hyperparams[0]			# dimension / embeddings
		self.L = hyperparams[1]			# total number of conv layers
		self.m = hyperparams[2]			# filter widths
		self.c = hyperparams[3]			# channels
		self.k_top = hyperparams[4]		# topmost conv layer
		self.fc1 = self.k_top*2 		# fully-connected
		self.fc2 = hyperparams[5]

		super(Net, self).__init__()
		self.conv0 	= nn.Conv2d(self.c[0], self.c[1], kernel_size=(self.m[0],1), stride=(1,1), padding=(self.m[0]-1,0), bias=True)
		self.conv1 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv2 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv3 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv4 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv5 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv6 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv7 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv8 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv9 	= nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv10 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv11 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv12 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv13 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv14 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv15 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv16 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv17 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv18 = nn.Conv2d(self.c[1], self.c[1], kernel_size=(self.m[1],1), stride=(1,1), padding=(self.m[1]-1,0), bias=True)
		self.conv19 = nn.Conv2d(self.c[1], self.c[2], kernel_size=(self.m[2],1), stride=(1,1), padding=(self.m[2]-1,0), bias=True)
		
		self.k_max_pool = k_max_pool
		self.fold = fold
		
		self.prelu1 = nn.PReLU()
		self.prelu2 = nn.PReLU()
		self.drop = nn.Dropout(p=0.5)
		self.fc = nn.Linear(self.fc1, self.fc2)

	def forward(self, x):
		s = len(x[0][0])	# projected sentence length

		# wide convolution, fold, dynamic k-max pool
		k = max(self.k_top, int((self.L)/float(self.L)*s))
		x = self.prelu1(self.k_max_pool(self.fold(self.conv0(x)), k))
		k = max(self.k_top, int((self.L-1)/float(self.L)*s))
		x = self.prelu2(self.k_max_pool(self.conv1(x), k))
		k = max(self.k_top, int((self.L-2)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv2(x), k))
		k = max(self.k_top, int((self.L-3)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv3(x), k))
		k = max(self.k_top, int((self.L-4)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv4(x), k))
		k = max(self.k_top, int((self.L-5)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv5(x), k))
		k = max(self.k_top, int((self.L-6)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv6(x), k))
		k = max(self.k_top, int((self.L-7)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv7(x), k))
		k = max(self.k_top, int((self.L-8)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv8(x), k))
		k = max(self.k_top, int((self.L-9)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv9(x), k))
		k = max(self.k_top, int((self.L-10)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv10(x), k))
		k = max(self.k_top, int((self.L-11)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv11(x), k))
		k = max(self.k_top, int((self.L-12)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv12(x), k))
		k = max(self.k_top, int((self.L-13)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv13(x), k))
		k = max(self.k_top, int((self.L-14)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv14(x), k))
		k = max(self.k_top, int((self.L-15)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv15(x), k))
		k = max(self.k_top, int((self.L-16)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv16(x), k))
		k = max(self.k_top, int((self.L-17)/float(self.L)*s))
		x = F.relu(self.k_max_pool(self.conv17(x), k))

		k = max(self.k_top, int((self.L-18)/float(self.L)*s))
		x = F.elu(self.k_max_pool(self.fold(self.conv18(x)), k))

		x = F.elu(self.k_max_pool(self.fold(self.conv19(x)), self.k_top))

		# fully connected layer
		x = x.view(-1, self.fc1)
		x = self.drop(self.fc(x))
		x = x.view(-1, self.c[2]*self.fc2)
		return F.log_softmax(x, dim=len(x))

def train(model, optimizer, epoch, distribution):
	model.train()
	for batch in range(0, distribution.size()):
		x_train, y_train = distribution.next()
		data, target = createVar(x_train), createVar(y_train)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()

def test(model, epoch, distribution, verbose = True):
	model.eval()
	test_loss = 0
	correct = 0
	answers = []
	for batch in range(0, distribution.size()):
		x_test, y_test = distribution.next()
		data, target = createVar(x_test), createVar(y_test)
		output = model(data)
		test_loss += F.nll_loss(output, target).item()
		pred = output.data.max(1)[1]
		answers.append([int(pred.item()), int(y_test.item())])
		correct += pred.eq(target.data.item()).cpu().sum().item()

	test_loss /= distribution.size()
	if verbose:
		sys.stdout.write('\r[+] Epoch: {}, Average loss: {:.4f}, '
			'Accuracy: {}/{} ({:.0f}%)'.format(epoch, test_loss, correct, distribution.size(),
		100. * correct / distribution.size())) 			
		sys.stdout.flush()

	test_acc = 100. * correct / distribution.size()
	curve.add_points(epoch, test_loss, 'loss')
	curve.add_points(epoch, test_acc, 'accuracy')
	return (answers, test_loss, correct)

def run(model, epochs, train_set, test_set, verbose = True):
	optimizer = optim.Adagrad(model.parameters(), lr=0.3, lr_decay=1e-3)
	for epoch in range(1, epochs + 1):
		train(model, optimizer, epoch, train_set)
		results = test(model, epoch, test_set, verbose)
	return results

# load saved parameters
def load_model():
	model = Net()
	if args.cuda: model.cuda()
	m = open(os.path.join(BASE, 'data/conv.data'), 'r')
	model.load_state_dict(torch.load(m))
	model.eval()
	return model

# evaluate model on sample
def predict():
	model = load_model()
	x_val = buildCustomSample(args.predict)
	data = createVar(x_val)
	output = model(data)
	pred = output.data.max(1)[1]
	classification = int(pred.item())
	for idx, item in enumerate(classes):
		if (classification==idx):
			if len(item) > 1: print("\n[!] Multiple primitives detected!")
			print("\n[+] Classification:")
			for crypto in item:
				print("\t[*] " + primitives[crypto])

# evaluate model on distribution
def evaluate():
	model = load_model()
	matrix = [[0]*len(classes) for i in range(len(classes))]
	width = int(subprocess.check_output(['stty', 'size']).split()[1])
	center = (((width/2)-len(classes)-9) * " ") 
	try:
		# build confusion matrix
		confusion_set = Features(args.evaluate)
		size = confusion_set.size()
		answers, loss, correct = test(model, 1, confusion_set, False)
		for result in answers: matrix[result[0]][result[1]]+=1
		tpos, fpos, fneg = 0, 0, 0
		for i in range(len(matrix)):
			tpos+=matrix[i][i]
			fpos+=sum(matrix[:][i])-matrix[i][i]
			fneg+=sum(matrix[i][:])-matrix[i][i]
		tpos/=float((len(matrix)))
		fpos/=float((len(matrix)))
		fneg/=float((len(matrix)))
		precision=tpos/(tpos+fpos)
		recall=tpos/(tpos+fneg)
		f1=2*((precision*recall)/(precision+recall))
		print("[+] n = " + str(len(answers)) + "\n")
		for x, y in enumerate(classes): print("[*] "+str(x)+" = "+' & '.join(y))
		sys.stdout.write("\n" + center + '   ')
		for x in range(len(classes)): print("{: >3}".format(x), end=' ')
		sys.stdout.write('\n')
		for y in range(len(matrix)):
			sys.stdout.write('\n' + center + str(y) + "  ")
			for x in range(len(matrix)):
				print("{: >3}".format(matrix[x][y]), end=' ')
		print("\n\n[!] Accuracy: " + str(correct) + "/" + str(len(answers)) + " (" + str(int(correct / float(len(answers)) * 100)) + "%)")
		print("[+] F1: " + "{:.2f}".format(f1) + "\n")
	except ValueError:
		print("\n[!] Invalid distribution.")

# hyper-parameter optimisation
def tune(train, test):

	# conv layers; filter widths, topmost conv, fully-connected

	# [16, 54, [20, 3, 14], [1, 2, 4], 500, 40]
	hp = [[52,54], [20], [3], [14], [500,550,600,700], [5,13,26,30,40,50]]

	combinations = list(itertools.product(*hp))
	random.shuffle(combinations)
	print ("[+] Testing " + str(len(combinations)) + " combinations.\n")

	highest = 0
	params = ""
	found = False

	def tuning_exit():
		sys.stdout.write('\r[!] Exiting...\n')
		if highest != 0: print("\n[!] Best Performance: " + params)
		sys.exit(0)

	def animate():
		for c in itertools.cycle(['|', '/', '-', '\\']):
			if found: break
			else:
				sys.stdout.write('\r[*] Searching ' + c)
				sys.stdout.flush()
				time.sleep(0.2)
	try:
		t = threading.Thread(target=animate)
		t.daemon = True
		t.start()

		for combination in combinations:
			hyperparams[1] = combination[0]
			hyperparams[2][0] = combination[1]
			hyperparams[2][1] = combination[2]
			hyperparams[2][2] = combination[3]
			hyperparams[4] = combination[4]
			hyperparams[5] = combination[5]
			torch.manual_seed(args.seed)
			if args.cuda: torch.cuda.manual_seed(args.seed)
			model = Net()
			if args.cuda: model.cuda()
			answers, loss, correct = run(model, args.tune, train, test, False)
			if correct > highest:
				found = True
				highest = correct
				params = ', '.join(str(x) for x in hyperparams)
				print("\r[+] Hyper-parameters: " + ', '.join(str(x) for x in hyperparams))
				print("[+] Accuracy: " + str(correct) + "/" + str(test.size()) + ", Loss: " + str(loss) + "\n")
				found = False
		found = True
		print("\n[!] Best Performance: " + params)
	except (KeyboardInterrupt, SystemExit):
		tuning_exit()

def main(argv):
	if args.predict: predict()
	elif args.evaluate: evaluate()
	else:
		train_set = Features(os.path.join(HEAD, args.train))
		test_set = Features(os.path.join(HEAD, args.test))
		if args.tune: tune(train_set, test_set)
		else:
			torch.manual_seed(args.seed)
			if args.cuda: torch.cuda.manual_seed(args.seed)
			model = Net()
			if args.cuda: model.cuda()

			def training_exit(signal, frame):
				print('\n\n[!] Saving model state and exiting.')
				f = open('data/conv.data', 'w')
				torch.save(model.state_dict(), f)
				sys.exit(0)

			signal.signal(signal.SIGINT, training_exit)
			print("[+] Labels: " + str(len(classes)))

			# train model
			test(model, 0, test_set, True)
			run(model, args.epochs, train_set, test_set, True)
			f = open('data/conv.data', 'w')
			torch.save(model.state_dict(), f)
			print("\n\n[!] Finished!")

			curve.points('training', 'loss')
			curve.draw('training', 'loss')
			curve.points('training', 'accuracy')
			curve.draw('training', 'accuracy')
			curve.show()

if __name__ == "__main__":
	main(sys.argv)
