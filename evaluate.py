from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylim(-0.5,3.5)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def get_num_correct(preds, labels):
	"""
	get the number of predictions (dimension nxk) where the maximum index over k matches the labels (dimension n)
	"""
	return preds.argmax(dim=1).eq(labels).sum().item()
	
@torch.no_grad()
def get_all_preds_and_labels(model, loader):
	"""
	get all predictions and real labels
	"""
	all_preds = torch.tensor([])
	all_labels = torch.tensor([], dtype=torch.long)
	for batch in loader:
		images, labels = batch

		preds = model(images)
		all_preds = torch.cat(
			(all_preds, preds)
			,dim=0
		)
		all_labels = torch.cat(
			(all_labels, labels)
			,dim=0
		)
	return all_preds, all_labels
	
def get_accuracy(preds, labels):
	preds_correct = get_num_correct(preds, labels)
	return preds_correct/len(labels)
	
def analyze_accuracy(model, data_set, label_names):
	"""
	Print the accuracy and return a plot of the confusion matrix
	"""
	train_preds, train_labels = get_all_preds_and_labels(model, torch.utils.data.DataLoader(data_set))
	print("Accuracy:", get_accuracy(train_preds, train_labels))	 
	cm = confusion_matrix(train_labels, train_preds.argmax(dim=1))
	plot_confusion_matrix(cm, label_names)
