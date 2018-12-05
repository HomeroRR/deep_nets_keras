import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

	

def plot_losses(f_path, pp, desc):
	train_loss, train_acc = None, None
	train_loss, train_acc = None, None
	train_loss, train_acc = None, None
	with open(f_path, 'r') as json_log:
		dic = json.loads(json_log.read())
		print(dic)
		fig = plt.figure()
		plt.plot([0],[dic["train"]["loss"]], 'o',color="red", label="Train Loss")
		plt.plot([0],[dic["train"]["accuracy"]], 'o', color="black", label="Train Accuracy")
		plt.plot([0],[dic["eval"]["loss"]], 'o', color="blue", label="Eval Loss")
		plt.plot([0],[dic["eval"]["accuracy"]], 'o', color="pink", label="Eval Accuracy")
		plt.plot([0],[dic["test"]["loss"]], 'o', color="cyan", label="Test Loss")
		plt.plot([0],[dic["test"]["accuracy"]], 'o', color="green", label="Test Accuracy")
		plt.xlabel('final epoch')
		plt.ylabel('value')
		plt.title("Train, eval, test final loss and accuracy for "+desc)
		plt.legend()
		plt.show()
		pp.savefig(fig)

if __name__ == '__main__':
	f_path = '../results/ImageNet/resnet50/learningrate0.01_batchsize128/models/log.txt'
	if len(sys.argv)>1:
		f_path = str(sys.argv[1])
	p_pieces = f_path.split("/")
	dataset, model_type, params,_,_ = p_pieces[-5:]
	pp = PdfPages('loss.pdf')
	plot_losses(f_path, pp,dataset + " "+ model_type)
	pp.close()