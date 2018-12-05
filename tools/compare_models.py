import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_data_filenames(data_dir='../results/ImageNet'):
	label_counter = 0
	file_paths = set()
	for subdir, dirs, files in os.walk(data_dir):
		for folder in dirs:
			for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
				for file in folder_files:
					if file =="loss_log.json":
						path_pieces = folder_subdir.split("\\")
						p_pieces = []
						for piece in path_pieces:
							p_pieces += piece.split("/")
						model_type, params, _,partition_type = p_pieces[-4:]
						file_paths.add(tuple([os.path.join(folder_subdir, file)]+p_pieces[-4:]+[file]))
	#print(len(file_paths))
	return list(file_paths)

def load_json_epochs(f_path):
	with open(f_path, 'r') as json_log:
		dic = json.loads(json_log.read())
		vals = sorted([(key, (dic[key]["loss"], dic[key]["accuracy"])) for key in dic])
		X,Y = zip(*vals)
		L,A = zip(*Y)
		return X,L,A

def plotGraph(X,L,A, pp, title):
	fig = plt.figure()
	plt.plot(X,A, color="red", label="Accuracy")
	plt.plot(X,L, color="black", label="Loss")
	plt.xlabel('epoch')
	plt.ylabel('value')
	plt.title(title)
	plt.legend()
	plt.show()
	pp.savefig(fig)
	return

if __name__ == '__main__':
	data_dir='../results/ImageNet'
	if len(sys.argv)>1:
		data_dir=str(sys.argv[1])
	file_paths = load_data_filenames(data_dir)
	pp = PdfPages('results.pdf')
	for f_paths in file_paths:
		f_path,model_type, params, _,partition_type,file = f_paths
		X,L,A = load_json_epochs(f_path)
		plotGraph(X,L,A,pp, model_type +" "+ params)
	pp.close()