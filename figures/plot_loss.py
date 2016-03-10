#!/usr/bin/env python
# coding: utf-8
# @author: Alexandr Kalinin <akalinin@umich.edu>

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_results_file(filename):
	# parse network output
	results_file = pd.read_csv(filename)
	first_epoch_idx = results_file.iloc[:,0].str.contains('epoch').argmax()
	results = results_file.iloc[first_epoch_idx:, 0]
	mistakes = results[results.str.contains('Mistakes')]
	train_mistakes = mistakes[results.str.contains('Train')]
	test_mistakes = mistakes[results.str.contains('Validation')]
	test_reps = len(test_mistakes) / len(train_mistakes)
	print test_reps
	train_err = train_mistakes.str.extract('(Mistakes:\s?[0-9,.]*)').str.extract('(\d+\.?\d*)')
	test_err = test_mistakes.str.extract('(Mistakes:\s?[0-9,.]*)').str.extract('(\d+\.?\d*)')
	train_err = train_err.astype(float)
	test_err = test_err.astype(float)
	it = iter(test_err)
	test_err = [sum([next(it) for _ in xrange(test_reps)]) / test_reps for _ in range(len(test_err) / test_reps)]
	train_err = train_err.reset_index(drop=True)
	err = pd.DataFrame([train_err, pd.Series(test_err)]).transpose()
	err.columns = ['train', 'test']
	plot_loss(err)

def plot_loss(err):
	#plot train and test loss
	err.plot()
	plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotting training and testing loss for SparseConvNet output')
    parser.add_argument('filename', metavar='F', nargs=1, help='filename')
    args = parser.parse_args()
    parse_results_file(args.filename[0])