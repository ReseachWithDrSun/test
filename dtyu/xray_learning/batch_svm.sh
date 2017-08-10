#!/bin/bash

nohup python run_fbb_svm.py --action 1 --label 0 --fixed -c 2
nohup python run_fbb_svm.py --action 1 --label 1 --fixed -c 12.5
nohup python run_fbb_svm.py --action 1 --label 2 --fixed -c 0.08
nohup python run_fbb_svm.py --action 1 --label 3 --fixed -c 0.5
nohup python run_fbb_svm.py --action 1 --label 4 --fixed -c 0.5
nohup python run_fbb_svm.py --action 1 --label 5 --fixed -c 2
nohup python run_fbb_svm.py --action 1 --label 6 --fixed -c 0.0002
nohup python run_fbb_svm.py --action 1 --label 7 --fixed -c 0.125
nohup python run_fbb_svm.py --action 1 --label 8 --fixed -c 0.25
nohup python run_fbb_svm.py --action 1 --label 9 --fixed -c 4

