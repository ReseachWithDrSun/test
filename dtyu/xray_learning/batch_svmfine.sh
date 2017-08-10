#!/bin/bash

nohup python run_fbb_svm.py --action 1 --label 0 --fine -c 1
nohup python run_fbb_svm.py --action 1 --label 1 --fine -c 100
nohup python run_fbb_svm.py --action 1 --label 2 --fine -c 0.01
nohup python run_fbb_svm.py --action 1 --label 3 --fine -c 1
nohup python run_fbb_svm.py --action 1 --label 4 --fine -c 1
nohup python run_fbb_svm.py --action 1 --label 5 --fine -c 1
nohup python run_fbb_svm.py --action 1 --label 6 --fine -c 0.0001
nohup python run_fbb_svm.py --action 1 --label 7 --fine -c 1
nohup python run_fbb_svm.py --action 1 --label 8 --fine -c 1
nohup python run_fbb_svm.py --action 1 --label 9 --fine -c 1

