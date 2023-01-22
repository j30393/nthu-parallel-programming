#!/bin/sh
# this script is use to link the testcase to the real cases

DIR=/home/pp21/share/.testcase/hw3

for dir in $DIR/*; do
	echo $dir
	echo $(basename "${dir}")
	ln -s $dir $(basename "${dir}")
done
