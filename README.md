# SCODE

SCODE : an efficient regulatory network inference algorithm from single-cell RNA-Seq during differentiation.

## Reference

In submission.

## Requirements

SCODE is written with R, and use MASS library to calculate pseudo inverse matrix.

## Download

```
git clone https://github.com/hmatsu1226/SCODE
cd SCODE
```
Or download from "Download ZIP" button and unzip it.

## Running SCODE
Optimize linear ODE and infer regulatory network from time course data.

##### Usage
```
Rscript SCODE.R <Input_file1> <Input_file2> <Output_dir> <G> <D> <C> <I>
```

* Input_file1 : G x C matrix of expression data
* Input_file2 : Time point data (e.g. pseudo-time data)
* Output_dir : Result files are outputted in this directory
* G : The number of transcription factors
* D : The number of z
* C : The number of cells
* I : The number of iterations of optimization

##### Example of running SCODE
```
Rscript SCODE.R data/exp_train.txt data/time_train.txt out 100 4 356 100
```

##### Format of Input_file1
The Input_file1 is the G x C matrix of expression data (separated with 'TAB').
Each row corresponds to each gene, and each column corresponds to each cell.

##### Example of Input_file1
```
1.24	1.21	1.28	...
0.0 	0.19	0.0	...
.
.
.
```

##### Format of Input_file2
The Input_file2 contains the time point data (pseudo-time) of each cell.

* Col1 : Information of a cell (e.g. index of a cell, experimental time point)
* Col2 : Time parameter (e.g. pseudo-time) (normalized from 0.0 to 1.0)

##### Example of Input_file2
```
0	0.065
0	0.037
0	0.007
.
.
.
72	0.873
72	0.964
```

## Output files of SCODE
SCODE outputs some files as below, and the files are named to correspond with the names of the variables in the paper.

#### A.txt
G x G matrix, which corresponds to infered regulatory network.
Aij represents the regulatory relationship from TF j to TF i.

#### B.txt
D x D diagonal matrix, which corresponds to the optimized parameters of ODE of z.

#### W.txt
G x D matrix, which corresponds to W of linear regression.

#### RSS.txt
The residual sum of squares of linear regression.

<br>
<br>
# Downstream analysis

## Calculation of RSS (RSS.R)
To choose appropriate size of z, we recommend to calculate RSS of independent test data.

#### Usage
```
Rscript RSS.R <Input_file1> <Input_file2> <Input_dir> <Output_file> <G> <D> <C>
```
* Input_file1 : G x C matrix of expression data
* Input_file2 : Time point data (e.g. pseudo-time data)
* Input_dir : The directory that W.txt and B.txt are saved (Output_dir of SCODE)
* Output_file : RSS for this data
* G : The number of transcription factors
* D : The number of z
* C : The number of cells

#### Example of running RSS.R
```
Rscript RSS.R data/exp_test.txt data/time_test.txt out out/RSS_test.txt 100 4 100
```

## Reconstruction of expression dynamics (Reconstruct_dynamics.R)
Calculate the dynamics from optimized linear ODE.

#### Usage
```
Rscript Reconstruct_dynamics.R <Input_file1> <Input_file2> <Output_file> <G>
```
* Input_file1 : Initial value of x
* Input_file2 : A.txt
* Output_file : (G+1) x 101 matrix of reconstructed expression data
* G : The number of transcription factors

#### Example of running Reconstruct_dynamics.R
```
Rscript Reconstruct_dynamics.R data/init.txt out/A.txt out/dynamics.txt 100
```

##### Format of Input_file1
The Input_file1 is the initial values of x (separated with 'TAB').
Each row corresponds to each gene.
* Col1 : Index of a gene
* Col2 : Initial value

##### Example of Input_file1
```
0	1.253
1	1.266
2	1.548
.
.
.
```

##### Format of Output_file
The Output_file is the (G+1) x 101 matrix of reconstructed expression dynamics (separated with 'TAB').
The first column corresponds to time parameter (from 0.0 to 1.0 with 0.01 interval).
Each row corresponds to each gene, and each column corresponds to each time point.

##### Example of Output_file
```
0	0.01	0.02	...
1.253	1.241	1.233	...
1.266 	1.053	0.937	...
.
.
```

## License
Copyright (c) 2016 Hirotaka Matsumoto
Released under the MIT license