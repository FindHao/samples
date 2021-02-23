# Samples

## collect running informations

cd bfs/
path/to/runnsys.sh ./bfs ../data/graph1MW_6.txt
cd ../results/
../selectnsys.py ../bfs/reports/

## get hot kernels

nsys profile  --stats=true ./vectorAdd
