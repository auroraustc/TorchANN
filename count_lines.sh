for i in ./python/no_mpi/*.py ./python/no_mpi/*.c* ./c/*.[c,h]* ./c/train_nompi/*.[c,h]*; do echo $i; done | xargs wc -l
