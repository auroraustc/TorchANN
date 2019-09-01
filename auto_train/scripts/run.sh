export PYTHONPATH=$PYTHONPATH:/home/aurora/Softwares/lammps-7Aug19/lib/message/cslib/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aurora/Softwares/lammps-7Aug19/lib/message/cslib/src
~/Softwares/lammps-7Aug19/install/bin/lmp_intel_cpu_intelmpi -v mode file <in.client.W &
python TorchANN_wrap.py file 
wait
