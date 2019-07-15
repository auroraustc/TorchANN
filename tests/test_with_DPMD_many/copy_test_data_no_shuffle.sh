export OMP_NUM_THREADS=1

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))
}

if [ ! -f "dataset_loc" ];then
echo "Error: File **dataset_loc** which specifies the absolute path of directories containing array.forces.raw, array.numbers.raw, array.positions.raw, cell.raw and info.energy.raw must be provided!"
exit
fi

if [ ! -f "raw_to_set.sh" ];then
echo "Error: File **raw_to_set.sh** which convert force.raw, type.raw, coord.raw, box.raw and energy.raw to deepmd npy format must be provided!"
exit
fi

if [ ! -f "input.json" ];then
cat >input.json<<!
{
    "_comment": " model parameters",
    "use_smooth":	true,
    "sel_a":		[199,199],
    "rcut_smth":	7.7,
    "rcut":		8.0,
    "filter_neuron":	[
        66,
        131,
        120
    ],
    "filter_resnet_dt":	false,
    "axis_neuron":	8,
    "fitting_neuron":	[
        240,
        133,
        66
    ],
    "fitting_resnet_dt":false,
    "coord_norm":	true,
    "type_fitting_net":	false,

    "_comment": " traing controls",
    "systems":		[
        "systems/"
    ],
    "set_prefix":	"set",    
    "stop_batch":	1,
    "batch_size": [
        1
    ],
    "start_lr":		5.0000000000e-04,
    "decay_steps":	10000,
    "decay_rate":	0.95,

    "start_pref_e":	1,
    "limit_pref_e":	1,
    "start_pref_f":	1000,
    "limit_pref_f":	1,
    "start_pref_v":	0.0,
    "limit_pref_v":	0.0,

    "seed":		1,

    "_comment": " display and restart",
    "_comment": " frequencies counted in batch",
    "disp_file":	"lcurve.out",
    "disp_freq":	1,
    "numb_test":	1,
    "save_freq":	1,
    "save_ckpt":	"model.ckpt",
    "load_ckpt":	"model.ckpt",
    "disp_training":	true,
    "time_training":	true,
    "profiling":	true,
    "profiling_file":	"timeline.json",

    "_comment":		"that's all"
}
!
fi

if [ ! -f "PARAMS.json" ];then
cat >PARAMS.json<<!
{
    "cutoff_1": 7.700,
    "cutoff_2": 8.000,
    "cutoff_3": 0.000,
    "cutoff_max": 8.000,
    "N_types_all_frame": 2,
    "type_index_all_frame": [
        0,
        1
    ],
    "N_Atoms_max": 452,
    "SEL_A_max": 150,
    "SEL_A_ele":[199,199],
    "Nframes_tot": 1,
    "sym_coord_type": 1,
    "N_sym_coord": 600,
    "batch_size": 1,
    "stop_epoch": 1,
    "num_filter_layer": 3,
    "filter_neuron": [
        66,
        131,
        120
    ],
    "axis_neuron": 8,
    "num_fitting_layer": 3,
    "fitting_neuron": [
        240,
        133,
        66
    ],
    "start_lr": 5.0000000000e-04,
    "decay_steps": -1,
    "decay_epoch": 1,
    "decay_rate": 9.5000000000e-01,
    "start_pref_e": 1.0000000000e+00,
    "limit_pref_e": 1.0000000000e+00,
    "start_pref_f": 1.0000000000e+04,
    "limit_pref_f": 1.0000000000e+00,
    "check_step": 1000,
    "check_batch": -1,
    "check_epoch": -1,
    "output_step": -1,
    "output_batch": -1,
    "output_epoch": 10,
    "save_step": -1,
    "save_batch": -1,
    "save_epoch": 10
}
!
fi

for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
rm -r ./$j
done


for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
mkdir $j
done

tot_num_dataset=`cat dataset_loc | wc -l`

test_num=$(rand 15 $tot_num_dataset)
echo "Total number of tests per directory: "$test_num

for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
for i in `seq 1 $test_num`
do
mkdir $j/$i
done
done

for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
for i in `seq 1 $test_num`
do
mkdir $j/$i/TorchANN
mkdir $j/$i/DPMD
mkdir $j/$i/DPMD/systems
select_which_line=$(rand 1 $tot_num_dataset)
selected_path=`sed -n "${select_which_line}p" dataset_loc`
cp $selected_path/array.forces.raw $j/$i
cp $selected_path/array.numbers.raw $j/$i
cp $selected_path/array.positions.raw $j/$i
cp $selected_path/cell.raw $j/$i
cp $selected_path/info.energy.raw $j/$i
done
done

for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
cd $j
for i in `seq 1 $test_num`
do 
cp ../PARAMS.json $i/TorchANN
cp ../input.json $i/DPMD
cp ../raw_to_set.sh $i/DPMD/systems
done
cd ../
done

for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
cd $j
for i in `seq 1 $test_num`
do
nframes=$(rand 1 5)
tail -n $nframes $i/array.forces.raw > $i/TorchANN/force.raw
tail -n $nframes $i/array.forces.raw > $i/DPMD/systems/force.raw
tail -n $nframes $i/array.positions.raw > $i/TorchANN/coord.raw
tail -n $nframes $i/array.positions.raw > $i/DPMD/systems/coord.raw
tail -n $nframes $i/cell.raw > $i/TorchANN/box.raw
tail -n $nframes $i/cell.raw > $i/DPMD/systems/box.raw
tail -n $nframes $i/info.energy.raw > $i/TorchANN/energy.raw
tail -n $nframes $i/info.energy.raw > $i/DPMD/systems/energy.raw
tail -n $nframes $i/array.numbers.raw > $i/TorchANN/type.raw
tail -n $nframes $i/array.numbers.raw > $i/DPMD/systems/type.raw
sed -i 's/15/0/g' $i/DPMD/systems/type.raw
sed -i 's/79/1/g' $i/DPMD/systems/type.raw
sed -i '2,999d' $i/DPMD/systems/type.raw
done
cd ..
done





for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
cd $j
for i in `seq 1 $test_num`
do
cd $i
filter1=$(rand 25 150)
filter2=$(rand 25 150)
filter3=$(rand 25 150)
fitting1=$(rand 200 300)
fitting2=$(rand 100 250)
fitting3=$(rand 50 150)
fitting4=$(rand 25 100)
axis=$(rand 1 16)
comma=,
bra=[
ket=]
space8="        "
sed -i "21c \ \ \ \ \ \ \ \ $filter1$comma" TorchANN/PARAMS.json
sed -i "22c \ \ \ \ \ \ \ \ $filter2$comma" TorchANN/PARAMS.json
sed -i "23c \ \ \ \ \ \ \ \ $filter3" TorchANN/PARAMS.json
sed -i "28c \ \ \ \ \ \ \ \ $fitting1$comma" TorchANN/PARAMS.json
sed -i "29c \ \ \ \ \ \ \ \ $fitting2$comma" TorchANN/PARAMS.json
sed -i "30c \ \ \ \ \ \ \ \ $fitting3$comma\ $fitting4" TorchANN/PARAMS.json
sed -i "25c \ \ \ \ \"axis_neuron\":\ $axis$comma" TorchANN/PARAMS.json

sed -i "8c \ \ \ \ \ \ \ \ $filter1$comma" DPMD/input.json
sed -i "9c \ \ \ \ \ \ \ \ $filter2$comma" DPMD/input.json
sed -i "10c \ \ \ \ \ \ \ \ $filter3" DPMD/input.json
sed -i "15c \ \ \ \ \ \ \ \ $fitting1$comma" DPMD/input.json
sed -i "16c \ \ \ \ \ \ \ \ $fitting2$comma" DPMD/input.json
sed -i "17c \ \ \ \ \ \ \ \ $fitting3$comma\ $fitting4" DPMD/input.json
sed -i "13c \ \ \ \ \"axis_neuron\":\ $axis$comma" DPMD/input.json

batchsize=`cat TorchANN/energy.raw | wc -l`
echo "batch_size for "$j"/"$i": "$batchsize
sed -i "17c \ \ \ \ \"batch_size\":\ $batchsize$comma" TorchANN/PARAMS.json
sed -i "30c \ \ \ \ \ \ \ \ $batchsize" DPMD/input.json
echo $batchsize > batch_size_train
echo $batchsize > batch_size_train

python3 -c\
'
import numpy as np
f=open("./n_types","w")
print(len(np.unique(np.loadtxt("./DPMD/systems/type.raw"))),file=f)
f.close()
'
n_types=`cat n_types`

if [ "$j" == "enough_sel_a" ];then
sel_a_1=$(rand 250 300)
sel_a_2=$(rand 250 300)
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$comma$sel_a_2$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$comma$sel_a_2$ket$comma" DPMD/input.json
if [ $n_types == 1 ];then
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$ket$comma" DPMD/input.json
sed -i 's/1/0/g' DPMD/systems/type.raw
fi
fi

if [ "$j" == "insufficient_sel_a_for_0" ];then
sel_a_1=$(rand 15 50)
sel_a_2=$(rand 250 300)
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$comma$sel_a_2$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$comma$sel_a_2$ket$comma" DPMD/input.json
if [ $n_types == 1 ];then
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$ket$comma" DPMD/input.json
sed -i 's/1/0/g' DPMD/systems/type.raw
fi
fi

if [ "$j" == "insufficient_sel_a_for_1" ];then
sel_a_1=$(rand 250 300)
sel_a_2=$(rand 13 99)
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$comma$sel_a_2$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$comma$sel_a_2$ket$comma" DPMD/input.json
if [ $n_types == 1 ];then
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$ket$comma" DPMD/input.json
sed -i 's/1/0/g' DPMD/systems/type.raw
fi
fi

if [ "$j" == "insufficient_sel_a_for_all" ];then
sel_a_1=$(rand 3 47)
sel_a_2=$(rand 7 109)
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$comma$sel_a_2$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$comma$sel_a_2$ket$comma" DPMD/input.json
if [ $n_types == 1 ];then
sed -i "13c \ \ \ \ \"SEL_A_ele\":\ $bra$sel_a_1$ket$comma" TorchANN/PARAMS.json
sed -i "4c \ \ \ \ \"sel_a\":\t\t$bra$sel_a_1$ket$comma" DPMD/input.json
sed -i 's/1/0/g' DPMD/systems/type.raw
fi
fi

cd ../
done
cd ../
done

echo "Start to run dp_train and TorchANN!"
for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
cd $j
for i in `seq 1 $test_num`
do
cd $i
echo "Running for system "$j"/"$i

cd DPMD
cd systems
./raw_to_set.sh > gen_runlog 2>&1
cd ../
~/PycharmProjects/deepmd_analysis/deepmd_test/bin/dp_train -t 1 input.json >runlog 2>&1
cd ../

cd TorchANN
/home/aurora/Documents/Study/Machine_Learning/DeePMD_torch/c/GENERATE_P > gen_runlog 2>&1
/home/aurora/Documents/Study/Machine_Learning/DeePMD_torch/python/no_mpi/train_noclassify_nompi.py > train_runlog 2>&1
cd ../


cd ../
done
cd ../
done

echo "Start to extract force for DPMD and TorchANN!"
for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
cd $j
for i in `seq 1 $test_num`
do
cd $i

#The file batch_size_train under this directory records batch_size for this frame
cd DPMD
grep force_hat runlog | sed -n '2p' | sed 's/\[/\ /g' | sed 's/\]/\ /g' | sed 's/force_hat_reshape://g' > force_DPMD 
cd ../
cd TorchANN
sed -n -e '/Force/,/Additional/p' train_runlog | sed '/Force/d' | sed '/Add/d' | sed 's/\[/\ /g' | sed 's/\]/\ /g' | sed 's/,/\ /g' | sed 's/tensor(//g' | sed 's/dev.*)//g' > force_TorchANN
sed -i ':a;N;s/\n/\ /g;ta' force_TorchANN
cd ../

b=`cat TorchANN/energy.raw | wc -l`
echo $b
echo $b > batch_size
echo $j"/"$i
python3 -c \
'
import numpy as np
batch_size=np.loadtxt("batch_size", dtype=np.int32)
DPMD=np.loadtxt("DPMD/force_DPMD", dtype=np.float64)
TORCH=np.loadtxt("TorchANN/force_TorchANN", dtype=np.float64)
DPMD=DPMD.reshape([batch_size, -1, 3])
TORCH=TORCH.reshape(DPMD.shape)
DPMD=DPMD.astype(np.float64)
TORCH=TORCH.astype(np.float64)
np.save("F_D", DPMD)
np.save("F_T", TORCH)

TORCH_ARRANGE=TORCH

DIFF=((DPMD-TORCH_ARRANGE)**2)**0.5
MAX_MIN=np.ndarray(2, dtype=DPMD.dtype)
MAX_MIN[0]=DIFF.max()
MAX_MIN[1]=DIFF.min()
np.savetxt("MAX_MIN", MAX_MIN)
'

cd ../
done
cd ../
done

echo aaa > MAX_MIN_all
for j in enough_sel_a insufficient_sel_a_for_0 insufficient_sel_a_for_1 insufficient_sel_a_for_all
do
for i in `seq 1 $test_num`
do
echo $j"/"$i >> MAX_MIN_all
cat $j/$i/MAX_MIN >> MAX_MIN_all 
done
done
sed -i '1d' MAX_MIN_all

