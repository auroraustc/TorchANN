#!/bin/bash

#---Provide any necessary defination here ---------------------------
exec_g=EXEC_G
exec_p=EXEC_P
#nprocs=1
#machinefile=./machinefile

#---Prepare the "parameter" input file here -------------------------
#if [ ! -f INCAR.lasp.external ]; then 
#  cp INCAR INCAR.lasp.external
#  sed -i '/ISTART/d'       INCAR.lasp.external
#  sed -i '/ICHARG/d'       INCAR.lasp.external
#  sed -i '/LWAVE/d'        INCAR.lasp.external 
#  sed -i '/LCHARG/d'       INCAR.lasp.external
#  sed -i '/IBRION/d'       INCAR.lasp.external
#  sed -i '/NSW/d'          INCAR.lasp.external
#  sed -i '/ISIF/d'         INCAR.lasp.external
#  sed -i '/ISYM/d'         INCAR.lasp.external
#  sed -i '/NWRITE/d'       INCAR.lasp.external
#  echo "ISTART = 1" >>     INCAR.lasp.external
#  echo "ICHARG = 1" >>     INCAR.lasp.external
#  echo "LWAVE = .TRUE." >> INCAR.lasp.external
#  echo "LCHARG = .TRUE." >>INCAR.lasp.external
#  echo "IBRION = 0" >>     INCAR.lasp.external
#  echo "ISYM = 0" >>       INCAR.lasp.external
#  echo "NSW  = 0" >>       INCAR.lasp.external
#  echo "ISIF = 3" >>       INCAR.lasp.external   
#  echo "NWRITE = 2" >>     INCAR.lasp.external  
#  sed -i 's/FALSE/Auto/g'  INCAR.lasp.external
#fi
#\cp -f INCAR.lasp.external INCAR

#---Prepare the "coordinate" input file here ------------------------
#echo " For LASP external use" >POSCAR
#echo "1.0000" >>POSCAR
#head -3 external.coord >>POSCAR
#a=(`grep VRHFIN POTCAR |awk '{print $2}'|sed 's/=//'|sed 's/://'`)
#echo ${a[*]} >> POSCAR
#j=0
#\rm -f lasp.coord.tmp
#for i in ${a[*]}
#do
#  b[$j]=`grep "$i " external.coord|wc -l` 
#  grep "$i " external.coord >>lasp.coord.tmp
#  let "j=$j+1"
#done
#echo ${b[*]} >>POSCAR
#echo "C" >>POSCAR
#cat lasp.coord.tmp|awk '{print " "$2" "$3" "$4}' >>POSCAR
#
cp external.coord type.raw
cp external.coord box.raw
cp external.coord coord.raw
sed -i '1,3d' type.raw
sed -i 's/Au.*/79\ /g' type.raw
sed -i 's/P.*/15\ /g' type.raw
sed -i ':a;N;s/\n/\ /g;ta' type.raw
sed -i '4,9999d' box.raw
sed -i ':a;N;s/\n/\ /g;ta' box.raw
sed -i '1,3d' coord.raw
awk '{print $2,$3,$4}' coord.raw > coord.raww
mv coord.raww coord.raw
sed -i ':a;N;s/\n/\ /g;ta' coord.raw
cp coord.raw force.raw
awk '{print $1}' coord.raw > energy.raw

#---Run the executable file here ------------------------------------
#if [ "$1" == "T" ]; then \rm -f CHGCAR WAVECAR;fi 
$exec_g > runlog_g
$exec_p 
cat coord.raw >> coord_all.raw
cat box.raw >> box_all.raw
cat type.raw >> type_all.raw

#---Extract energy here ---------------------------------------------
#grep TOTEN OUTCAR |tail -1|awk '{print $5}' >external.ene
grep Epoch PREDICT.OUT | awk '{print $8}' > external.ene
grep Epoch PREDICT.OUT | awk '{print $8}' >> energy_all.raw

#---Extract force here ----------------------------------------------
#n=`cat lasp.coord.tmp |wc -l`
#for ((i=1;i<=$n;i++))
#do
#  echo f.$i.f >>external.ene
#done
#let "n=$n+1"
#grep -A$n TOTAL-FORCE OUTCAR |sed 1,2d|awk '{print $4" "$5" "$6}'>lasp.force.tmp
#j=1
#for i in `cat lasp.coord.tmp|awk '{print $NF}'`
#do
#  f=`sed -n "$j"p lasp.force.tmp`
#  sed -i "s/f.$i.f/$f/" external.ene
#  let "j=$j+1"
#done
sed -n -e '/Force/,/Stress/p' PREDICT.OUT | sed '/Stress/d' | sed '/Force/d' >> external.ene 
sed -n -e '/Force/,/Stress/p' PREDICT.OUT | sed '/Stress/d' | sed '/Force/d' > force.tmp
sed -i ':a;N;s/\n/\ /g;ta' force.tmp
cat force.tmp >> force_all.raw
rm force.tmp


#---Extract stress here ---------------------------------------------
#c=(`grep 'FORCE on cell' OUTCAR -A13|tail -1|awk '{$1=""}1'`)
#v=`grep "volume of cell" OUTCAR|tail -1|awk '{print $NF}'`
#for i in {0..5}
#do
#  c[$i]=`echo "${c[$i]}/$v"|bc -l`
#done
#echo ${c[0]} ${c[3]} ${c[5]}  >>external.ene
#echo ${c[3]} ${c[1]} ${c[4]}  >>external.ene
#echo ${c[5]} ${c[4]} ${c[2]}  >>external.ene

grep 'Stress' -A3 PREDICT.OUT | sed '/Stress/d' > stress_tmp
#python3 ./treat_stress.py
python3 -c \
'
import numpy as np

box=np.loadtxt("./box.raw")
box=box.reshape(-1, 3)
volume=np.abs(np.dot(np.cross(box[0], box[1]),box[2]))
stress=np.loadtxt("./stress_tmp")
f_stress_all=open("stress_all.raw", "a")
np.savetxt(f_stress_all, stress.reshape(1, -1), fmt="%10.6f")
f_stress_all.close()
stress=stress/volume
np.savetxt("./stress_tmpp", stress, fmt="%10.6f")
'
rm stress_tmp
cat stress_tmpp >> external.ene
rm stress_tmpp


