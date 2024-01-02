#!/usr/bin/env bash

#export one_test=1
export KMax=50

./experiment_no_vr_a9a.sh > experiment_no_vr_a9a.txt &
wait
echo "1/7 no VR test finished"

./experiment_no_vr_mushroom.sh > experiment_no_vr_mushroom.txt &
wait
echo "2/7 no VR test finished"

./experiment_no_vr_duke.sh > experiment_no_vr_duke.txt &
wait
echo "3/7 no VR test finished"

./experiment_no_vr_gisette.sh > experiment_no_vr_gisette.txt &
wait
echo "4/7 no VR test finished"

./experiment_no_vr_madelon.sh > experiment_no_vr_madelon.txt &
wait
echo "5/7 no VR test finished"

./experiment_no_vr_phishing.sh > experiment_no_vr_phishing.txt &
wait
echo "6/7 no VR test finished"

./experiment_no_vr_w8a.sh > experiment_no_vr_w8a.txt &
wait
echo "7/7 no VR test finished"



./experiment_vr_all_a9a.sh > experiment_vr_all_a9a.txt &
wait
echo "1/7 with-VR test finished"

./experiment_vr_all_mushroom.sh > experiment_vr_all_mushroom.txt &
wait
echo "2/7 with-VR test finished"

./experiment_vr_all_duke.sh > experiment_vr_all_duke.txt &
wait
echo "3/7 with-VR test finished"

./experiment_vr_all_gisette.sh > experiment_vr_all_gisette.txt &
wait
echo "4/7 with-VR test finished"

./experiment_vr_all_madelon.sh > experiment_vr_all_madelon.txt &
wait
echo "5/7 with-VR test finished"

./experiment_vr_all_phishing.sh > experiment_vr_all_phishing.txt &
wait
echo "6/7 with-VR test finished"

./experiment_vr_all_w8a.sh > experiment_vr_all_w8a.txt &
wait
echo "7/7 with-VR test finished"
