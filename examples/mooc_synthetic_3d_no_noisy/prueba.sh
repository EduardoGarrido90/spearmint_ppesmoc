for ((i=0;i<100;i+=1)); do
	echo $i
	j=$(($i + 1))
	sed -i -- "s/NUM_EXP = ${i}/NUM_EXP = ${j}/g" pesmoc/prog.py
	sed -i -- "s/NUM_EXP = ${i}/NUM_EXP = ${j}/g" random/prog.py
	sed -i -- "s/NUM_EXP = ${i}/NUM_EXP = ${j}/g" pesmoc/prog_no_noisy.py
	sed -i -- "s/NUM_EXP = ${i}/NUM_EXP = ${j}/g" random/prog_no_noisy.py
	sed -i -- "s/\"random_seed\"     : ${i}/\"random_seed\"     : ${j}/g" pesmoc/config.json
	sed -i -- "s/\"random_seed\"     : ${i}/\"random_seed\"     : ${j}/g" random/config.json
	sed -i -- "s/\"experiment-name\" : \"synthetic_${i}_noiseless\"/\"experiment-name\" : \"synthetic_${j}_noiseless\"/g" pesmoc/config.json
	sed -i -- "s/\"experiment-name\" : \"synthetic_${i}_noiseless\"/\"experiment-name\" : \"synthetic_${j}_noiseless\"/g" random/config.json
done
