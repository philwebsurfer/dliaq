#!/bin/sh
for model in  model_baseline00.h5 model_best01a.h5 model_conv00.h5 model_conv02.h5 model_dnn00.h5 model_dnn01.h5
do 
	echo '################################################# Running '$model' ##############################################################################'
	AIP_MODEL_DIR='' python -m trainer.task \
		-b 256 \
		--sequence_length=4320 \
		--sampling_rate=2 \
		--stride=2 \
		--steps=10 \
		--epochs=50 \
		--model=../data/models.untrained/${model} \
		https://github.com/philwebsurfer/dliaq/raw/main/data/data_5min.pickle.gz \
		../data/output-hyper05min-w15-stride2-samplingrate2
done
