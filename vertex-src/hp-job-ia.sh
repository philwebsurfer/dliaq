#!/bin/sh
# References:
# - https://cloud.google.com/sdk/gcloud/reference/ai/hp-tuning-jobs/create
# - https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec
# - https://cloud.google.com/vertex-ai/docs/reference/rest/v1/StudySpec
# - https://stackoverflow.com/a/67610900/7323086
if [ -z $1 ] | [ -z $2 ] #| [ -z $3 ]
then
	echo "Usage: $0: [job-name] [config-file]" # [dataset-source-path] [output-path]"
	echo ""
	echo "job-name:      GCloud name for this job. Should be without spaces."
	echo "config-file:   YAML configuration file."
	echo "               Further info: https://cloud.google.com/sdk/gcloud/reference/ai/hp-tuning-jobs/create#--config."
	echo -e "\n\nExample"
	echo "$0 hyper15-10min-w45-stride2-samplingrate2 hp-10min-w45-stride2-samplingrate2.yaml"
	echo -e "\n\nJSON to YAML converter info https://stackoverflow.com/a/67610900/7323086"
	#echo "dataset-source-path: container path to the dataset for training job."
	#echo "output-path:         container path to the dataset for trained "
        #echo "                     models and outputs."
	exit 1
fi

echo "Running gcloud ai hyperparameter tunning job: ..."
echo -e  "Start time $(date '+%Y-%m-%d %H:%M:%S')\n==============================\n"
gcloud ai \
  hp-tuning-jobs \
  create \
  --config=$2 \
  --display-name=$1 \
  --max-trial-count=11 \
  --parallel-trial-count=11 \
  --region=us-central1 #\
#  --verbosity debug
echo -e  "\n==============================\nEnd time $(date '+%Y-%m-%d %H:%M:%S')\n"
