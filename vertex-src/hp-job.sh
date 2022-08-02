#!/bin/sh
# References:
# - https://cloud.google.com/sdk/gcloud/reference/ai/hp-tuning-jobs/create
# - https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec
# - https://cloud.google.com/vertex-ai/docs/reference/rest/v1/StudySpec
# - https://stackoverflow.com/a/67610900/7323086

function usage {
		echo "Usage: $0: [--region=us-central1] job-name config-file" # [dataset-source-path] [output-path]"
		echo ""
		echo "--region:      GCloud region for this job. Default: us-central1 (IA, US)."
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
}
if [ -z $1 ] | [ -z $2 ] | [[ ${#} -eq 0 ]]
then
	usage
fi

region='us-central1'

TEMP=$(getopt -o 'h,r:' --long 'help,region:' -n "$0" -- "$@")

# Note the quotes around "$TEMP": they are essential!
eval set -- "$TEMP"
unset TEMP

while true; do
  case "${1}" in
	  '-r' | '--region')
		  if [ -z $2 ]; then
			  echo 'Region empty! Check usage!' >&2
			  usage
			  exit 2
		  fi
		  region="$2"
		  shift 2
		  continue
		  ;;
	  '-' | '--')
		  shift
		  break
		  ;;
	  '-h' | '--help' | ? | *)
		  echo "Invalid option: -${OPTARG}."
		  echo
		  usage
		  exit 1
		  ;;
  esac
done

jobname="$1"
configfile="$2"

if [ -z $configfile ] | [[ '' == $configfile ]] | [[ ! -f $configfile ]] #| [ !-r $configfile ]
then
	echo "File $configfile "'does not exist!' >&2
	echo 'Terminating execution!!!' >&2
	exit 252
fi
if [ -z $jobname ]; then
	echo "Job name not set $jobname "'! Check usage.' >&2
	echo 'Terminating execution!!!' >&2
	usage
	exit 252
fi
# echo configfile'="'$configfile'"'
# echo jobname'="'$jobname'"'
	

maxTrialCount="$(awk '/maxTrialCount:/{sub(/^[^:]+:\s+/, "", $0);print}' < $configfile)"
parallelTrialCount="$(awk '/parallelTrialCount:/{sub(/^[^:]+:\s+/, "", $0);print}' < $configfile)"

echo "Running gcloud ai hyperparameter tunning job: ..."
echo
echo '---------------------------------------------------'
echo '| GCP Job Name:             '${jobname}  
echo '| Job Submitted File Name:  '${configfile}  
echo '| Max Trial Count:          '${maxTrialCount}  
echo '| Max Parallel Trial Count: '${parallelTrialCount}
echo '| Region:                   '${region}
echo '---------------------------------------------------'
echo
echo -e  "Start time $(date '+%Y-%m-%d %H:%M:%S')\n==============================\n"
echo gcloud ai \
  hp-tuning-jobs \
  create \
  --config=$2 \
  --display-name=$1 \
  --max-trial-count=${maxTrialCount} \
  --parallel-trial-count=${parallelTrialCount} \
  --region=${region} #\
#  --verbosity debug
echo -e  "\n==============================\nEnd time $(date '+%Y-%m-%d %H:%M:%S')\n"
