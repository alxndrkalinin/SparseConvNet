#!/bin/bash

# call: "mergeNucleiNucleoli.sh path/to/dataset", path has to have nuclei/off and nucleoli/off directories
# output: creates directory 'merged' above and creates directories for runs

# classes are required to find subfolders in nuclei and nucleoli directories
cellClasses=('norm' 'ss')

datasetDir=$1

for cellClass in "${cellClasses[@]}"; do
	nucleiPath="$datasetDir/nuclei/off/$cellClass"
	nucleoliPath="$datasetDir/nucleoli/off/${cellClass}_off_named"
	nucleiRunNum="$(ls $nucleiPath | wc -l)"
	nucleoliRunNum="$(ls $nucleoliPath | wc -l)"

	# make sure number of runs matches for nuclei and nucleoli
	if [ "$nucleiRunNum" -eq "$nucleoliRunNum" ]; then
		echo $cellClass
		# create output directory
		# mkdir -p "$datasetDir/merged"
		mergedPath="$datasetDir/merged/$cellClass"
		# mkdir -p $mergedPath
		runs=$(ls $nucleiPath)
		runs=($runs)
		
		for run in "${runs[@]}"; do
			nuclei="$(ls $nucleiPath/$run)"
			nucleoli="$(ls $nucleoliPath/$run)"
			nuclei=($nuclei)
			nucleoli=($nucleoli)
			mergedRunPath="$mergedPath/$run"
			# mkdir -p $mergedRunPath

			for nucleus in "${nuclei[@]}"; do
				nucleusName="${nucleus%.off*}"
				nucleusPath="$nucleiPath/$run/$nucleus"
				echo "===========$nucleusName============="
				mergedNucleusPath=""

				for nucleolus in "${nucleoli[@]}"; do
					nucleolusName="${nucleolus%.off*}"

					if [[ $nucleolusName == *$nucleusName* ]]; then
						nucleolusPath="$nucleoliPath/$run/$nucleolus"
						if [[ -z $mergedNucleusPath ]]; then
							mergedNucleusPath="$mergedRunPath/${nucleusName}_merged.off"
							# echo "$nucleolusPath + $nucleusPath ==> $mergedNucleusPath"
							python merge_off.py $nucleolusPath $nucleusPath $mergedNucleusPath
						else
							# echo "$nucleolusPath + $mergedNucleusPath ==> $mergedNucleusPath"
							python merge_off.py $nucleolusPath $mergedNucleusPath $mergedNucleusPath
						fi
					fi
				done
			done
		done
	else
		echo "Number of runs in class $cellClass doesn't not match"
		break
	fi
done < $1