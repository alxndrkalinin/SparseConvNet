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
		mkdir -p "$datasetDir/merged"
		mergedPath="$datasetDir/merged/$cellClass"
		mkdir -p $mergedPath
		runs=$(ls $nucleiPath)
		runs=($runs)

		for run in "${runs[@]}"; do
			nuclei="$(ls $nucleiPath/$run)"
			nucleoli="$(ls $nucleoliPath/$run)"
			nuclei=($nuclei)
			nucleoli=($nucleoli)
			mergedRunPath="$mergedPath/$run"
			mkdir -p $mergedRunPath

			for nucleus in "${nuclei[@]}"; do
				nucleusName="${nucleus%.off*}"
				nucleusPath="$nucleiPath/$run/$nucleus"
				# echo "===========$nucleusName============="
				mergedNucleusPath=""

				for nucleolus in "${nucleoli[@]}"; do
					# echo "-----$nucleolus-----"
					nucleolusName="${nucleolus%.off*}"

					if [[ $nucleolusName == *$nucleusName* ]]; then
						nucleolusPath="$nucleoliPath/$run/$nucleolus"

						if [[ -z $mergedNucleusPath ]]; then
							mergedNucleusPath="$mergedRunPath/${nucleusName}_merged.off"
							arglist="$nucleolusPath $nucleusPath $mergedNucleusPath"

							python merge_off.py $arglist
						else
							# echo "$nucleolusPath + $mergedNucleusPath ==> $mergedNucleusPath"
							arglist="$nucleolusPath $mergedNucleusPath $mergedNucleusPath"
							python merge_off.py $arglist
						fi
					fi
				done
			done
		done
	else
		echo "Number of runs in class $cellClass doesn't not match"
		exit 1
	fi
done
< $1