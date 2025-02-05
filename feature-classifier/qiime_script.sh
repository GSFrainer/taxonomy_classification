#!/bin/bash

trap 'echo -e "${RED}Execution aborted!${NO_COLOR}"; exit 1' ERR

RED='\033[0;31m'
GREEN='\033[0;32m'
NO_COLOR='\033[0m'

levels="class order family genus species"

# splitters="RandomSplit IsolatedRandomSplit StratifiedSplit IsolatedStratifiedSplit"
# splitter=""
splitters="prop_0-1/min_5/RandomSplit_0" # prop_0-1/min_5/RandomSplit_14 prop_0-1/min_5/RandomSplit_56 prop_0-1/min_5/RandomSplit_84 prop_0-1/min_5/RandomSplit_92 prop_0-1/min_5/RandomSplit_101 prop_0-1/min_5/RandomSplit_105 prop_0-1/min_5/RandomSplit_227 prop_0-1/min_5/StratifiedSplit2_0 prop_0-1/min_5/StratifiedSplit2_14 prop_0-1/min_5/StratifiedSplit2_56 prop_0-1/min_5/StratifiedSplit2_84 prop_0-1/min_5/StratifiedSplit2_92 prop_0-1/min_5/StratifiedSplit2_101 prop_0-1/min_5/StratifiedSplit2_105 prop_0-1/min_5/StratifiedSplit2_227 prop_0-1/min_10/RandomSplit_0 prop_0-1/min_10/RandomSplit_14 prop_0-1/min_10/RandomSplit_56 prop_0-1/min_10/RandomSplit_84 prop_0-1/min_10/RandomSplit_92 prop_0-1/min_10/RandomSplit_101 prop_0-1/min_10/RandomSplit_105 prop_0-1/min_10/RandomSplit_227 prop_0-1/min_10/StratifiedSplit2_0 prop_0-1/min_10/StratifiedSplit2_14 prop_0-1/min_10/StratifiedSplit2_56 prop_0-1/min_10/StratifiedSplit2_84 prop_0-1/min_10/StratifiedSplit2_92 prop_0-1/min_10/StratifiedSplit2_101 prop_0-1/min_10/StratifiedSplit2_105 prop_0-1/min_10/StratifiedSplit2_227 prop_0-2/min_5/RandomSplit_0 prop_0-2/min_5/RandomSplit_14 prop_0-2/min_5/RandomSplit_56 prop_0-2/min_5/RandomSplit_84 prop_0-2/min_5/RandomSplit_92 prop_0-2/min_5/RandomSplit_101 prop_0-2/min_5/RandomSplit_105 prop_0-2/min_5/RandomSplit_227 prop_0-2/min_5/StratifiedSplit2_0 prop_0-2/min_5/StratifiedSplit2_14 prop_0-2/min_5/StratifiedSplit2_56 prop_0-2/min_5/StratifiedSplit2_84 prop_0-2/min_5/StratifiedSplit2_92 prop_0-2/min_5/StratifiedSplit2_101 prop_0-2/min_5/StratifiedSplit2_105 prop_0-2/min_5/StratifiedSplit2_227 prop_0-2/min_10/RandomSplit_0 prop_0-2/min_10/RandomSplit_14 prop_0-2/min_10/RandomSplit_56 prop_0-2/min_10/RandomSplit_84 prop_0-2/min_10/RandomSplit_92 prop_0-2/min_10/RandomSplit_101 prop_0-2/min_10/RandomSplit_105 prop_0-2/min_10/RandomSplit_227 prop_0-2/min_10/StratifiedSplit2_0 prop_0-2/min_10/StratifiedSplit2_14 prop_0-2/min_10/StratifiedSplit2_56 prop_0-2/min_10/StratifiedSplit2_84 prop_0-2/min_10/StratifiedSplit2_92 prop_0-2/min_10/StratifiedSplit2_101 prop_0-2/min_10/StratifiedSplit2_105 prop_0-2/min_10/StratifiedSplit2_227 prop_0-05/min_5/RandomSplit_0 prop_0-05/min_5/RandomSplit_14 prop_0-05/min_5/RandomSplit_56 prop_0-05/min_5/RandomSplit_84 prop_0-05/min_5/RandomSplit_92 prop_0-05/min_5/RandomSplit_101 prop_0-05/min_5/RandomSplit_105 prop_0-05/min_5/RandomSplit_227 prop_0-05/min_5/StratifiedSplit2_0 prop_0-05/min_5/StratifiedSplit2_14 prop_0-05/min_5/StratifiedSplit2_56 prop_0-05/min_5/StratifiedSplit2_84 prop_0-05/min_5/StratifiedSplit2_92 prop_0-05/min_5/StratifiedSplit2_101 prop_0-05/min_5/StratifiedSplit2_105 prop_0-05/min_5/StratifiedSplit2_227 prop_0-05/min_10/RandomSplit_0 prop_0-05/min_10/RandomSplit_14 prop_0-05/min_10/RandomSplit_56 prop_0-05/min_10/RandomSplit_84 prop_0-05/min_10/RandomSplit_92 prop_0-05/min_10/RandomSplit_101 prop_0-05/min_10/RandomSplit_105 prop_0-05/min_10/RandomSplit_227 prop_0-05/min_10/StratifiedSplit2_0 prop_0-05/min_10/StratifiedSplit2_14 prop_0-05/min_10/StratifiedSplit2_56 prop_0-05/min_10/StratifiedSplit2_84 prop_0-05/min_10/StratifiedSplit2_92 prop_0-05/min_10/StratifiedSplit2_101 prop_0-05/min_10/StratifiedSplit2_105 prop_0-05/min_10/StratifiedSplit2_227"


while [[ $# -gt 0 ]]; do
    case "$1" in
        --level)
            if [[ " $levels " == *" $2 "* ]]; then
                levels=$2
            else
                echo -e "${RED}Invalid level option${NO_COLOR}"
                exit 1
            fi
            shift 2
            ;;
        --splitter)
            if [[ " $splitters " == *" $2 "* ]]; then
                splitters=$2
            else
                echo -e "${RED}Invalid splitter option${NO_COLOR}"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo -e "${RED}Invalid option '$1' ${NO_COLOR}"
            exit 1
            ;;
    esac
done

# if ! [[ " $splitters " == *" $splitter "* ]]; then
#     echo -e "${RED}Invalid splitter option${NO_COLOR}"
#     exit 1
# fi

echo "splitter, level, start_time, test_time, end_time" >> feature_classifier_results.txt
for splitter in $splitters
do

    for level in $levels
    do
        start_time=$(date +%s)

        if [ -d "../new_data/$splitter/$level" ]; then 
            echo "Ok"
        else
            echo -e "-- It will not be executed at ${RED}$level${NO_COLOR} level. The directory ../new_data/$splitter/$level does not exist."
            echo "$splitter,$level,$start_time,," >> feature_classifier_results.txt
            continue
        fi
        
        mkdir -p $splitter/$level
        
        dir="./$splitter/$level/"
        if [ -n "$(ls -A "$dir")" ]; then
            echo -e "-- It will not be executed at ${RED}$level${NO_COLOR} level. The directory is not empty."
            echo "$splitter,$level,$start_time,," >> feature_classifier_results.txt
            continue
        fi

        echo -e "\n Running feature classifier for $level level with $splitter ..."

        echo "      Importing train reference sequence..."
        echo -n "      "
        qiime tools import --type 'FeatureData[Sequence]' --input-path ../new_data/$splitter/$level/pr2_train.fasta --output-path ./$splitter/$level/ref-seqs.qza

        echo "      Importing train reference taxonomy..."
        echo -n "      "
        qiime tools import --type 'FeatureData[Taxonomy]' --input-format HeaderlessTSVTaxonomyFormat --input-path ../new_data/$splitter/$level/pr2_train_taxonomy.txt --output-path ./$splitter/$level/ref-taxonomy.qza

        echo "      Training classifier..."
        echo -n "      "
        qiime feature-classifier fit-classifier-naive-bayes --i-reference-reads ./$splitter/$level/ref-seqs.qza --i-reference-taxonomy ./$splitter/$level/ref-taxonomy.qza --o-classifier ./$splitter/$level/classifier.qza

        test_time=$(date +%s)
        echo "      Importing test sequences..."
        echo -n "      "
        qiime tools import --type 'FeatureData[Sequence]' --input-path ../new_data/$splitter/$level/pr2_test.fasta --output-path ./$splitter/$level/test-seqs.qza

        echo "      Classifying test sequences..."
        echo -n "      "
        qiime feature-classifier classify-sklearn --i-classifier ./$splitter/$level/classifier.qza --i-reads ./$splitter/$level/test-seqs.qza --o-classification ./$splitter/$level/test-taxonomy.qza

        end_time=$(date +%s)
        echo "      Generating qiime visualization file..."
        echo -n "      "
        qiime metadata tabulate --m-input-file ./$splitter/$level/test-taxonomy.qza --o-visualization ./$splitter/$level/test-taxonomy.qzv

        echo "      Exporting results..."
        echo -n "      "
        qiime tools extract --input-path ./$splitter/$level/test-taxonomy.qza --output-path ./$splitter/$level/results

        folders=$(ls -d ./$splitter/$level/results/*/)

        for folder in $folders
        do
            mv $folder* ./$splitter/$level/results
            rm -r $folder
        done
        echo -e "      Saving results at ./$splitter/$level/results..."
        echo -e "      ${GREEN}Done: $level${NO_COLOR}\n"
        echo "$splitter,$level,$start_time,$test_time,$end_time" >> feature_classifier_results.txt
    done
done