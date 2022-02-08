#2020_2_24_period1_to_2020_3_13.csv 
#2020_2_9_period2_to_2020_3_21.csv
#2020_9_12_period3_to_2020_10_15.csv
#2020_9_12_period4_to_2020_11_14.csv
#2020_6_1_period5_to_2020_9_1.csv
#2020_1_1_period6_to_2020_12_31

export FILENAME="2020_6_1_period5_to_2020_9_1.csv"
export PERIOD="period5"

python3 checkfeaturesimpdetails.py \
  --labels-file="$FILENAME" \
  --pollutantsnames="avg_wco_"$PERIOD"_2020,avg_wnh3_"$PERIOD"_2020,avg_wnmvoc_"$PERIOD"_2020,avg_wno2_"$PERIOD"_2020,avg_wno_"$PERIOD"_2020,avg_wo3_"$PERIOD"_2020,avg_wpans_"$PERIOD"_2020,avg_wpm10_"$PERIOD"_2020,avg_wpm2p5_"$PERIOD"_2020,avg_wso2_"$PERIOD"_2020,sum_wnh3_ex_q75_"$PERIOD"_2020,sum_wnmvoc_ex_q75_"$PERIOD"_2020,sum_wno2_ex_q75_"$PERIOD"_2020,sum_wno_ex_q75_"$PERIOD"_2020,sum_wpans_ex_q75_"$PERIOD"_2020,sum_wpm10_ex_q75_"$PERIOD"_2020,sum_wpm2p5_ex_q75_"$PERIOD"_2020,sum_wo3_ex_q75_"$PERIOD"_2020,sum_wco_ex_q75_"$PERIOD"_2020,sum_wso2_ex_q75_"$PERIOD"_2020" \
  --featstouse="density,commutersdensity,lat,depriv,Ratio0200ver65,avg_wpm10_"$PERIOD"_2020,avg_wpm2p5_"$PERIOD"_2020,avg_wno2_"$PERIOD"_2020,avg_wno_"$PERIOD"_2020,avg_wnh3__"$PERIOD"_2020,avg_wpans_"$PERIOD"_2020,avg_wnmvoc_"$PERIOD"_2020,avg_wo3_"$PERIOD"_2020,avg_wco_"$PERIOD"_2020,avg_wso2_"$PERIOD"_2020"

mkdir -p ./results/"$PERIOD"/using_lat/
mv rf_model_* ./results/"$PERIOD"/using_lat/

python3 checkfeaturesimpdetails.py \
  --labels-file="$FILENAME" \
  --pollutantsnames="avg_wco_"$PERIOD"_2020,avg_wnh3_"$PERIOD"_2020,avg_wnmvoc_"$PERIOD"_2020,avg_wno2_"$PERIOD"_2020,avg_wno_"$PERIOD"_2020,avg_wo3_"$PERIOD"_2020,avg_wpans_"$PERIOD"_2020,avg_wpm10_"$PERIOD"_2020,avg_wpm2p5_"$PERIOD"_2020,avg_wso2_"$PERIOD"_2020,sum_wnh3_ex_q75_"$PERIOD"_2020,sum_wnmvoc_ex_q75_"$PERIOD"_2020,sum_wno2_ex_q75_"$PERIOD"_2020,sum_wno_ex_q75_"$PERIOD"_2020,sum_wpans_ex_q75_"$PERIOD"_2020,sum_wpm10_ex_q75_"$PERIOD"_2020,sum_wpm2p5_ex_q75_"$PERIOD"_2020,sum_wo3_ex_q75_"$PERIOD"_2020,sum_wco_ex_q75_"$PERIOD"_2020,sum_wso2_ex_q75_"$PERIOD"_2020" \
  --featstouse="density,commutersdensity,lat,depriv,Ratio0200ver65,sum_wpm10_ex_q75_"$PERIOD"_2020,sum_wpm2p5_ex_q75_"$PERIOD"_2020,sum_wno2_ex_q75_"$PERIOD"_2020,sum_wno_ex_q75_"$PERIOD"_2020,sum_wnh3_ex_q75_"$PERIOD"_2020,sum_wpans_ex_q75_"$PERIOD"_2020,sum_wnmvoc_ex_q75_"$PERIOD"_2020,sum_wo3_ex_q75_"$PERIOD"_2020,sum_wco_ex_q75_"$PERIOD"_2020,sum_wso2_ex_q75_"$PERIOD"_2020"

mkdir -p ./results/"$PERIOD"/using_lat_ex_q75/
mv rf_model_* ./results/"$PERIOD"/using_lat_ex_q75/

python3 checkfeaturesimpdetails.py \
  --labels-file="$FILENAME" \
  --pollutantsnames="avg_wco_"$PERIOD"_2020,avg_wnh3_"$PERIOD"_2020,avg_wnmvoc_"$PERIOD"_2020,avg_wno2_"$PERIOD"_2020,avg_wno_"$PERIOD"_2020,avg_wo3_"$PERIOD"_2020,avg_wpans_"$PERIOD"_2020,avg_wpm10_"$PERIOD"_2020,avg_wpm2p5_"$PERIOD"_2020,avg_wso2_"$PERIOD"_2020,sum_wnh3_ex_q75_"$PERIOD"_2020,sum_wnmvoc_ex_q75_"$PERIOD"_2020,sum_wno2_ex_q75_"$PERIOD"_2020,sum_wno_ex_q75_"$PERIOD"_2020,sum_wpans_ex_q75_"$PERIOD"_2020,sum_wpm10_ex_q75_"$PERIOD"_2020,sum_wpm2p5_ex_q75_"$PERIOD"_2020,sum_wo3_ex_q75_"$PERIOD"_2020,sum_wco_ex_q75_"$PERIOD"_2020,sum_wso2_ex_q75_"$PERIOD"_2020" \
  --featstouse="density,commutersdensity,depriv,Ratio0200ver65,avg_wpm10_"$PERIOD"_2020,avg_wpm2p5_"$PERIOD"_2020,avg_wno2_"$PERIOD"_2020,avg_wno_"$PERIOD"_2020,avg_wnh3__"$PERIOD"_2020,avg_wpans_"$PERIOD"_2020,avg_wnmvoc_"$PERIOD"_2020,avg_wo3_"$PERIOD"_2020,avg_wco_"$PERIOD"_2020,avg_wso2_"$PERIOD"_2020"

mkdir -p ./results/"$PERIOD"/without_lat/
mv rf_model_* ./results/"$PERIOD"/without_lat/

python3 checkfeaturesimpdetails.py \
  --labels-file="$FILENAME" \
  --pollutantsnames="avg_wco_"$PERIOD"_2020,avg_wnh3_"$PERIOD"_2020,avg_wnmvoc_"$PERIOD"_2020,avg_wno2_"$PERIOD"_2020,avg_wno_"$PERIOD"_2020,avg_wo3_"$PERIOD"_2020,avg_wpans_"$PERIOD"_2020,avg_wpm10_"$PERIOD"_2020,avg_wpm2p5_"$PERIOD"_2020,avg_wso2_"$PERIOD"_2020,sum_wnh3_ex_q75_"$PERIOD"_2020,sum_wnmvoc_ex_q75_"$PERIOD"_2020,sum_wno2_ex_q75_"$PERIOD"_2020,sum_wno_ex_q75_"$PERIOD"_2020,sum_wpans_ex_q75_"$PERIOD"_2020,sum_wpm10_ex_q75_"$PERIOD"_2020,sum_wpm2p5_ex_q75_"$PERIOD"_2020,sum_wo3_ex_q75_"$PERIOD"_2020,sum_wco_ex_q75_"$PERIOD"_2020,sum_wso2_ex_q75_"$PERIOD"_2020" \
  --featstouse="density,commutersdensity,depriv,Ratio0200ver65,sum_wpm10_ex_q75_"$PERIOD"_2020,sum_wpm2p5_ex_q75_"$PERIOD"_2020,sum_wno2_ex_q75_"$PERIOD"_2020,sum_wno_ex_q75_"$PERIOD"_2020,sum_wnh3_ex_q75_"$PERIOD"_2020,sum_wpans_ex_q75_"$PERIOD"_2020,sum_wnmvoc_ex_q75_"$PERIOD"_2020,sum_wo3_ex_q75_"$PERIOD"_2020,sum_wco_ex_q75_"$PERIOD"_2020,sum_wso2_ex_q75_"$PERIOD"_2020"

mkdir -p ./results/"$PERIOD"/without_lat_ex_q75/
mv rf_model_* ./results/"$PERIOD"/without_lat_ex_q75/
