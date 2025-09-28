#!/bin/sh

#curl -v -X POST \
#  http://localhost:8000/api/text-to-wav \
#  -H "Content-Type: application/json" \
#  -d '{"text": "作为雨燕租车经理人"}' \
#  --output ./test_audios/itts_comb_test_10.wav
# http://chatbot.sharestyleai.com:8000/api/text-to-wav

curl -v -X POST \
  http://chatbot.sharestyleai.com:8000/api/text-to-wav\
  -H "Content-Type: application/json" \
  -d '{"text": "安排验车**：商家会指派验车人员，并与您协商确定具体的验车时间和地点。"}' \
 --output ./output_wav/np_05.wav