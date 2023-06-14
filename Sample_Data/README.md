Referring to Stanford's HELM, a query sent to the model consists of three parts: instruction, train examples, and test input.

The format of the prompt can be modified by adjusting the instruction and various prefixes to adapt to different tasks. It is also possible to modify the decoding parameters to specify the maximum length of the output and the token to stop decoding at.

Specifically, our own data format can follow the JSON specification as follows:
``` JSON
{
  "adapter_spec": {//This part is about defining the prompt format and decoding parameters, which need to be provided when creating the dataset.
    "instructions": "",//Task Description
    "input_prefix": "Passage: ",
    "input_suffix": "\n",
    "reference_prefix": "A. ",//If it's not a multiple-choice question, there's no need to mark it.
    "reference_suffix": "\n",
    "output_prefix": "Answer: ",
    "output_suffix": "\n",
    "instance_prefixw": "\n",
    "max_train_instances": 5,//Number of prompt examples.
    "max_eval_instances": 1000,//Amount of data to be evaluated.
    "max_tokens": 5,//Maximum length limit of model output.
    "stop_sequences": [
      "\n"// If encountering this token, the model needs to stop decoding.
    ],
    "decoding_parameters": {//If you need to create decoding parameters, please write them here.
      "temperature": 1,
      },
    "ouput_format": "list", // If it is set as a list here, our request completions will have segmented results. If it is set as a string, our request completions will only have one result, taking xxx["completions"][0]["text"] as the original output of our model group, without being segmented.
  },
  "request_states": [
    {
      "instance": {// a single instance
        "input": {// input text
          "text": "Daniel went to the kitchen.\nSandra went back to the kitchen.\nDaniel moved to the garden.\nSandra grabbed the apple.\nSandra went back to the office.\nSandra dropped the apple.\nSandra went to the garden.\nSandra went back to the bedroom.\nSandra went back to the office.\nMary went back to the office.\nDaniel moved to the bathroom.\nSandra grabbed the apple.\nSandra travelled to the garden.\nSandra put down the apple there.\nMary went back to the bathroom.\nDaniel travelled to the garden.\nMary took the milk.\nSandra grabbed the apple.\nMary left the milk there.\nSandra journeyed to the bedroom.\nJohn travelled to the office.\nJohn went back to the garden.\nSandra journeyed to the garden.\nMary grabbed the milk.\nMary left the milk.\nMary grabbed the milk.\nMary went to the hallway.\nJohn moved to the hallway.\nMary picked up the football.\nSandra journeyed to the kitchen.\nSandra left the apple.\nMary discarded the milk.\nQuestion: Where was the apple before the garden?"// question text
        },
        "references": [//Refer to the options or the correct answer. If it is a multiple-choice question, there will be multiple references, but only the options that are correct (taking into account multiple correct answers) will have the tag "correct". If it is not a multiple-choice question, there will only be one standard answer reference.
          {
            "output": {
              "text": "bedroom" // answer text
            },
            "tags": [
              "correct"// If it is a multiple-choice question, there will be a "correct" tag, which indicates that this option is correct.
            ]
          }
        ],
        "split": "test",
        "id": "id1295"
      },//Those who provides data only needs to write their instance here. If the key "request" exists in the JSON, it means that the evaluation team has already run it. The evaluation script should retrieve the results from under this key to calculate the score.
      "request":{//This is where the output of the evaluation team is located, and the output results can be found under "completions" in the "text" section of the "result".
        "result": {
        "success": true,
        "completions": [
          {
            "text": " office",// this is the output of the model
            "logprob": -1.4051814,
            "tokens": [// each token
              {
                "text": " office",
                "logprob": -1.4051814,
              }
            ]
          }
        ],
        "cached": true,
        "request_time": 1.622053623199463,
        "request_datetime": 1669584580
      },
      }
    } 
  ]
}

```

Note:

- There is no need to differentiate between train and eval instances when providing data, as long as the total is 5 (demonstration) + 100 (test examples) or more. We will sample during running.
- If CoT examples need to be written, train and test need to be differentiated. In such cases, indicate in the "split" key under "references" and write the CoT logic in the "text" key under "input."
- For multiple-choice questions, an answer_mapping needs to be provided, as in Holistic Evaluation of Language Models (HELM).