import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    AdaLoraConfig,
    AdaLoraModel,
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter
import numpy as np
import random
import copy
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.nn.functional import normalize

print(f"RANK: {os.environ.get('RANK')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

# rank = int(os.environ["RANK"])
# local_rank = int(os.environ["LOCAL_RANK"])
# world_size = int(os.environ["WORLD_SIZE"])
# cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
# assigned_gpu = cuda_visible_devices[local_rank]
# os.environ["CUDA_VISIBLE_DEVICES"] = assigned_gpu
# torch.cuda.set_device(0)
# print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# # Set the CUDA device for this process
# # torch.cuda.set_device(local_rank)

# # Initialize the process group
# dist.init_process_group(
#     backend="nccl",
#     init_method="env://",
#     world_size=world_size,
#     rank=rank,
# )

# print(f"RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")
# print(f"Using GPU: {torch.cuda.current_device()}")



def fl_finetune(
        # model/data params
        global_model: str = 'huggyllama/llama-7b',
        data_path: str = './data',
        output_dir: str = './fedgpt-llama7b-5-2/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 5,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 128,  # 64,
        local_micro_batch_size: int = 16,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        # aggregation mode
        stacking: bool = False,
        florist: bool = False,
        threshold: int = 0.9,
        # evaluation
        dev_data_path: str = './mmlu_test_1444.jsonl',
        # heterogeneous
        heter: bool = False,
        local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4],
        zero_padding: bool = False,
        Adalora: bool = False,
        full: bool = False
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"threshold: {threshold}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        print("DDP is activated")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if global_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            use_auth_token=True,
        )
    elif global_model == 'meta-llama/Llama-3.2-1B' or global_model == 'meta-llama/Llama-2-7b-hf':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token="your_token",
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            token="your_token",
        )

    if global_model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b' or global_model == 'meta-llama/Llama-3.2-1B' or global_model == 'meta-llama/Llama-2-7b':
        tokenizer = AutoTokenizer.from_pretrained(global_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(global_model, token="your_token",)

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if data_path == './data/8':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["context"],
                data_point["response"],
            )
        elif data_path == './data_wiz/10' or data_path == './data_mix/20' or data_path == './data_wiz/8':
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["output"],
            )
        else:
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    #model = prepare_model_for_int8_training(model)
    if full == False:
        if florist:
            config = LoraConfig(
                    base_model_name_or_path=global_model,
                    r = lora_r,
                    lora_alpha = lora_alpha,
                    target_modules = lora_target_modules,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                )
            # model = get_peft_model(model, config)
        elif stacking:
            if heter:
                config_ori = LoraConfig(
                base_model_name_or_path=global_model,
                r = sum(local_ranks),
                lora_alpha = lora_alpha * num_clients,
                target_modules = lora_target_modules,
                lora_dropout = lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )
            else:
                config_ori = LoraConfig(
                    base_model_name_or_path=global_model,
                    r = lora_r * num_clients,
                    lora_alpha = lora_alpha * num_clients,
                    target_modules = lora_target_modules,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                )
        else:
            if zero_padding:
                config_ori = LoraConfig(
                    base_model_name_or_path=global_model,
                    r = max(local_ranks),
                    lora_alpha = lora_alpha * max(local_ranks),
                    target_modules = lora_target_modules,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                )
                # model = get_peft_model(model, config_ori)
            else:
                config = LoraConfig(
                    base_model_name_or_path=global_model,
                    r = lora_r,
                    lora_alpha = lora_alpha,
                    target_modules = lora_target_modules,
                    lora_dropout = lora_dropout,
                    bias = "none",
                    task_type = "CAUSAL_LM",
                )
                model = get_peft_model(model, config)

    print("Number of available GPUs:")
    print(f"Rank: {os.environ.get('RANK')}, Node GPUs: {torch.cuda.device_count()}")


    if not ddp and torch.cuda.device_count() > 1:
        print("***********")
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []
    total_rank_comm = 0
    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)
        # new_output_dir = os.path.join("/groups/jdass/", output_dir)
        for client_id in selected_clients_set:
            if full == False:
                if Adalora:
                    config = AdaLoraConfig(
                        r=local_ranks[client_id],
                        lora_alpha=2*local_ranks[client_id],
                        target_modules=lora_target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        base_model_name_or_path=global_model,
                    )
                    model_client = copy.deepcopy(model)
                    model_client = get_peft_model(model_client, config)
                else:
                    if stacking or florist:
                        if heter:
                            config = LoraConfig(
                            r=local_ranks[client_id],
                            lora_alpha=2*local_ranks[client_id],
                            target_modules=lora_target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            base_model_name_or_path=global_model,
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                        else:
                            config = LoraConfig(
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            target_modules=lora_target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            base_model_name_or_path=global_model,
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)

                    else:
                        if heter:
                            config = LoraConfig(
                            r=local_ranks[client_id],
                            lora_alpha=2*local_ranks[client_id],
                            target_modules=lora_target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            base_model_name_or_path=global_model,
                            )
                            model_client = copy.deepcopy(model)
                            model_client = get_peft_model(model_client, config)
                        else:
                            model_client = model

            else:
                model_client = model

            client = GeneralClient(client_id, model_client, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model_client, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        #local_dataset_len_dict = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

        model, total_rank, total_param = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       stacking,
                       florist,
                       threshold,
                       lora_r,
                       heter,
                       local_ranks,
                       zero_padding,
                       full
                       )
        total_rank_comm += total_rank
        new_output_dir = os.path.join("/groups/jdass/", output_dir)


        if full == False:
            if florist:
                new_weights = torch.load(os.path.join(new_output_dir, str(epoch), 'florist', 'adapter_model.bin'))
                for name, param in model_client.named_parameters():
                    key = name.replace(".default", "")
                    param_device = param.device
                    print(key)
                    if key in new_weights:
                        print("Yes")
                        param.data = new_weights[key]
                        param.data = param.data.to(param_device)
                print("Reduced weights successfully loaded into the model.")


            elif stacking:
                config_ori.save_pretrained(
                    os.path.join(new_output_dir, str(epoch), 'flora'),
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
                model = PeftModel.from_pretrained(model, os.path.join(new_output_dir, str(epoch), 'flora'))

            else:
                out_path = os.path.join(new_output_dir, str(epoch), 'fedit')
                # if heter:
                #     os.makedirs(out_path,exist_ok=True)
                #     # torch.save(model.state_dict(), os.path.join(out_path, "adapter_model.bin"))
                #     config_ori.save_pretrained(
                #         os.path.join(out_path),
                #         load_in_8bit=False,
                #         torch_dtype=torch.float16,
                #         device_map=device_map,
                #     )
                # else:
                #     os.makedirs(out_path,exist_ok=True)
                #     # torch.save(model.state_dict(), os.path.join(out_path, "adapter_model.bin"))
                #     config.save_pretrained(
                #         os.path.join(out_path),
                #         load_in_8bit=False,
                #         torch_dtype=torch.float16,
                #         device_map=device_map,
                #     )
                # # model = PeftModel.from_pretrained(model, out_path)
                new_weights = torch.load(os.path.join(new_output_dir, str(epoch), 'fedit', 'adapter_model.bin'))
                for name, param in model.named_parameters():
                    key = name.replace(".default", "")
                    param_device = param.device
                    print(key)
                    if key in new_weights:
                        print("Yes")
                        param.data = new_weights[key]
                        param.data = param.data.to(param_device)
        else:
            config = AutoConfig.from_pretrained(global_model)
            tokenizer.save_pretrained(os.path.join(new_output_dir, str(epoch)),
                    load_in_8bit=False,
                    torch_dtype=torch.float32,
                    device_map=device_map,)
            config.save_pretrained(os.path.join(new_output_dir, str(epoch)),
                    load_in_8bit=False,
                    torch_dtype=torch.float32,
                    device_map=device_map,)

            print('save model')


        '''x_dir = os.path.join(output_dir, str(epoch))
        current_dir = x_dir # + "/temp/"
        print(current_dir)'''
        #arc_easy,hellaswag,mmlu,truthfulqa
        #os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_easy,hellaswag,mmlu,truthfulqa --device cuda --output_path {current_dir}".format(current_dir = current_dir))
        #os.system("lm_eval --model_args pretrained={current_dir},parallelize=True,load_in_4bit=False --tasks arc_easy,hellaswag,mmlu,truthfulqa --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
        if stacking:
            model = model.merge_and_unload()
        elif florist:
            model = model_client.merge_and_unload()
            # model.save_pretrained(os.path.join(output_dir, str(epoch) + '/final'),
            #         load_in_8bit=False,
            #         torch_dtype=torch.float32,
            #         device_map=device_map,)

        # if epoch < (num_communication_rounds - 1):
        #     rm_dir = os.path.join(output_dir, str(epoch))
        #     os.system("rm -rf {xxxxx}".format(xxxxx = rm_dir))

    else:
        acc = global_evaluation(model, tokenizer, prompter, dev_data_path)

    print('Acc of Epoch', str(epoch), 'is:', acc)
    acc_list.append(acc)

    print(acc_list)
    #os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = new_output_dir + 'log.txt'
    file = open(filename,'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') + "; Total rank across all rounds: " + str(total_rank_comm) + "; Total parameters: " + str(total_param) + '\n'
        if florist:
            log_text = "florist-" + str(global_model) + ", Threshold-" + str(threshold) + ":\n"

        elif stacking:
            log_text = "FLoRA-" + str(global_model) + ":\n"

        else:
            log_text = "FedIT-" + str(global_model) + ":\n"

        file.write(log_text)
        file.write(s)
    file.close()
    print("Log Saved")

if __name__ == "__main__":
    fire.Fire(fl_finetune)