**从零实现多模态大模型,简单的方法！**

就是一个小玩具，我测了一下可以玩的起来的，需要的话两张3090完全可以的！一张3090需要改一下batch_size_per_gpu。会有点儿勉强！因为有些文字比较长！


数据集：

pretrain:

[预训练图像](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) [下载image.zip]


[预训练文字](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/tree/main/LLaVA-CC3M-Pretrain-595K) [下载chat-translated.json]

训练： CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 pretrain.py


sft阶段：


[微调图像和文字](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/tree/main)[下载sft_data.jsonl 和 sft_images.zip]

微调用我提供的脚步convert一下，变成share-gpt的格式！

微调： CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 sft_train.py

测试：

把 test.py 里面的图像和文字改一下就行。我这提供的是一个小老虎，没什么问题。


**从零实现大语言模型,简单的方法！**

可以自己写一个大语言模型并从头实现。我实现了基本的功能。并且可以完成一些日常的问答。代码在LLM_from_zero 文件夹中。模型是112M. 3090可以跑通！

数据集：

pretrain:

[预训练数据](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)[pretrain_hq.jsonl]


accelerate launch --num_processes=4 pretrain.py 

sft

[微调数据](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/files)[sft_data_zh.jsonl]

使用这个微调数据，dataset选择 **SFTDataset_** 这个类，因为这个数据集比较大，同时是多轮对话数据，只包含中文的文本。  

[微调数据可选](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)[sft_1024.jsonl]


使用这个微调数据，dataset选择 **SFTDataset** 这个类，数据集较小，但是中文效果可能没有那么好。  


accelerate launch --num_processes=4 sft_train.py 

强化学习后训练

updating


MOE 混合专家模型

updating


