从零实现多模态大模型,简单的方法！

就是一个小玩具，我测了一下可以玩的起来的，需要的话两张3090完全可以的！一张3090需要改一下batch_size_per_gpu。会有点儿勉强！因为有些文字比较长！


数据集：

pretrain:

[预训练图像](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) [下载image.zip]


[预训练文字](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/tree/main/LLaVA-CC3M-Pretrain-595K) [下载chat-translated.json]


sft阶段：


[微调图像和文字](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/tree/main)[下载sft_data.jsonl 和 sft_images.zip]

微调用我提供的脚步convert一下，变成share-gpt的格式！
