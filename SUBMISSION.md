## Summary of Submission Related Information

1. The data I used for testing are from:
    [news dataset](https://huggingface.co/datasets/stanford-oval/ccnews/tree/main) for positive labels and [common crawl](https://huggingface.co/datasets/agentlans/common-crawl-sample/tree/main/en) for negative labels
   
    I have not added the data to this repository due to its size, but you can download it to `data/raw` directory.
   
    The processed data is stored in `data/processed`
2. I used the script from `scripts/create_datasets.py` to create the .txt files that fasttext can consume
3. The requirements.txt contains all packages required for installing 
4. You could score single inputs of text, or batch (with labels) to get validation/test performance using the apis
5. Run the app using `python app.py`
6. You can get more information from `README.md` or `docs/api_usage.md`
7. I also have a dockerfile for this app that you can refer to. However, it is much faster to run training outside the docker image (unless you want to allocate more resources by editing the `docker/docker-compose.yml` file).

## Fasttext Related Information
1. The model will automatically use data that I have pre-processed from common-crawl (and stored in `data/processed/trained_negative.txt`) regardless of what you upload as the positive set.
2. I have enabled ability to autotune hyperparameters using fasttext autotune api which will tune the lr, word-ngram and char-ngram, etc. But you will need a val dataset for the same.
3. You can get performance on a batch of unseen data by uploading the val/test dataset to the batch score api.

## Usage of LLM-Coding Assistant
I used Claude Code (3.5 sonnet) for translating some of my design ideas into code.