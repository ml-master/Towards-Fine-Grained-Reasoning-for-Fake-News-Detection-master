## FinerFact

### !! Unofficial  implementation

This is the PyTorch implementation for the FinerFact model in the AAAI 2022 paper Towards Fine-Grained Reasoning for Fake News Detection

### MutualReinforce

the first part of the framework need  needs to be implemented from scratch, so the finally result will be different from the original.

### Finally result

![image-20240703102452196](C:\Users\m\AppData\Roaming\Typora\typora-user-images\image-20240703102452196.png)

### Dataset

[FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

- `gossipcop_fake.csv` - Samples related to fake news collected from GossipCop

- `gossipcop_real.csv` - Samples related to real news collected from GossipCop

### Instruction to run code

* Take politifact as an example. Make sure you have put the following training and test files under `data/`. 
  * `Train_bert-base-cased_politifact_130_5.pt`
  * `Test_bert-base-cased_politifact_130_5.pt` 
* If the `Train_*.pt` and `Test_*.pt` files are not present, you can run `preprocess/preprocess.py` to split the training data (e.g. `bert-base-cased_politifact_130_5.pt`) into `Train_*.pt` and `Test_*.pt`. You can download the data [here](https://drive.google.com/drive/folders/1gyTsMHDCSEbHLE-PgfTgOPeXhK_uZlWS?usp=sharing)
* Download the files for pretrained BERT model and put them under `bert_base/`. You should have the following 3 files in `bert_base/`:
  * `pytorch_model.bin`
  * `vocab.txt`
  * `bert_config.json`
* make sure you have set the `root` path given by `get_root_dir()` in `utils/utils` to your own data path of `fake_news_data/`. Mine is `root = "C:\\Workspace\\FakeNews\\fake_news_data"` on Windows and `root = "../../fake_news_data"`
* run the `train.py` file using `kgat/` as the working directory:
  * `python train.py --outdir . --config_file P.ini`, or
  * `python train.py --outdir . --config_file G.ini`



