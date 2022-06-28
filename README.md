# KnowBERT_huggingface
This is the code of KnowBERT, re-writed by Huggingface. We only rewrite the code of pretraining wiki linker, where `knowbert_wiki_linker.jsonnet` is used as configs.
It contains the data processing part and the KnowBERT model part, you can run each file (`data_utils.py` or `modeling.py`) for better understanding. Note that, we didn't implement the running part.


### 1. code of `data_utils.py`, `modeling.py`
> 1.1 preparing data (Downloading from `knowbert_wiki_linker.jsonnet`).
> `aida_train.txt`, `aida_dev.txt` is used for training or evaluating the pre-trained entity linker.
> `prob_yago_crosswikis_wikipedia_p_e_m.txt`, and `wiki_id_to_string.json`


### 2.  Reference materials
> 2.1 [Allennlp document](http://docs.allennlp.org/v0.9.0/api/allennlp.commands.html)  
> 2.2 [An example](https://github.com/wj-Mcat/allennlp-tutorials/blob/master/code-tutorials/01-simple-lstm-tagger/train.py)  
> 2.3 [The framework of Allennlp](https://zhuanlan.zhihu.com/p/110362086)  

### 3. Some questions I encountered.
> 3.1 How is a class instantiated and how are parameters passed from `XXX.jsonnet` file?  
> A class is instantiated according to the decorator, the parameters are passed by a default function `_from_parameters`, refer to Allennlp doc for details.  

> 3.2 How to  execute the functions in class?  
> After instantiating Trainer, executing `Trainer.train()` is going to execute the model (e.g., loading data, training model). It

> 3.3 How to read training dataset?  
> training file -> XXXReader->text_to_read->Instances->_read->Iterable object->batch->model.


### 4.  Some sketches
> see `sketches.pdf`.


