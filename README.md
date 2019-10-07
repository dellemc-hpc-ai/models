# Neural Machine Translation

This implementation extends the tensorflow's official implementation of the transformer model for neural machine translation. Provides way to perform distributed training using horovod library. Multi-head attention block used in the transformer architecture. At a high-level, the scaled dot-product attention can be thought as finding the relevant information, in the form of values (V) based on Query (Q) and Keys (K). Multi-head attention can be thought of as several attention layers in parallel, which together can identify distinct aspects of the input.


## Getting Started

Clone the files to a lcoal  directory or mounted directory in case of nauta.  

```
# install required python libraries in case of bare metal
pip3 install --user -r official/requirements.txt
```

Note: Check the setup steps under official/README.md for more details.


## General Instructions

Ensure that the tensorflow-models directory is appended to the PYTHONPATH correctly.

This can be done by updating the sys.path in multiclass_horovod_optimizers.py.

```
sys.path.append('/path/to/tensorflow-models')
```

Using Nauta:

```
sys.path.append('/mnt/output/home/tensorflow-models')
```


## Data

The training dataset used is the [WMT English-German] (http://data.statmt.org/wmt17/translation-task/) parallel corpus, which contains 4.5M English-German sentence pairs.

## Train

Train using 4 nodes and 4 processes each.

Export the required environment variables. This needs to be be udpated in templates values.yaml in case of nauta.

```
export OMP_NUM_THREADS=9
export KMP_BLOCKTIME=0
```

Set APP_DIR, DATA_DIR and MODEL_DIR paths accordingly.

```
mpiexec --hostfile $hostfile -cpus-per-proc 9 --map-by socket --oversubscribe -n 16 -x OMP_NUM_THREADS -x KMP_BLOCKTIME python ${APP_DIR}/official/transformer/transformer_main_hvd.py --data_dir=${DATA_DIR} --model_dir=${MODEL_DIR} --vocab_file=${DATA_DIR}/vocab.ende.32768 --param_set=big --steps_between_evals=1000 --train_steps=20000 --batch_size=5000 --intra_op=18 --inter_op=1 --num_parallel_calls=18 --learning_rate=0.001 --learning_rate_warmup_steps=1000 --max_length=256 --vocab_size=33945 --save_checkpoints_secs=3600 --log_step_count_steps=20 --lr_scheme=2 --warmup_init_lr=1e-07 --layer_postprocess_dropout=0.3
```

Using Nauta:

Update the paths to transformer_main_hvd.py, data_dir, model_dir, vocab_file accordingly.

```
nctl experiment submit --name nmt-transformer-test -t multinode-tf-training-horovod /path/to/tensorflow-models/official/transformer/transformer_main_hvd.py -- --data_dir=/mnt/output/home/data/nmt/nmt_data --model_dir=/mnt/output/experiment --vocab_file=/mnt/output/home/data/nmt/nmt_data/vocab.ende.32768 --param_set=big --steps_between_evals=100 --train_steps=100 --batch_size=5000 --intra_op=18 --inter_op=1 --num_parallel_calls=18 --learning_rate=0.001 --learning_rate_warmup_steps=1000 --max_length=256 --vocab_size=33945 --save_checkpoints_secs=3600 --log_step_count_steps=20 --lr_scheme=2 --warmup_init_lr=1e-07 --layer_postprocess_dropout=0.3

```

## Create translation files

update paths accordingly.

```
python /path/to/tensorflow-models/official/transformer/translate.py --model_dir=/path/to/nmt-transformer-test --vocab_file=/path/to/data/nmt/nmt_data/vocab.ende.32768 --param_set=big --file=/path/to/tensorflow-models/official/transformer/test_data/newstest2014.en --file_out=/path/to/translation.en
```

Using Nauta:

```
nctl experiment submit --name nmt-translate-single -t tf-training-tfjob /path/to/tensorflow-models/official/transformer/translate.py -- --model_dir=/mnt/output/home/nmt-transformer-test --vocab_file=/mnt/output/home/data/nmt/nmt_data/vocab.ende.32768 --param_set=big --file=/mnt/output/home/tensorflow-models/official/transformer/test_data/newstest2014.en --file_out=/mnt/output/experiment/translation.en
```

## Eval

Evaluate the model by loading the reference and translation files and calculate BLEU scores.

Update paths to compute_bleu.py, reference and translation accordingly.

```
python /path/to/tensorflow-models/official/transformer/compute_bleu.py
--reference=/path/to/tensorflow-models/official/transformer/test_data/newstest2014.de --translation=/path/to/translation.en
```


Using Nauta:

Update paths to compute_bleu.py, reference and translation accordingly.

```
nctl experiment submit --name nmt-compute-bleu -t tf-training-tfjob /path/to/tensorflow-models/official/transformer/compute_bleu.py -- --reference=/mnt/output/home/tensorflow-models/official/transformer/test_data/newstest2014.de --translation=/mnt/output/home/nmt-translate-single-1/translation.en

```


## Related articles
  
	https://arxiv.org/abs/1905.04035

	https://community.emc.com/community/products/rs_for_ai/blog/2019/04/02/scaling-nmt-with-intel-xeon-scalable-processors

	https://community.emc.com/community/products/rs_for_ai/blog/2019/04/02/scaling-nmt-challenges-and-solution
    
	https://community.emc.com/community/products/rs_for_ai/blog/2019/04/02/effectiveness-of-large-batch-training-for-neural-machine-translation-nmt




# Original Tensorflow Models README

# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)
