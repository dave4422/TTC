# bin_sup_con_learning


`python train.py experiment=contrastive experiment/specs=butterfly batch_size=8 module.optimizer_name=sgd`


### running experiemnts

#### data

##### data source
data.data_module._args_.0="/local_ssd/projects/iNat21/"

##### changeing augementations

++train_transform._args_.0.color_jitter=[0.8,0.8,0.8,0.2]

##### changeing normalization

#### controlling model and loss function


## build bolts from source

`ip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade`