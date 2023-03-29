## Architecture

```python
train.py--datasets.py
		--algorithms.py
		--evaluate.py
		--strategy.py
```

`strategy.py` has not been implemented yet.

## `train.py`

- set the training parameters `args` 
- preprocessing for datasets based on the model
- training the model
- evaluate the model

## `datasets.py`

### Datasets

Our basic class is `assetDataset`, which returns traditional tabular data. And others are inherited from that, adding other items of data.

```python
assetDataset--imgDataset--imgOptDataset--imgOptVolDataset
			--          --             --
    					--			   --
        							   --
```

### Data generation

#### pixels

We calculate `maxValue - minValue` and divide them equally into `pSect` pieces. 

If the value `V` for a given series at point `P` falls in $pSect \cdot \frac{V-minValue}{maxValue-minValue}$ then we pixel the point $P_V$.

#### Range

For different window size, we take different `pSect` for the consideration of the larger volatility and range of value brought by larger window sizes.

#### series $\to$ picture

| pixel | value | meaning          |
| ----- | ----- | ---------------- |
| black | 0     | NULL             |
| white | 255   | curve this point |

#### Channels

Different channels might refers to different markets or different areas of the world, representing different predictive semantics.

#### Output pixels

##### Solution 1

> classification - cross-entropy - set the range of value in advance - use the mean value of a given range as the predicted value 

`mode = "clf"`

In our paper, we aim to predict the state of the asset in the future. We assume different state corresponds to different range of prices. And this could help the process of making trading decisions.

Feasibility: the Chinese stock market has `harden` restrictions, hence we could have a fixed range of the price of the given asset in the next stages. And we could divide the range to get a finite number of states.

##### Solution 2

> We predict the changing ratio of a given index, for example, Return, to avoid the  problem of in-equal `maxValue` and `minValue` of the data of a given window.

`mode = "reg"`

We predict the changing ratio of the target index.

#### Resize

We resize the picture to `[hparams['pic_size'],hparams['pic_size']]` for the training process of  `CNN` models.

## `algorithm.py`

### Ours

> ```python
> class SimpleCNN(nn.Module) # simple CNN methods
> class AECNN(nn.Module) # AutoEncoder construction
> ```

### Baselines

> ```python
> class AR # Autoregression
> class DNN(nn.Module) # ordinary deep neuro network
> ```

## Testing examples

There are some samples of running our codes to investigate the performance of different models on different datasets. 

```shell
# trained on single datasets using deep learning methods
python3 -m picAsset.train\
	    --data_dir ./picAsset/data/\
        --algorithm SimpleCNN\
        --dataset imgDataset\
        --pic_size 120\
        --window_size 60\
        --market 50SH

# fitted on single datasets using traditional methods
python3 -m picAsset.fit\
	    --data_dir ./picAsset/data/\
        --algorithm AR\
        --dataset assetDataset\
        --pic_size 240\
        --window_size 60\
        --market 600031       
        
# trained on joint datasets using deep learning methods
python3 -m picAsset.joint_train\
	    --data_dir ./picAsset/data/\
        --algorithm SimpleCNN\
        --dataset imgDataset\
        --pic_size 60\
        --window_size 5

# fitted on joint datasets using traditional methods
python3 -m picAsset.joint_fit\
	    --data_dir ./picAsset/data/\
        --algorithm AR\
        --dataset assetDataset\
        --pic_size 240\
        --window_size 60
```

