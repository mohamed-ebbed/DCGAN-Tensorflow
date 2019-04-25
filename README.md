# DCGAN-Tensorflow

Implementation of [Deep Convolutional Generative Adverserial Network (DCGAN)](https://arxiv.org/abs/1511.06434) in Tensorflow with support for Tensorboard and configured to be trained on [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
### Generator 
![DCGAN net](https://cdn-images-1.medium.com/max/1200/1*rdXKdyfNjorzP10ZA3yNmQ.png)
### Disciminator
![Discriminator](http://bamos.github.io/data/2016-08-09/discrim-architecture.png)

## Requirements

Here are the requirements to run DCGAN-Tensorflow

- Python 2.7 or Python 3.3+
- Tensorflow >= 0.12
- Numpy
- tqdm
- json
- Bunch


## Usage

Edit the values in **config.json** file to fit your experiment as follows:

```json
{
    "exp_name" : "dcgan_small_64",
    "num_epochs": 20,
    "batch_size": 128,
    "img_size":64,
    "lr":0.0002,
    "beta1":0.5,
    "summaries_period":100, 
    "gf_dim":64,
    "df_dim":64,
    "data_dir": "../img_align_celeba",
    "noise_shape":100,
    "max_to_keep":3
}

```

**summaries_period**: How frequently summaries are written for example 20, then it will write summaries after every 20 steps.

**gf_dim**: Number of filters in the last layer in the Generator network.

**gf_dim**: Number of filters in the first layer in the Discriminator network.

**max_to_keep**: Maximum number of checkpoints kept in the checkpoints directory.

And then you run the code as follows:

```bash
python main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
