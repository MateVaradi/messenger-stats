# Messenger Stats
Analyze your Messenger group chat data

## Usage

### Basic usage
1. Clone this repo.
2. Download your Messenger conversations from Facebook. [Here](https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/) is how to do it.
3. Among the downloaded files, find the group chat you want to analyze. Move the `.json` files of this conversation to `data/<my_chat>/`
4. Run the default pdf report yourself

`python analyze.py "my_chat"`

The resulting report will be in `results/<my_chat>/`.


### Advanced usage

[This notebook](https://github.com/MateVaradi/messenger-stats/blob/main/notebooks/analysis.ipynb) runs the plots from the pdf report one by one, but running them in Jupytter allows for some ad-hoc changes to the plots.

Alternatively, one can modify the plotting functions in `plotting.py` to achieve custom changes to the output.
