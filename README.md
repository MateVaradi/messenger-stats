# Messenger Stats
Analyze your Messenger group chat data

## Overview

This repository contains code to report some interesting statistics about group conversations. The report currently includes the following statistics:
- messaging volume per user (in terms of the number of messages sent and the number of words sent),
- average daily usage statistics per user (i.e. average number of messages sent per hour of the day),
- historic usage statistics per user (i.e. number of messages sent per month),
- most commonly sent and received emoji reactions per user,
- number of messages with a minimum number of reactions of a certain emoji per user,
- sample messages with a minimum number of reactions of a certain emoji,
- network of reactions of a certain emoji. The width of a directed edge from member *a* to *b* represent the relative proportion of member *a*'s reactions that are sent to member *b*.

## Usage

### Basic usage
1. Clone this repo.
2. Download your Messenger conversations from Facebook. [Here](https://www.zapptales.com/en/download-facebook-messenger-chat-history-how-to/) is how to do it.
3. Among the downloaded files, find the group chat you want to analyze. Move the `.json` files of this conversation to `data/<my_chat>/`
4. Run the default pdf report yourself

`python analyze.py "my_chat"`

The resulting pdf report will be in `results/<my_chat>/`.


### Advanced usage

[This notebook](https://github.com/MateVaradi/messenger-stats/blob/main/notebooks/analysis.ipynb) runs the plots from the pdf report one by one. Running them in Jupyter allows for some ad-hoc changes to the plots.

Alternatively, one can modify the plotting functions in `plotting.py` to achieve custom changes to the output.
