# Learning to Select Cuts for Efficient Mixed-Integer Programming

This is the code for implementing Cut Ranking presented in the paper:
[Learning to Select Cuts for Efficient Mixed-Integer Programming](https://arxiv.org/abs/2105.13645).

It is based on the open-source MIP solver Python-MIP:
[python-mip](https://www.python-mip.com/).

## Installation
- To install Python-MIP, run: 
```
pip install mip
```
- Main Python Dependencies: Python (3.6), tensorflow (1.x)

## Quick Start

### Collect Training Data
To collect training bags and labels, set cut_choice_method to "CollectLabels" (Line 183 of collect_labels.py) and set the problem_idx (Line 174 of collect_labels.py), and run:
```shell
python collect_labels.py
 ```
### Load Training Data 
To test loading the data, set the correct training data path (Line 15 of data_loader.py), and run:
```shell
python data_loader.py
 ```
 
 ### Train the Cut Ranking Model
 To train the model, set the saved model path (Line 66 of train.py), and run: 
 ```shell
python train.py
 ```
 
 ### Test on new instances 
 To test the trained model, you can set the path of trained models in choose_cut.py (Line 312-313), and set cut_choice_method to "Test" (Line 183 of collecet_labels.py), and run:
  ```shell
python collect_labels.py
 ```

## Paper citation

If you used this code for your experiments or found it helpful, consider citing the following paper:

<pre>
@misc{huang2021learning,
      title={Learning to Select Cuts for Efficient Mixed-Integer Programming}, 
      author={Zeren Huang and Kerong Wang and Furui Liu and Hui-ling Zhen and Weinan Zhang and Mingxuan Yuan and Jianye Hao and Yong Yu and Jun Wang},
      year={2021},
      eprint={2105.13645},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

</pre>
