ViZDoom FPS AI based on A3C algorithm
---

### Scenarios

* Health pack gathering: easy and hard version
* Defend in the center
* Deadly corridor
* Death match
* D3_battle: scenario from IntelVCL team


### Solved Scenarios


Note that, except for deadly corridor and death match, A3C has gained not bad results on the rest scenarios


### Requirements


* python==3.5+
* vizdoom==1.1.3
* tensorflow==1.2
* numpy==1.13.1
* opencv-python==3.2.0
* scipy==0.19.0
* pygame==1.9.3

### Documents

#### The structure of the code

In every scenario, there is a `network.py` script, an `agent.py` script and a `main.py` file to run the code

All the configurations are in the `configs.py` file in each scenario. Or you can use the shell scripts to run it. 
The contents in both `configs.py` and shell scripts are self-explained.

The `check_point` directory stores checkpoint data of the neural network in each specific step, and the `summaries` 
directory stores events files contain data while helps developers to understand what's going on with the model only during training stage.
You can use `tensorboard` command to visualise the data, for example

    setsid tensorboard --logdir=D3_battle --port=7400 > logs.D3_battle 2>&1

Note that, if you want to train the model, no GUI will be prompted out since game window will not visible as you can see
 in the code `game.set_window_visible(self.play)`, when training `self.play` is `False`.

So, after training a model, how can I run it locally? You should set `IS_TRAIN` in `config.py` to `False` and set the `model_file` value, for instance `'model-30150.ckpt'`.

To get started, you can run the code of the code of healthpack gathering scenario to get familiar with the code.

If you have any problem, please open an issue!

The code of A3C framework was modified from [awjuliani's repo](https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb).

#### Some suggestions for developers

* Try `setsid` command to run the program in background, for example

    setsid python3 main.py > train.log 2>&1
    
* If you want to kill a running job on the server, use `pkill -TERM -P THE_PID` to kill all its children processes in stead of `kill -9 THE_PID`.

#### Some useful commands

Get the runing processes' directory

    $ for id in `ps -ef | grep main.py | awk '{print $2}'`; do ls -l /proc/${id}/cwd ; done    
    ls: cannot access /proc/10800/cwd: No such file or directory
    lrwxrwxrwx 1 doom doom 0 Aug 30 23:19 /proc/23963/cwd -> /home/doom/Code/battle_new
    lrwxrwxrwx 1 doom doom 0 Aug 29 19:30 /proc/38867/cwd -> /home/doom/Code/battle
    lrwxrwxrwx 1 doom doom 0 Sep  2 00:10 /proc/39261/cwd -> /home/doom/Code/battle_decay
    lrwxrwxrwx 1 doom doom 0 Sep  2 00:10 /proc/39706/cwd -> /home/doom/Code/battle_new_2

Get the utilization of GPU for every 5 seconds

    while true; do nvidia-smi --query-gpu=utilization.memory --format=csv && nvidia-smi --query-gpu=utilization.gpu --format=csv; sleep 5; done
    
#### Videos

You can watch the videos of the results on [Youtube](https://www.youtube.com/channel/UCn_UdbGa4DaxBsOYNGMfnHg?view_as=subscriber).


#### Some helpful links

* [tensorflow-how-can-i-restore-model-when-some-new-layers-are-added](https://stackoverflow.com/questions/45487323/tensorflow-how-can-i-restore-model-when-some-new-layers-are-added)
* [tensorflow-why-there-are-3-files-after-saving-the-model](https://stackoverflow.com/questions/41265035/tensorflow-why-there-are-3-files-after-saving-the-model)
* [python3-importerror-no-module-named-tkinter-on-ubuntu](https://stackoverflow.com/questions/44237302/python3-importerror-no-module-named-tkinter-on-ubuntu)
* [after-building-tensorflow-from-source-seeing-libcudart-so-and-libcudnn-errors](https://stackoverflow.com/questions/42013316/after-building-tensorflow-from-source-seeing-libcudart-so-and-libcudnn-errors/44147506#44147506)
* [TensorFlow](https://www.tensorflow.org)
