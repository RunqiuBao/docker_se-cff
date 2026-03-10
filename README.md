# Trainer repo for rfdetr

## Quick start
### prepare dataset
check Bao's desktop, there are these folders at `/mydata/mujin_datasets/`
```
jp_post
├── generate_dataset.py
├── train
│   ├── seq0
│   │   ├── ***.mip
│   │   ├── ***.json
│   ├── seq1
│   ├── ...
│   ├── seqN
├── valid
│   ├── ...
├── test
│   ├── ...
jp_post_generated
│   ├── seq0
│   │   ├── datalist.txt
│   │   ├── imagedata
│   │   │   ├── ***.pkl
│   │   ├── objdet
│   │   │   ├── debug/
│   │   │   ├── annotations.xml
│   │   │   ├── images
│   ├── seq1
│   ├── ...
│   ├── seqN
│   ├── views4annotate
│   │   ├── seq0
│   │   ├── ...
│   │   ├── seqN
├── valid
│   ├── ...
├── test
│   ├── ...
```
- put the mip and json into `seqX/` folders
- run `python3 generate_dataset.py --seq seqX --rotate` to generate data into `jp_post_generated`.
- I use cvat for data annotations, create tasks for each seqX like this:
![create_tasks](.readme/create_tasks.png)
- annotate
![annotate](.readme/annotate.png)
- and export into dataset. remember to check `SaveImages`.
![export_task](.readme/export_task.png)
- put the exported dataset into each folder in seqX and unzip:
![objdet](.readme/objdet.png)
- now enable the seqX in `/home/mujin/nvme/code/3rdpartycode/docker_pytorch_trainnn/src/components/datasets/mujinextended/constant.py`. When data are not enough, I am using same train sequences for validation. But ideally if there are enough data, valid. should have its own sequences.

### train model
- in my desktop, use `dce` in bash and enter `mujinenv_dockerpytorch` env for training.
- cd `/root/code/docker_pytorch_trainnn/script/`
- `./distributed_main.sh` to start training.

### export to onnx
- similarly as `train model`, uncomment the `--only_test` and `--is_save_onnx` option to do onnx export.

## License

MIT license.
