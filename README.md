# opencv-bird-detector

# Install

1. Install [Miniconda](https://conda.io/docs/user-guide/install/windows.html)
2. Download this repository and open it in Command Prompt
3. Create identical environment ([additional information](https://conda.io/docs/user-guide/tasks/manage-environments.html#building-identical-conda-environments))
```bash
conda create --name myenv --file env.txt
```
4. Activate environment ([additional information](https://conda.io/docs/user-guide/tasks/manage-environments.html#activating-an-environment))
```bash
activate myenv
```
or
```bash
source activate myenv
```

# Run

```bash
python main.py
```

Numeric keys *from 1 to 4* can be used to switch stages:

1. Source frame
2. Filtered frame
3. Foreground mask
4. Detected blobs frame

*Escape* to exit.

# License

[MIT](http://vjpr.mit-license.org)
