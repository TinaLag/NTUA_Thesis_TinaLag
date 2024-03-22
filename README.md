# How to install locally 
1) Open pycharm 
2) Set desired venv 
3) Cd to target package directory (it should have a setup.py inside)
4) run
```
$ python -m pip install -e .
```
5) Do
```
from thesis import arima 
```

# How to install from requirements.txt
Preparation:
- Method 1) 
```
$ pip freeze > requirements.txt
```
- Method 2) 
```
$ pip install pipreqs
$ pipreqs --force --encoding=utf-8 . 
```

Go to new venv and do 
```
$ pip install -r requirement.txt
```


# How to clone repository from git --through terminal (it can be done from menu)
1) Open pycharm and navigate to terminal
2) Clone repository using repository_url copied from github:
    ```
    $ git clone <repository_url>
    ```
3) Create and activate virtual environment:
   ```
   $ python -m venv venv_name    #venv_name = venv
   $ source venv/bin/activate
   ```
4) Navigate to the cloned project directory within terminal and install the required dependencies using pip:
   ```
   $ pip install -r requirements.txt
   ```
   If there's no requirements.txt file I need to install each dependence using pip install.
5) Open Project in Pycharm
6) SetUp interpreter through pycharm --here I use the interpreter from the virtual env I created (Python3.8(thesis))


# How to add changes to git after edit
```
$ git add .
$ git commit -m 'setup'          # 'setup' = name of edit
$ git push                       # asks for username and password (password = git token)
```

# They are asked for git commit if device hasn't been connected to git account before:
```
$ git config --global user.email "tina.lagodonti@gmail.com"
$ git config --global user.name "TinaLag"
```

