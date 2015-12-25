### How to run (Centos 7)

1. Install the native dependencies for compiling application dependencies (Centos 7):
  ```
  $ sudo yum update -y
  $ sudo yum groupinstall -y 'development tools'
  $ sudo yum install -y libjpeg-devel lapack-devel openssl-devel bzip2-devel sqlite-devel readline-devel libpng-devel freetype-devel wget
  ```

2. Install OpenCV using this tutorial : http://docs.opencv.org/master/dd/dd5/tutorial_py_setup_in_fedora.html

3. Install Python 2.7.10 using pyenv:
  ```
  $ git clone https://github.com/yyuu/pyenv.git ~/.pyenv
  $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
  $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
  $ echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
  $ source ~/.bash_profile
  $ pyenv install 2.7.10
  $ pyenv global 2.7.10
  ```

4. Install pip:
  ```
  $ wget https://bootstrap.pypa.io/get-pip.py
  $ python get-pip.py
  ```

5. Clone the repo:
  ```
  $ git clone https://github.com/panggi/assignment-itb-if5181.git
  $ cd assignment-itb-if5181
  ```

6. Initialize and activate a virtualenv:
  ```
  $ pip install virtualenv
  $ virtualenv --no-site-packages env
  $ source env/bin/activate
  ```

7. Install the application dependencies:
  ```
  $ pip install -r requirements.txt
  ```

8. Run the development server:
  ```
  $ python app.py
  ```

9. Navigate to [http://localhost:5181](http://localhost:5181)
