# **Problem**

*Generate a dataset of 400 2-dimensional data vectors which consists of four groups
of 100 data vectors. The four groups are modeled by Gaussian distributions with
means* ```m1 = [0,0]```, ```m2 = [4,0]```, ```m3 = [0,4]```, ```m4 = [5,4]``` , *respectively, and covariance matrices* ```S1 = I```, ```S2 = [[1, 0.2], [0.2, 1.5]]```, ```S3 = [[1, 0.4], [0.4, 1.1]]``` , ```S4 = [[0.3, 0.2], [0.2, 0.5]]``` , *respectively.Plot the data vectors. Measure the Euclidean distance between any two data points and determine maximum* (```dmax```) *and minimum* (```dmin```) *Euclidean distances.*

<br><br>

### 1. Create virtual environment
```bash
python3 -m venv env
```

### 2. Activate virtual environment
```bash
source env/bin/activate
```

### 3. Install required dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Run main file
```bash
python3 main.py
```