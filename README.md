# **Problem**

*Generate a dataset of 400 2-dimensional data vectors which consists of four groups
of 100 data vectors. The four groups are modeled by Gaussian distributions with
means $m _{1}=\begin{pmatrix}0&0\end{pmatrix}^{T}$ , $m _{2}=\begin{pmatrix}4&0\end{pmatrix}^{T}$ , $m _{3}=\begin{pmatrix}0&4\end{pmatrix}^{T}$ , $m _{4}=\begin{pmatrix}5&4\end{pmatrix}^{T}$ , respectively, and covariance matrices $S _{1} = I$ , $S_{2}=\begin{pmatrix}1&0.2\\0.2&1.5\end{pmatrix}$ , $S_{3}=\begin{pmatrix}1&0.4\\0.4&1.1\end{pmatrix}$ , $S_{4}=\begin{pmatrix}0.3&0.2\\0.2&0.5\end{pmatrix}$ , respectively.Plot the data vectors. Measure the Euclidean distance between any two data points and determine maximum* ($d_{max}$) *and minimum* ($d_{min}$) *Euclidean distances.*

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